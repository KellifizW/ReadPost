"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
修復輸入驗證過嚴問題，確保廣泛查詢（如「分析吹水台時事主題」）進入分析流程。
修復 'prioritize' 錯誤，強化 prioritize_threads_with_grok 的錯誤處理。
修復 list_titles 意圖未正確生成標題列表問題，確保僅列出帖子標題。
修復 stream_grok3_response 錯誤使用總結模板問題，限制 metadata 並強化提示詞。
主要函數：
- analyze_and_screen：分析問題，識別意圖，放寬語義要求，動態設置篩選條件。
- stream_grok3_response：生成流式回應，動態選擇模板，確保 list_titles 僅列出標題。
- process_user_question：處理用戶問題，抓取帖子並生成總結。
- clean_html：清理 HTML 標籤。
"""

import aiohttp
import asyncio
import json
import re
import random
import datetime
import time
import logging
import streamlit as st
import os
import hashlib
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器（簡化版）
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 自定義日誌格式器，將時間戳設為香港時區
class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

# 控制台處理器：輸出到 stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90  # 秒

# 定義全局錯誤類別
class AppError(Exception):
    def __init__(self, message, user_message=None):
        self.message = message
        self.user_message = user_message or message
        super().__init__(self.message)

class PromptBuilder:
    """
    提示詞生成器，從 prompts.json 載入模板並動態構建提示詞。
    """
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "prompts.json")
        
        logger.info(f"Attempting to load prompts.json from: {config_path}")
        
        if not os.path.exists(config_path):
            raise AppError(f"prompts.json not found at: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.config = json.loads(content)
                logger.info(f"Loaded prompts.json successfully")
        except json.JSONDecodeError as e:
            raise AppError(f"JSON parse error in prompts.json: {e}")
        except Exception as e:
            raise AppError(f"Failed to load prompts.json: {str(e)}")

    def build_analyze(self, query, cat_name, cat_id, conversation_context=None, thread_titles=None, metadata=None, thread_data=None):
        config = self.config["analyze"]
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        data = config["data"].format(
            thread_titles=json.dumps(thread_titles or [], ensure_ascii=False),
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False)
        )
        prompt = f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        return prompt

    def build_prioritize(self, query, cat_name, cat_id, threads):
        config = self.config.get("prioritize", None)
        if not config:
            raise AppError("Prompt configuration for 'prioritize' not found")
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id
        )
        data = config["data"].format(
            threads=json.dumps(threads, ensure_ascii=False)
        )
        prompt = f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        return prompt

    def build_response(self, intent, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        config = self.config["response"].get(intent, self.config["response"]["general"])
        context = config["context"].format(
            query=query,
            selected_cat=selected_cat,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        data = config["data"].format(
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False),
            filters=json.dumps(filters or {}, ensure_ascii=False)
        )
        prompt = f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        return prompt

    def get_system_prompt(self, mode):
        return self.config["system"].get(mode, "")

def clean_html(text):
    """
    清理 HTML 標籤，規範化文本。
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}")
        return text

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，識別意圖，放寬語義要求，確保廣泛查詢進入分析流程。
    """
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
    category_keywords = ["吹水台", "熱門台", "時事台", "娛樂台", "科技台"]
    theme_keywords = ["時事", "新聞", "熱話", "政治", "經濟", "社會", "國際", "娛樂", "科技", "on9", "搞亂", "無聊"]
    is_category_related = any(keyword in user_query for keyword in category_keywords)
    is_theme_related = any(keyword in user_query.lower() for keyword in theme_keywords)
    is_popular_query = "熱門" in user_query
    
    prompt = prompt_builder.build_analyze(
        query=user_query,
        cat_name=cat_name,
        cat_id=cat_id,
        conversation_context=conversation_context,
        thread_titles=thread_titles,
        metadata=metadata,
        thread_data=thread_data
    )
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        raise AppError(f"Grok 3 API key missing: {str(e)}", user_message="缺少 API 密鑰，請聯繫支持。")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": prompt_builder.get_system_prompt("analyze")},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.7
    }
    
    logger.info(f"Starting intent analysis for query: {user_query}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    response_text = await response.text()
                    if status_code != 200:
                        logger.warning(f"Intent analysis failed: status={status_code}, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        raise AppError(
                            f"API request failed with status {status_code}",
                            user_message="無法連接到分析服務，請稍後重試。"
                        )
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Intent analysis failed: missing choices, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        raise AppError(
                            "Invalid API response: missing choices",
                            user_message="分析服務返回無效數據，請稍後重試。"
                        )
                    
                    result = json.loads(data["choices"][0]["message"]["content"])
                    
                    if is_category_related or is_theme_related:
                        result["direct_response"] = False
                        result["intent"] = "summarize_posts"
                        result["theme"] = "on9" if "on9" in user_query.lower() else ("時事" if is_theme_related else "general")
                        result["filters"] = {"min_replies": 20, "min_likes": 5}
                        result["theme_keywords"] = ["on9", "搞亂", "無聊"] if "on9" in user_query.lower() else (theme_keywords if is_theme_related else [])
                        result["needs_advanced_analysis"] = is_theme_related or "on9" in user_query.lower()
                        result["post_limit"] = 10
                        result["reply_limit"] = 50
                        result["category_ids"] = [cat_id]
                        result["data_type"] = "both"
                        result["processing"] = "summarize"
                    
                    if is_popular_query:
                        result["filters"] = {
                            "min_replies": 5,
                            "min_likes": 2,
                            "sort": "popular"
                        }
                        result["post_limit"] = 10
                        result["reply_limit"] = 50
                        result["intent"] = "summarize_posts"
                        result["theme"] = "general"
                        result["data_type"] = "both"
                        result["processing"] = "summarize"
                    
                    result.setdefault("direct_response", False)
                    result.setdefault("intent", "summarize_posts")
                    result.setdefault("theme", "general")
                    result.setdefault("category_ids", [cat_id])
                    result.setdefault("data_type", "both")
                    result.setdefault("post_limit", 5)
                    result.setdefault("reply_limit", 50)
                    result.setdefault("filters", {"min_replies": 20, "min_likes": 5})
                    result.setdefault("processing", "summarize")
                    result.setdefault("candidate_thread_ids", [])
                    result.setdefault("top_thread_ids", [])
                    result.setdefault("needs_advanced_analysis", False)
                    result.setdefault("reason", "")
                    result.setdefault("theme_keywords", [])
                    logger.info(f"Intent analysis completed: intent={result['intent']}, theme={result['theme']}")
                    return result
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.warning(f"Intent analysis error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise AppError(
                f"Analysis failed after {max_retries} attempts: {str(e)}",
                user_message="分析服務暫時不可用，請稍後重試。"
            )

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id):
    """
    使用 Grok 3 根據問題語義排序帖子，返回最相關的帖子ID。
    強化錯誤處理，確保始終返回字典格式，修復 'prioritize' 錯誤。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        raise AppError(f"Grok 3 API key missing: {str(e)}", user_message="缺少 API 密鑰，請聯繫支持。")

    prompt_builder = PromptBuilder()
    try:
        prompt = prompt_builder.build_prioritize(
            query=user_query,
            cat_name=cat_name,
            cat_id=cat_id,
            threads=[{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads]
        )
    except Exception as e:
        raise AppError(f"Failed to build prioritize prompt: {str(e)}", user_message="無法生成排序提示，請稍後重試。")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Thread prioritization failed: status={response.status}, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        raise AppError(
                            f"API request failed with status {response.status}",
                            user_message="無法完成帖子排序，請稍後重試。"
                        )
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Thread prioritization failed: missing choices, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        raise AppError(
                            "Invalid API response: missing choices",
                            user_message="排序服務返回無效數據，請稍後重試。"
                        )
                    
                    content = data["choices"][0]["message"]["content"]
                    logger.info(f"Raw API response for prioritization: {content}")
                    try:
                        result = json.loads(content)
                        if not isinstance(result, dict) or "top_thread_ids" not in result or "reason" not in result:
                            logger.warning(f"Invalid prioritization result format: {content}")
                            raise AppError(
                                "Invalid result format: missing required keys",
                                user_message="排序結果格式錯誤，請稍後重試。"
                            )
                        logger.info(f"Thread prioritization succeeded: {result}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse prioritization result as JSON: {content}, error: {str(e)}")
                        if content.strip() == "prioritize":
                            raise AppError(
                                "Prompt configuration error: API returned 'prioritize'",
                                user_message="排序配置錯誤，請聯繫支持。"
                            )
                        raise AppError(
                            f"Failed to parse API response as JSON: {str(e)}",
                            user_message="無法解析排序結果，請稍後重試。"
                        )
        except Exception as e:
            logger.warning(f"Thread prioritization error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise AppError(
                f"Prioritization failed after {max_retries} attempts: {str(e)}",
                user_message="排序服務暫時不可用，請稍後重試。"
            )

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None):
    """
    使用 Grok 3 API 生成流式回應，根據意圖動態選擇模板，確保 list_titles 僅列出標題。
    修復 list_titles 未正確使用 response.list 模板問題，限制 metadata 僅包含 thread_id 和 title。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 20, "min_likes": 5}
    prompt_builder = PromptBuilder()
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    # 明確設置 intent，確保與 analysis 一致
    intent = processing if isinstance(processing, str) else processing.get('intent', 'summarize')
    
    # 當 intent 為 list_titles 時，僅使用簡化 metadata（僅 thread_id 和 title），忽略其他字段
    if intent == "list_titles":
        thread_data = {}  # 清空 thread_data，避免傳遞回覆
        simplified_metadata = [
            {"thread_id": item["thread_id"], "title": item["title"]}
            for item in metadata
        ]
        metadata = simplified_metadata  # 替換原始 metadata
        logger.info(f"Intent set to list_titles, using simplified metadata with only thread_id and title")
    
    # 限制回覆數，防止提示詞過長（僅非 list_titles 意圖）
    max_replies_per_thread = 20
    filtered_thread_data = {}
    if intent != "list_titles":
        for tid, data in thread_data.items():
            replies = data.get("replies", [])
            sorted_replies = sorted(
                [r for r in replies if r.get("msg")],
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:max_replies_per_thread]
            
            if not sorted_replies and replies:
                logger.warning(f"No valid replies for thread_id={tid}, using raw replies")
                sorted_replies = replies[:max_replies_per_thread]
            
            filtered_thread_data[tid] = {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": sorted_replies,
                "fetched_pages": data.get("fetched_pages", [])
            }
    
    # 若無回覆且有 metadata，回退到僅使用 metadata（僅非 list_titles 意圖）
    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata and intent != "list_titles":
        logger.warning(f"Filtered thread data has no replies, using metadata for summary")
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": [],
                "fetched_pages": data.get("fetched_pages", [])
            } for tid, data in thread_data.items()
        }
    
    # 構建提示詞，明確使用 intent，並添加明確指令
    prompt = prompt_builder.build_response(
        intent=intent,
        query=user_query,
        selected_cat=selected_cat,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=filtered_thread_data,
        filters=filters
    )
    
    # 為 list_titles 添加明確指令，確保僅生成標題列表
    if intent == "list_titles":
        prompt += "\n\n**明確指令**：僅列出帖子標題和 ID，格式必須為：\n- 帖子 ID: [thread_id] 標題: [title]\n不包含任何總結、回覆數、點讚數或其他資訊，僅限最多10個帖子。"
        logger.info("Added explicit instruction for list_titles to enforce title-only output")
    
    # 檢查提示詞長度
    prompt_length = len(prompt)
    if prompt_length > GROK3_TOKEN_LIMIT:
        max_replies_per_thread = max_replies_per_thread // 2
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": data["replies"][:max_replies_per_thread],
                "fetched_pages": data.get("fetched_pages", [])
            } for tid, data in filtered_thread_data.items()
        }
        prompt = prompt_builder.build_response(
            intent=intent,
            query=user_query,
            selected_cat=selected_cat,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=filtered_thread_data,
            filters=filters
        )
        if intent == "list_titles":
            prompt += "\n\n**明確指令**：僅列出帖子標題和 ID，格式必須為：\n- 帖子 ID: [thread_id] 標題: [title]\n不包含任何總結、回覆數、點讚數或其他資訊，僅限最多10個帖子。"
        logger.info(f"Truncated prompt: original_length={prompt_length}, new_length={len(prompt)}")
    
    # 若提示詞過短，簡化數據（僅非 list_titles 意圖）
    if prompt_length < 500 and intent in ["summarize", "sentiment"]:
        logger.warning(f"Prompt too short, retrying with simplified data")
        simplified_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "like_count": data.get("like_count", 0),
                "replies": data["replies"][:5]
            } for tid, data in filtered_thread_data.items()
        }
        prompt = prompt_builder.build_response(
            intent=intent,
            query=user_query,
            selected_cat=selected_cat,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=simplified_thread_data,
            filters=filters
        )
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": prompt_builder.get_system_prompt("response")},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": True
    }
    
    logger.info(f"Starting response generation for query: {user_query}, intent: {intent}")
    
    response_content = ""
    async with aiohttp.ClientSession() as session:
        try:
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        status_code = response.status
                        if status_code != 200:
                            response_text = await response.text()
                            logger.warning(f"Response generation failed: status={status_code}, attempt={attempt + 1}")
                            if attempt < 2:
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            raise AppError(
                                f"API request failed with status {status_code}",
                                user_message=f"API 請求失敗（狀態碼 {status_code}）。請稍後重試。"
                            )
                        
                        async for line in response.content:
                            if line and not line.isspace():
                                line_str = line.decode('utf-8').strip()
                                if line_str == "data: [DONE]":
                                    break
                                if line_str.startswith("data: "):
                                    try:
                                        chunk = json.loads(line_str[6:])
                                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if content:
                                            if "###" in content and ("Content Moderation" in content or "Blocked" in content):
                                                logger.warning(f"Content moderation detected: {content}")
                                                raise AppError(
                                                    "Content moderation detected",
                                                    user_message="回應內容被過濾，請嘗試其他查詢。"
                                                )
                                            response_content += content
                                            yield content
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON decode error in stream chunk: {str(e)}")
                                        continue
                        if not response_content:
                            logger.warning(f"No content generated, attempt={attempt + 1}")
                            if attempt < 2:
                                simplified_thread_data = {
                                    tid: {
                                        "thread_id": data["thread_id"],
                                        "title": data["title"],
                                        "no_of_reply": data.get("no_of_reply", 0),
                                        "like_count": data.get("like_count", 0),
                                        "replies": data["replies"][:5]
                                    } for tid, data in filtered_thread_data.items()
                                }
                                prompt = prompt_builder.build_response(
                                    intent=intent,
                                    query=user_query,
                                    selected_cat=selected_cat,
                                    conversation_context=conversation_context,
                                    metadata=metadata,
                                    thread_data=simplified_thread_data,
                                    filters=filters
                                )
                                if intent == "list_titles":
                                    prompt += "\n\n**明確指令**：僅列出帖子標題和 ID，格式必須為：\n- 帖子 ID: [thread_id] 標題: [title]\n不包含任何總結、回覆數、點讚數或其他資訊，僅限最多10個帖子。"
                                payload["messages"][-1]["content"] = prompt
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            if metadata:
                                fallback_prompt = prompt_builder.build_response(
                                    intent="summarize",
                                    query=user_query,
                                    selected_cat=selected_cat,
                                    conversation_context=conversation_context,
                                    metadata=metadata,
                                    thread_data={},
                                    filters=filters
                                )
                                payload["messages"][-1]["content"] = fallback_prompt
                                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as fallback_response:
                                    if fallback_response.status == 200:
                                        data = await fallback_response.json()
                                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                        if content:
                                            response_content = content
                                            yield content
                                            return
                            response_content = "無法生成詳細總結，可能是數據不足。以下是討論區的通用概述：討論涵蓋時事、娛樂等多主題，網民觀點多元。"
                            yield response_content
                            return
                        logger.info(f"Response generation completed: length={len(response_content)}")
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    logger.warning(f"Response generation error: {str(e)}, attempt={attempt + 1}")
                    if attempt < 2:
                        max_replies_per_thread = max_replies_per_thread // 2
                        filtered_thread_data = {
                            tid: {
                                "thread_id": data["thread_id"],
                                "title": data["title"],
                                "no_of_reply": data.get("no_of_reply", 0),
                                "last_reply_time": data.get("last_reply_time", 0),
                                "like_count": data.get("like_count", 0),
                                "dislike_count": data.get("dislike_count", 0),
                                "replies": data["replies"][:max_replies_per_thread],
                                "fetched_pages": data.get("fetched_pages", [])
                            } for tid, data in filtered_thread_data.items()
                        }
                        prompt = prompt_builder.build_response(
                            intent=intent,
                            query=user_query,
                            selected_cat=selected_cat,
                            conversation_context=conversation_context,
                            metadata=metadata,
                            thread_data=filtered_thread_data,
                            filters=filters
                        )
                        if intent == "list_titles":
                            prompt += "\n\n**明確指令**：僅列出帖子標題和 ID，格式必須為：\n- 帖子 ID: [thread_id] 標題: [title]\n不包含任何總結、回覆數、點讚數或其他資訊，僅限最多10個帖子。"
                        payload["messages"][-1]["content"] = prompt
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    raise AppError(
                        f"Response generation failed: {str(e)}",
                        user_message=f"生成回應失敗（{str(e)}）。請稍後重試。"
                    )
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
        finally:
            await session.close()

def clean_cache(max_age=3600):
    """
    清理過期緩存數據，防止記憶體膨脹。
    """
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]

def unix_to_readable(timestamp):
    """
    將 Unix 時間戳轉換為香港時區的普通時間格式（YYYY-MM-DD HH:MM:SS）。
    """
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert timestamp {timestamp}: {str(e)}")
        return "1970-01-01 00:00:00"

async def process_user_question(user_question, selected_cat, cat_id, order, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    """
    處理用戶問題，分階段抓取並分析 LIHKG 帖子。
    修復 'prioritize' 錯誤，強化錯誤處理。
    """
    try:
        logger.info(f"Processing user question: {user_question}, category: {selected_cat}, order: {order}")
        
        clean_cache()
        
        if rate_limit_until > time.time():
            logger.warning(f"Rate limit active until {rate_limit_until}")
            raise AppError(
                f"Rate limit active until {rate_limit_until}",
                user_message=f"速率限制中，請在 {datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} 後重試。"
            )
        
        if progress_callback:
            progress_callback("正在抓取帖子列表", 0.1)
        
        post_limit = min(analysis.get("post_limit", 5), 20)
        reply_limit = analysis.get("reply_limit", 50)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 20)
        min_likes = filters.get("min_likes", 5)
        sort_method = filters.get("sort", "popular")
        time_range = filters.get("time_range", "recent")
        top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
        previous_thread_ids = previous_thread_ids or []
        
        intent = analysis.get("intent", "summarize_posts")
        
        thread_data = []
        rate_limit_info = []
        initial_threads = []
        
        for page in range(1, 4):  # 抓取 3 頁
            result = await get_lihkg_topic_list(
                cat_id=cat_id,
                order=order,
                start_page=page,
                max_pages=1,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until
            )
            request_counter = result.get("request_counter", request_counter)
            last_reset = result.get("last_reset", last_reset)
            rate_limit_until = result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(result.get("rate_limit_info", []))
            items = result.get("items", [])
            for item in items:
                item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
            initial_threads.extend(items)
            if not items:
                logger.warning(f"No threads fetched for cat_id={cat_id}, order={order}, page={page}")
            if len(initial_threads) >= 135:  # 最多 135 個帖子
                initial_threads = initial_threads[:135]
                break
            if progress_callback:
                progress_callback(f"已抓取第 {page}/3 頁帖子", 0.1 + 0.2 * (page / 3))
        
        if progress_callback:
            progress_callback("正在篩選帖子", 0.3)
        
        filtered_items = []
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            no_of_reply = item.get("no_of_reply", 0)
            like_count = int(item.get("like_count", 0))
            
            if no_of_reply >= min_replies and like_count >= min_likes and thread_id not in previous_thread_ids:
                filtered_items.append(item)
        
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            if thread_id not in st.session_state.thread_cache:
                st.session_state.thread_cache[thread_id] = {
                    "data": {
                        "thread_id": thread_id,
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item["last_reply_time"],
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [],
                        "fetched_pages": []
                    },
                    "timestamp": time.time()
                }
        
        # 根據 intent 處理 fetch_dates 意圖，跳過不必要的排序
        if intent == "fetch_dates":
            if progress_callback:
                progress_callback("正在處理日期相關資料", 0.4)
            sorted_items = sorted(
                filtered_items,
                key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                reverse=True
            )
            top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
            logger.info(f"Fetch dates mode, selected threads: {top_thread_ids}")
        else:
            # 其他意圖（如 summarize_posts）才執行優先排序
            if not top_thread_ids and filtered_items:
                if progress_callback:
                    progress_callback("正在重新分析帖子選擇", 0.4)
                prioritization = await prioritize_threads_with_grok(user_question, filtered_items, selected_cat, cat_id)
                if not isinstance(prioritization, dict):
                    logger.error(f"Invalid prioritization result: {prioritization}")
                    prioritization = {"top_thread_ids": [], "reason": "Invalid prioritization result"}
                top_thread_ids = prioritization.get("top_thread_ids", [])
                if not top_thread_ids:
                    logger.warning(f"Prioritization failed: {prioritization.get('reason', 'Unknown reason')}")
                    # 回退機制：按熱門程度排序
                    sorted_items = sorted(
                        filtered_items,
                        key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                        reverse=True
                    )
                    top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
                    logger.info(f"Fallback to popularity sorting: {top_thread_ids}")
                else:
                    logger.info(f"Grok prioritized threads: {top_thread_ids}")
            
            if not top_thread_ids and filtered_items:
                if sort_method == "popular" or order == "hot":
                    sorted_items = sorted(
                        filtered_items,
                        key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                        reverse=True
                    )
                else:
                    sorted_items = sorted(
                        filtered_items,
                        key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                        reverse=(time_range == "recent")
                    )
                top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
                logger.info(f"Generated top_thread_ids: {top_thread_ids}")
        
        if progress_callback:
            progress_callback("正在抓取候選帖子內容", 0.5)
        
        candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
        if not candidate_threads and filtered_items:
            candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items)))
            logger.info(f"No candidate threads, using random: {len(candidate_threads)} threads selected")
        
        for idx, item in enumerate(candidate_threads):
            thread_id = str(item["thread_id"])
            cache_key = thread_id
            cache_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
            
            if cache_data and cache_data.get("replies") and cache_data.get("fetched_pages"):
                thread_data.append(cache_data)
                continue
            
            if progress_callback:
                progress_callback(f"正在抓取帖子 {idx + 1}/{len(candidate_threads)}", 0.5 + 0.3 * ((idx + 1) / len(candidate_threads)))
            
            content_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit,
                fetch_last_pages=0,
                specific_pages=None,
                start_page=1
            )
            
            request_counter = content_result.get("request_counter", request_counter)
            last_reset = content_result.get("last_reset", last_reset)
            rate_limit_until = content_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(content_result.get("rate_limit_info", []))
            
            if content_result.get("replies"):
                thread_info = {
                    "thread_id": thread_id,
                    "title": content_result.get("title", item["title"]),
                    "no_of_reply": item.get("no_of_reply", content_result.get("total_replies", 0)),
                    "last_reply_time": item["last_reply_time"],
                    "like_count": item.get("like_count", 0),
                    "dislike_count": item.get("dislike_count", 0),
                    "replies": [
                        {
                            "post_id": reply.get("post_id"),
                            "msg": clean_html(reply.get("msg", "")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                        } for reply in content_result["replies"] if reply.get("msg")
                    ],
                    "fetched_pages": content_result.get("fetched_pages", [])
                }
                thread_data.append(thread_info)
                
                st.session_state.thread_cache[cache_key] = {
                    "data": thread_info,
                    "timestamp": time.time()
                }
            else:
                logger.warning(f"No replies fetched for thread_id={thread_id}")
        
        if analysis.get("needs_advanced_analysis", False) and thread_data:
            if progress_callback:
                progress_callback("正在進行進階分析", 0.8)
            
            advanced_analysis = await analyze_and_screen(
                user_query=user_question,
                cat_name=selected_cat,
                cat_id=cat_id,
                thread_titles=[item["title"] for item in thread_data],
                metadata=[{
                    "thread_id": item["thread_id"],
                    "title": item["title"],
                    "no_of_reply": item["no_of_reply"],
                    "like_count": item["like_count"],
                    "dislike_count": item["dislike_count"]
                } for item in thread_data],
                thread_data={item["thread_id"]: item for item in thread_data},
                is_advanced=True,
                conversation_context=conversation_context
            )
            
            thread_data = [
                item for item in thread_data
                if str(item["thread_id"]) in map(str, advanced_analysis.get("top_thread_ids", []))
            ]
            analysis.update(advanced_analysis)
        
        if not thread_data and st.session_state.thread_cache:
            logger.warning("No thread data, attempting cache recovery")
            thread_data = [
                cache["data"] for cache in st.session_state.thread_cache.values()
                if cache["data"].get("replies")
            ][:post_limit]
        
        if progress_callback:
            progress_callback("完成數據處理", 0.9)
        
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        fallback_thread_data = [
            cache["data"] for cache in st.session_state.thread_cache.values()
            if cache["data"].get("replies") and cache["data"]["thread_id"] in top_thread_ids
        ]
        raise AppError(
            f"Processing failed: {str(e)}",
            user_message=f"處理問題失敗：{str(e)}。請稍後重試或聯繫支持。"
        )
