"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
支援並行抓取、動態回覆抓取、語義嵌入意圖識別、上下文記憶及新意圖分類。
主要函數：
- analyze_and_screen：分析問題，識別意圖，支援語義嵌入。
- stream_grok3_response：生成流式回應，動態選擇模板。
- process_user_question：處理用戶問題，支援並行抓取與動態回覆。
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

# 配置日誌記錄器
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

# 控制台處理器
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90  # 秒

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
            logger.error(f"prompts.json not found at: {config_path}")
            raise FileNotFoundError(f"prompts.json not found at: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.config = json.loads(content)
                logger.info(f"Loaded prompts.json successfully")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in prompts.json: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load prompts.json: {str(e)}")
            raise

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
            logger.error("Prompt configuration for 'prioritize' not found in prompts.json")
            raise ValueError("Prompt configuration for 'prioritize' not found")
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

def unix_to_readable(unix_timestamp):
    """
    將 Unix 時間戳轉換為可讀格式（香港時區）。
    """
    try:
        timestamp = int(unix_timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        logger.warning(f"Invalid timestamp: {unix_timestamp}")
        return str(unix_timestamp)

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，使用語義嵌入識別意圖，支援新意圖分類與上下文記憶。
    """
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
    # 簡單關鍵詞觸發新意圖
    if "搜尋" in user_query or "找" in user_query or "關於" in user_query:
        keywords = re.findall(r'\w+', user_query)[:3]
        return {
            "direct_response": False,
            "intent": "search_keywords",
            "theme": keywords[0] if keywords else "general",
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 50,
            "filters": {"min_replies": 20, "min_likes": 5, "keywords": keywords},
            "processing": "search_keywords",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": True,
            "reason": f"Detected keyword search: {keywords}",
            "theme_keywords": keywords
        }
    if "推薦" in user_query or "建議" in user_query:
        return {
            "direct_response": False,
            "intent": "recommend_threads",
            "theme": "recommend",
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 50,
            "filters": {"min_replies": 50, "min_likes": 10, "sort": "popular"},
            "processing": "recommend_threads",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": True,
            "reason": "Detected recommendation request",
            "theme_keywords": []
        }
    if "跟進" in user_query or "監控" in user_query or "更新" in user_query:
        return {
            "direct_response": False,
            "intent": "monitor_events",
            "theme": "event_monitoring",
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 50,
            "filters": {"min_replies": 20, "min_likes": 5, "sort": "recent"},
            "processing": "monitor_events",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": True,
            "reason": "Detected event monitoring request",
            "theme_keywords": []
        }
    if "意見" in user_query or "立場" in user_query or "觀點" in user_query:
        return {
            "direct_response": False,
            "intent": "classify_opinions",
            "theme": "opinion_classification",
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 100,
            "filters": {"min_replies": 20, "min_likes": 5},
            "processing": "classify_opinions",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": True,
            "reason": "Detected opinion classification request",
            "theme_keywords": []
        }

    # 語義嵌入意圖識別
    candidate_intents = ["summarize_posts", "analyze_sentiment", "fetch_dates", "search_keywords", "recommend_threads", "monitor_events", "classify_opinions", "general_query"]
    context_summary = ""
    if conversation_context:
        for msg in conversation_context[-4:]:  # 最近兩輪
            context_summary += f"{msg['role']}: {msg['content']}\n"
    prompt = f"""
    你是意圖識別助手，考慮對話歷史分析問題意圖：
    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id}）
    對話歷史：{context_summary}
    候選意圖：{', '.join(candidate_intents)}
    輸出格式：{{
        "intent": "選定的意圖",
        "confidence": 0.0-1.0,
        "reason": "選擇原因"
    }}
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": "",
            "category_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "reply_limit": 0,
            "filters": {},
            "processing": "general",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "Missing API key",
            "theme_keywords": []
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    logger.info(f"Starting intent analysis for query: {user_query}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code != 200:
                        logger.warning(f"Intent analysis failed: status={status_code}, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {
                            "direct_response": False,
                            "intent": "summarize_posts",
                            "theme": "general",
                            "category_ids": [cat_id],
                            "data_type": "both",
                            "post_limit": 5,
                            "reply_limit": 50,
                            "filters": {"min_replies": 20, "min_likes": 5},
                            "processing": "summarize",
                            "candidate_thread_ids": [],
                            "top_thread_ids": [],
                            "needs_advanced_analysis": False,
                            "reason": f"API request failed with status {status_code}",
                            "theme_keywords": []
                        }
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Intent analysis failed: missing choices, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {
                            "direct_response": False,
                            "intent": "summarize_posts",
                            "theme": "general",
                            "category_ids": [cat_id],
                            "data_type": "both",
                            "post_limit": 5,
                            "reply_limit": 50,
                            "filters": {"min_replies": 20, "min_likes": 5},
                            "processing": "summarize",
                            "candidate_thread_ids": [],
                            "top_thread_ids": [],
                            "needs_advanced_analysis": False,
                            "reason": "Invalid API response: missing choices",
                            "theme_keywords": []
                        }
                    
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intent = result["intent"] if result["confidence"] > 0.7 else "summarize_posts"
                    reason = result["reason"]
                    return {
                        "direct_response": intent == "general_query",
                        "intent": intent,
                        "theme": "general",
                        "category_ids": [cat_id],
                        "data_type": "both",
                        "post_limit": 10,
                        "reply_limit": 50,
                        "filters": {"min_replies": 20, "min_likes": 5},
                        "processing": intent.replace("_posts", ""),
                        "candidate_thread_ids": [],
                        "top_thread_ids": [],
                        "needs_advanced_analysis": intent not in ["list_titles", "general_query"],
                        "reason": reason,
                        "theme_keywords": []
                    }
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.warning(f"Intent analysis error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return {
                "direct_response": False,
                "intent": "summarize_posts",
                "theme": "general",
                "category_ids": [cat_id],
                "data_type": "both",
                "post_limit": 5,
                "reply_limit": 50,
                "filters": {"min_replies": 20, "min_likes": 5},
                "processing": "summarize",
                "candidate_thread_ids": [],
                "top_thread_ids": [],
                "needs_advanced_analysis": False,
                "reason": f"Analysis failed after {max_retries} attempts: {str(e)}",
                "theme_keywords": []
            }

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id):
    """
    使用 Grok 3 根據問題語義排序帖子，返回最相關的帖子ID。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"top_thread_ids": [], "reason": "Missing API key"}

    logger.debug(f"Prioritizing {len(threads)} threads, sample_types: {[type(t.get('no_of_reply')) for t in threads[:3]]}")
    
    prompt_builder = PromptBuilder()
    # 規範化 threads 數據
    normalized_threads = [
        {
            "thread_id": t["thread_id"],
            "title": t["title"],
            "no_of_reply": int(t.get("no_of_reply", 0)),
            "like_count": int(t.get("like_count", 0))
        }
        for t in threads
    ]
    
    try:
        prompt = prompt_builder.build_prioritize(
            query=user_query,
            cat_name=cat_name,
            cat_id=cat_id,
            threads=normalized_threads
        )
    except Exception as e:
        logger.error(f"Failed to build prioritize prompt: {str(e)}")
        return {"top_thread_ids": [], "reason": f"Prompt building failed: {str(e)}"}
    
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
                        return {"top_thread_ids": [], "reason": f"API request failed with status {response.status}"}
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Thread prioritization failed: missing choices, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {"top_thread_ids": [], "reason": "Invalid API response: missing choices"}
                    
                    content = data["choices"][0]["message"]["content"]
                    logger.debug(f"Raw API response for prioritization: {content}")
                    try:
                        result = json.loads(content)
                        if not isinstance(result, dict) or "top_thread_ids" not in result or "reason" not in result:
                            logger.warning(f"Invalid prioritization result format: {content}")
                            return {"top_thread_ids": [], "reason": "Invalid result format: missing required keys"}
                        logger.info(f"Thread prioritization succeeded: {result}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse prioritization result as JSON: {content}, error: {str(e)}")
                        return {"top_thread_ids": [], "reason": f"Failed to parse API response as JSON: {str(e)}"}
        except Exception as e:
            logger.warning(f"Thread prioritization error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return {"top_thread_ids": [], "reason": f"Prioritization failed after {max_retries} attempts: {str(e)}"}

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None):
    """
    使用 Grok 3 API 生成流式回應，根據意圖和分類動態選擇模板。
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
    
    max_replies_per_thread = 20
    filtered_thread_data = {}
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
    
    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata:
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
    
    intent = processing.get('intent', 'summarize') if isinstance(processing, dict) else processing
    
    prompt = prompt_builder.build_response(
        intent=intent,
        query=user_query,
        selected_cat=selected_cat,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=filtered_thread_data,
        filters=filters
    )
    
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
        logger.info(f"Truncated prompt: original_length={prompt_length}, new_length={len(prompt)}")
    
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
    
    logger.info(f"Starting response generation for query: {user_query}")
    
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
                            yield f"錯誤：API 請求失敗（狀態碼 {status_code}）。請稍後重試。"
                            return
                        
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
                                                raise ValueError("Content moderation detected")
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
                            response_content = "無法生成詳細總結，可能是數據不足。以下是吹水台的通用概述：吹水台討論涵蓋時事、娛樂等多主題，網民觀點多元。"
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
                        payload["messages"][-1]["content"] = prompt
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試。"
                    return
        except Exception as e:
            logger.error(f"Unexpected error in response generation: {str(e)}")
            yield f"錯誤：生成回應時發生意外錯誤（{str(e)}）。請稍後重試。"
            return

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    """
    處理用戶問題，執行帖子篩選、並行抓取與動態回覆抓取。
    """
    thread_data = []
    rate_limit_info = []
    current_time = time.time()
    
    logger.debug(f"Processing query: {user_question}, intent={analysis.get('intent')}, cat_id={cat_id}")
    
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.datetime.now(HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} - Rate limit active until {datetime.datetime.fromtimestamp(rate_limit_until, HONG_KONG_TZ)}")
        logger.warning(f"Rate limit active until {datetime.datetime.fromtimestamp(rate_limit_until, HONG_KONG_TZ)}")
        return {
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    if current_time - last_reset >= 60:
        request_counter = 0
        last_reset = current_time
    
    # 動態回覆抓取：決定頁數與時段
    prompt = f"""
    根據問題判斷抓取回覆的頁數（1-10頁）與時段（head：前幾頁，tail：後幾頁，avg：平均分佈）：
    問題：{user_question}
    意圖：{analysis.get('intent')}
    輸出：{{
        "pages": 1-10,
        "segment": "head/tail/avg",
        "reason": "判斷原因"
    }}
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        pages, segment = 5, "head"
    else:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json={"model": "grok-3-beta", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}) as response:
                if response.status == 200:
                    data = await response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    pages = max(1, min(10, int(result["pages"])))  # 確保整數
                    segment = result["segment"]
                    logger.debug(f"Dynamic page selection: pages={pages}, segment={segment}")
                else:
                    logger.warning(f"Dynamic page selection failed: status={response.status}")
                    pages, segment = 5, "head"
    
    # 抓取帖子列表
    if progress_callback:
        progress_callback("正在抓取帖子列表", 0.2)
    
    topic_result = await get_lihkg_topic_list(
        cat_id=cat_id,
        request_counter=request_counter,
        last_reset=last_reset,
        rate_limit_until=rate_limit_until
    )
    request_counter = topic_result.get("request_counter", request_counter)
    last_reset = topic_result.get("last_reset", last_reset)
    rate_limit_until = topic_result.get("rate_limit_until", rate_limit_until)
    rate_limit_info.extend(topic_result.get("rate_limit_info", []))
    candidate_threads = topic_result.get("items", [])
    
    if not candidate_threads:
        logger.warning(f"No threads found for cat_id={cat_id}")
        return {
            "thread_data": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    logger.debug(f"Candidate threads: {len(candidate_threads)}, sample_types: {[type(t.get('no_of_reply')) for t in candidate_threads[:3]]}, sample_total_pages: {[t.get('total_pages') for t in candidate_threads[:3]]}")
    
    # 篩選帖子
    filters = analysis.get("filters", {"min_replies": 20, "min_likes": 5})
    min_replies = int(filters.get("min_replies", 20))  # 確保整數
    min_likes = int(filters.get("min_likes", 5))      # 確保整數
    
    filtered_threads = [
        item for item in candidate_threads
        if int(item.get("no_of_reply", 0)) >= min_replies and int(item.get("like_count", 0)) >= min_likes
    ]
    
    logger.debug(f"Filtered threads: {len(filtered_threads)}, filters: min_replies={min_replies}, min_likes={min_likes}")
    
    if not filtered_threads:
        logger.warning(f"No threads passed filters: min_replies={min_replies}, min_likes={min_likes}")
        return {
            "thread_data": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    # 優先級排序
    if progress_callback:
        progress_callback("正在排序帖子", 0.3)
    
    prioritization = await prioritize_threads_with_grok(
        user_query=user_question,
        threads=filtered_threads,
        cat_name=selected_cat,
        cat_id=cat_id
    )
    top_thread_ids = prioritization.get("top_thread_ids", [])
    reason = prioritization.get("reason", "No prioritization reason provided")
    
    logger.debug(f"Prioritization result: top_thread_ids={top_thread_ids}, reason={reason}")
    
    if not top_thread_ids:
        top_thread_ids = [item["thread_id"] for item in filtered_threads[:analysis.get("post_limit", 10)]]
        logger.info(f"Fallback to default thread selection: {top_thread_ids}")
    
    candidate_threads = [
        item for item in filtered_threads
        if str(item["thread_id"]) in [str(tid) for tid in top_thread_ids]
    ]
    
    if not candidate_threads:
        candidate_threads = filtered_threads[:analysis.get("post_limit", 10)]
        logger.warning(f"No prioritized threads found, using default: {len(candidate_threads)} threads")
    
    logger.debug(f"Final candidate threads: {len(candidate_threads)}")
    
    # 並行抓取帖子內容
    reply_limit = int(analysis.get("reply_limit", 50))  # 確保整數
    tasks = []
    for idx, item in enumerate(candidate_threads):
        thread_id = str(item["thread_id"])
        cache_key = thread_id
        cache_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
        if cache_data and cache_data.get("replies") and cache_data.get("fetched_pages"):
            # 規範化緩存數據
            try:
                normalized_cache_data = {
                    **cache_data,
                    "no_of_reply": int(cache_data.get("no_of_reply", 0)),
                    "like_count": int(cache_data.get("like_count", 0)),
                    "dislike_count": int(cache_data.get("dislike_count", 0)),
                    "total_pages": int(cache_data.get("total_pages", 1)) if cache_data.get("total_pages") else 1,  # 處理 total_pages
                    "replies": [
                        {
                            **reply,
                            "like_count": int(reply.get("like_count", 0)),
                            "dislike_count": int(reply.get("dislike_count", 0))
                        } for reply in cache_data.get("replies", [])
                    ]
                }
                logger.debug(f"Using cached data for thread_id={thread_id}, no_of_reply_type={type(normalized_cache_data['no_of_reply'])}, total_pages={normalized_cache_data['total_pages']}")
                thread_data.append(normalized_cache_data)
                continue
            except ValueError as e:
                logger.warning(f"Invalid cache data for thread_id={thread_id}: {str(e)}")
                # 跳過無效緩存數據
                st.session_state.thread_cache.pop(cache_key, None)
        
        specific_pages = None
        fetch_last_pages = pages if segment == "tail" else 0
        try:
            total_pages_raw = item.get("total_pages", 1)
            total_pages = int(total_pages_raw) if str(total_pages_raw).isdigit() else 1  # 處理異常格式
            logger.debug(f"Thread_id={thread_id}, total_pages_raw={total_pages_raw}, total_pages={total_pages}")
            start_page = 1 if segment == "head" else (1 if segment == "avg" else max(1, total_pages - pages))
            if segment == "avg":
                specific_pages = [int(i * total_pages / pages) + 1 for i in range(pages)] if total_pages > pages else list(range(1, total_pages + 1))
        except ValueError as e:
            logger.warning(f"Invalid total_pages for thread_id={thread_id}: {total_pages_raw}, error={str(e)}")
            total_pages = 1
            start_page = 1
        
        tasks.append(get_lihkg_thread_content(
            thread_id=thread_id,
            cat_id=cat_id,
            request_counter=request_counter,
            last_reset=last_reset,
            rate_limit_until=rate_limit_until,
            max_replies=reply_limit,
            fetch_last_pages=fetch_last_pages,
            specific_pages=specific_pages,
            start_page=start_page
        ))
        if progress_callback:
            progress_callback(f"準備抓取帖子 {idx + 1}/{len(candidate_threads)}", 0.5 + 0.3 * ((idx + 1) / len(candidate_threads)))
    
    # 並行執行抓取任務
    content_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for idx, content_result in enumerate(content_results):
        if isinstance(content_result, Exception):
            logger.warning(f"Failed to fetch thread_id={candidate_threads[idx]['thread_id']}: {str(content_result)}")
            continue
        request_counter = content_result.get("request_counter", request_counter)
        last_reset = content_result.get("last_reset", last_reset)
        rate_limit_until = content_result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(content_result.get("rate_limit_info", []))
        if content_result.get("replies"):
            thread_id = candidate_threads[idx]["thread_id"]
            try:
                thread_info = {
                    "thread_id": thread_id,
                    "title": content_result.get("title", candidate_threads[idx]["title"]),
                    "no_of_reply": int(candidate_threads[idx].get("no_of_reply", content_result.get("total_replies", 0))),
                    "last_reply_time": candidate_threads[idx]["last_reply_time"],
                    "like_count": int(candidate_threads[idx].get("like_count", 0)),
                    "dislike_count": int(candidate_threads[idx].get("dislike_count", 0)),
                    "total_pages": int(content_result.get("total_pages", 1)) if content_result.get("total_pages") else 1,  # 處理 total_pages
                    "replies": [
                        {
                            "post_id": reply.get("post_id"),
                            "msg": clean_html(reply.get("msg", "")),
                            "like_count": int(reply.get("like_count", 0)),
                            "dislike_count": int(reply.get("dislike_count", 0)),
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                        } for reply in content_result["replies"] if reply.get("msg")
                    ],
                    "fetched_pages": content_result.get("fetched_pages", [])
                }
                logger.debug(f"Thread_id={thread_id}, no_of_reply_type={type(thread_info['no_of_reply'])}, total_pages={thread_info['total_pages']}, reply_count={len(thread_info['replies'])}")
                thread_data.append(thread_info)
                st.session_state.thread_cache[thread_id] = {"data": thread_info, "timestamp": time.time()}
            except ValueError as e:
                logger.warning(f"Invalid data for thread_id={thread_id}: {str(e)}")
                continue
    
    if not thread_data:
        logger.warning(f"No valid thread data retrieved for query: {user_question}")
    
    logger.info(f"Query processing completed: thread_data_count={len(thread_data)}, rate_limit_info={len(rate_limit_info)}")
    
    return {
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }
