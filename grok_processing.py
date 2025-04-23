"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
改進：
- 支持多意圖查詢，動態分解子意圖（新增 extract_keywords、analyze_user_behavior、track_trends、contrast_analysis、recommend_posts）。
- 模組化提示詞，減少 prompts.json 依賴，支援動態參數（例如 time_range、category_ids）。
- 添加後處理驗證，確保回應包含所有意圖結果。
- 增強錯誤處理，支持回退回應。
主要函數：
- analyze_and_screen：分析問題，識別多意圖。
- stream_grok3_response：生成流式回應，動態組合提示詞並驗證結果。
- process_user_question：處理用戶問題，抓取帖子。
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
from collections import Counter
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_KONG")

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90

class AppError(Exception):
    def __init__(self, message, user_message=None):
        self.message = message
        self.user_message = user_message or message
        super().__init__(self.message)

class PromptBuilder:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "prompts.json")
        
        logger.info(f"Loading prompts.json from: {config_path}")
        if not os.path.exists(config_path):
            raise AppError(f"prompts.json not found at: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
                logger.info("Loaded prompts.json successfully")
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
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def build_prioritize(self, query, cat_name, cat_id, threads):
        config = self.config.get("prioritize", {})
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id
        )
        data = config["data"].format(
            threads=json.dumps(threads, ensure_ascii=False)
        )
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def build_response(self, intents, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        system = self.config["system"]["response"]
        context = self.config["analyze"]["context"].format(
            query=query,
            cat_name=selected_cat,
            cat_id="",
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        data = self.config["analyze"]["data"].format(
            thread_titles=json.dumps([item["title"] for item in metadata] if metadata else [], ensure_ascii=False),
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False)
        )
        instructions = []
        for intent in intents:
            component = self.config["response_components"].get(intent["type"], self.config["response_components"]["general"])
            # 動態格式化參數
            params = intent.get("parameters", {})
            instruction = component["instructions"].format(
                theme=intent.get("theme", "general"),
                limit=intent.get("limit", 10),
                parameters=json.dumps(params, ensure_ascii=False)
            )
            instructions.append(f"子任務 {intent['type']}（權重 {intent['weight']}）：\n{instruction}")
        prompt = f"{system}\n{context}\n{data}\n指令：\n{'\n'.join(instructions)}\n最終輸出格式：\n[{','.join(['{\"type\": \"' + i['type'] + '\", \"result\": {...}}' for i in intents])}]"
        return prompt

    def get_system_prompt(self, mode):
        return self.config["system"].get(mode, "")

def clean_html(text):
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
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
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
                    if response.status != 200:
                        logger.warning(f"Intent analysis failed: status={response.status}, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        raise AppError(f"API request failed with status {response.status}", user_message="無法連接到分析服務，請稍後重試。")
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Intent analysis failed: missing choices, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        raise AppError("Invalid API response: missing choices", user_message="分析服務返回無效數據，請稍後重試。")
                    
                    result = json.loads(data["choices"][0]["message"]["content"])
                    
                    # 確保 intents 是列表
                    if "intents" not in result:
                        result["intents"] = [{"type": "summarize_posts", "theme": "general", "weight": 1.0, "parameters": {}}]
                    result.setdefault("theme_keywords", [])
                    result.setdefault("post_limit", 10)
                    result.setdefault("reply_limit", 50)
                    result.setdefault("filters", {"min_replies": 20, "min_likes": 5, "sort": "popular"})
                    result.setdefault("category_ids", [cat_id])
                    result.setdefault("top_thread_ids", [])
                    result.setdefault("reason", "Default analysis")
                    logger.info(f"Intent analysis completed: intents={[i['type'] for i in result['intents']]}")
                    return result
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.warning(f"Intent analysis error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise AppError(f"Analysis failed after {max_retries} attempts: {str(e)}", user_message="分析服務暫時不可用，請稍後重試。")

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        raise AppError(f"Grok 3 API key missing: {str(e)}", user_message="缺少 API 密鑰，請聯繫支持。")

    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prioritize(
        query=user_query,
        cat_name=cat_name,
        cat_id=cat_id,
        threads=[{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads]
    )
    
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
                        raise AppError(f"API request failed with status {response.status}", user_message="無法完成帖子排序，請稍後重試。")
                    
                    content = await response.text()
                    logger.debug(f"Raw prioritize response: {content}")
                    
                    # 清理和修復 JSON
                    content = content.strip()
                    if content and not content.endswith('}'):
                        content += '}'
                    try:
                        result = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error: {str(e)}, content: {content}")
                        # 嘗試提取部分 JSON
                        match = re.search(r'\{"top_thread_ids":\s*\[[\d,\s]*\](?:,\s*"reason":\s*".*?")?\s*(?:\})?', content)
                        if match:
                            partial_content = match.group(0)
                            if not partial_content.endswith('}'):
                                partial_content += '}'
                            try:
                                result = json.loads(partial_content)
                                result.setdefault("reason", "Partial JSON extracted")
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse partial JSON: {partial_content}")
                                # 回退到簡單排序
                                sorted_threads = sorted(
                                    threads,
                                    key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                                    reverse=True
                                )
                                result = {
                                    "top_thread_ids": [t["thread_id"] for t in sorted_threads[:10]],
                                    "reason": "Fallback sorting due to invalid JSON"
                                }
                        else:
                            # 無有效 JSON，回退
                            sorted_threads = sorted(
                                threads,
                                key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                                reverse=True
                            )
                            result = {
                                "top_thread_ids": [t["thread_id"] for t in sorted_threads[:10]],
                                "reason": "Fallback sorting due to invalid JSON"
                            }
                    
                    if not isinstance(result, dict) or "top_thread_ids" not in result:
                        logger.warning(f"Invalid prioritization result format: {content}")
                        raise AppError("Invalid result format: missing required keys", user_message="排序結果格式錯誤，請稍後重試。")
                    logger.info(f"Thread prioritization succeeded: {result}")
                    return result
        except Exception as e:
            logger.warning(f"Thread prioritization error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise AppError(f"Prioritization failed after {max_retries} attempts: {str(e)}", user_message="排序服務暫時不可用，請稍後重試。")

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 20, "min_likes": 5}
    prompt_builder = PromptBuilder()
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    # 解析 intents
    intents = processing.get("intents", [{"type": "summarize_posts", "theme": "general", "weight": 1.0, "parameters": {}}])
    
    # 簡化 metadata 並限制回覆數
    max_replies_per_thread = 20
    simplified_metadata = [{"thread_id": item["thread_id"], "title": item["title"]} for item in metadata]
    filtered_thread_data = {}
    for tid, data in thread_data.items():
        replies = sorted(
            [r for r in data.get("replies", []) if r.get("msg")],
            key=lambda x: x.get("like_count", 0),
            reverse=True
        )[:max_replies_per_thread]
        filtered_thread_data[tid] = {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": replies,
            "fetched_pages": data.get("fetched_pages", [])
        }
    
    # 構建提示詞
    prompt = prompt_builder.build_response(
        intents=intents,
        query=user_query,
        selected_cat=selected_cat,
        conversation_context=conversation_context,
        metadata=simplified_metadata,
        thread_data=filtered_thread_data,
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
    
    logger.info(f"Starting response generation for query: {user_query}, intents: {[i['type'] for i in intents]}")
    
    response_content = ""
    response_json = []
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Response generation failed: status={response.status}, attempt={attempt + 1}")
                        if attempt < 2:
                            await asyncio.sleep(2 + attempt * 2)
                            continue
                        raise AppError(f"API request failed with status {response.status}", user_message="生成回應失敗，請稍後重試。")
                    
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
                                        response_content += content
                                        yield content
                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON decode error in stream chunk: {str(e)}")
                                    continue
                    
                    # 嘗試解析 JSON 回應
                    try:
                        response_json = json.loads(response_content)
                        if not isinstance(response_json, list):
                            response_json = [{"type": "general", "result": {"summary": response_content, "suggestion": "請提供更具體查詢"}}]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse response as JSON: {response_content}")
                        response_json = [{"type": "general", "result": {"summary": response_content, "suggestion": "請提供更具體查詢"}}]
                    
                    # 後處理驗證
                    missing_tasks = []
                    for intent in intents:
                        intent_type = intent["type"]
                        if not any(r["type"] == intent_type for r in response_json):
                            missing_tasks.append((intent_type, intent.get("limit", 10), intent.get("parameters", {})))
                    
                    if missing_tasks:
                        logger.warning(f"Missing tasks in response: {missing_tasks}")
                        for task, limit, params in missing_tasks:
                            fallback = None
                            if task == "list_titles":
                                fallback = {
                                    "type": "list_titles",
                                    "result": [
                                        {"thread_id": m["thread_id"], "title": m["title"]}
                                        for m in simplified_metadata[:limit]
                                    ]
                                }
                                yield "\n代表性帖子：\n" + "\n".join(
                                    f"- 帖子 ID: {m['thread_id']} 標題: {m['title']}"
                                    for m in fallback["result"]
                                )
                            elif task == "summarize_posts":
                                fallback = {
                                    "type": "summarize_posts",
                                    "result": {
                                        "theme": intent.get("theme", "general"),
                                        "content": f"無法生成詳細總結，可能是數據不足。以下是通用概述：{selected_cat}討論涵蓋多主題，網民觀點多元。",
                                        "quotes": []
                                    }
                                }
                                yield f"\n\n主題：{fallback['result']['theme']}\n內容：{fallback['result']['content']}"
                            elif task == "extract_keywords":
                                # 簡單關鍵詞提取回退
                                words = " ".join(m["title"] for m in simplified_metadata).split()
                                word_counts = Counter(words)
                                top_words = word_counts.most_common(3)
                                fallback = {
                                    "type": "extract_keywords",
                                    "result": [
                                        {"keyword": word, "frequency": count, "context": f"出現在標題中"}
                                        for word, count in top_words
                                    ]
                                }
                                yield "\n關鍵詞：\n" + "\n".join(
                                    f"- {r['keyword']}（出現次數：{r['frequency']}，上下文：{r['context']}）"
                                    for r in fallback["result"]
                                )
                            elif task == "analyze_user_behavior":
                                fallback = {
                                    "type": "analyze_user_behavior",
                                    "result": {
                                        "frequency": 0,
                                        "themes": ["未知"],
                                        "sentiment": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                                        "summary": "無法分析用戶行為，數據不足。"
                                    }
                                }
                                yield f"\n\n用戶行為分析：{fallback['result']['summary']}"
                            elif task == "track_trends":
                                fallback = {
                                    "type": "track_trends",
                                    "result": {
                                        "theme": intent.get("theme", "general"),
                                        "time_range": params.get("time_range", "unknown"),
                                        "trend": [],
                                        "summary": f"無法追踪{intent.get('theme', '話題')}趨勢，數據不足。"
                                    }
                                }
                                yield f"\n\n趨勢分析：{fallback['result']['summary']}"
                            elif task == "contrast_analysis":
                                fallback = {
                                    "type": "contrast_analysis",
                                    "result": [
                                        {
                                            "category_id": selected_cat,
                                            "theme": intent.get("theme", "general"),
                                            "keywords": [],
                                            "sentiment": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                                            "heat": 0
                                        }
                                    ]
                                }
                                yield f"\n\n對比分析：無法比較，數據不足。"
                            elif task == "recommend_posts":
                                fallback = {
                                    "type": "recommend_posts",
                                    "result": [
                                        {
                                            "thread_id": m["thread_id"],
                                            "title": m["title"],
                                            "description": "熱門帖子，可能與查詢相關"
                                        }
                                        for m in simplified_metadata[:limit]
                                    ]
                                }
                                yield "\n推薦帖子：\n" + "\n".join(
                                    f"- 帖子 ID: {r['thread_id']} 標題: {r['title']}（{r['description']}）"
                                    for r in fallback["result"]
                                )
                            if fallback:
                                response_json.append(fallback)
                    
                    logger.info(f"Response generation completed: length={len(response_content)}")
                    return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Response generation error: {str(e)}, attempt={attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(2 + attempt * 2)
                    continue
                yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試。"
                return
        yield "錯誤：生成回應失敗，請稍後重試或聯繫支持。"
        await session.close()

def clean_cache(max_age=3600):
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]

def unix_to_readable(timestamp):
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert timestamp {timestamp}: {str(e)}")
        return "1970-01-01 00:00:00"

async def process_user_question(user_question, selected_cat, cat_id, order, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    try:
        logger.info(f"Processing user question: {user_question}, category: {selected_cat}, order: {order}")
        
        clean_cache()
        
        if rate_limit_until > time.time():
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
        top_thread_ids = analysis.get("top_thread_ids", [])
        previous_thread_ids = previous_thread_ids or []
        
        intents = analysis.get("intents", [{"type": "summarize_posts", "theme": "general", "weight": 1.0, "parameters": {}}])
        
        thread_data = []
        rate_limit_info = []
        initial_threads = []
        
        for page in range(1, 4):
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
            if len(initial_threads) >= 135:
                initial_threads = initial_threads[:135]
                break
            if progress_callback:
                progress_callback(f"已抓取第 {page}/3 頁帖子", 0.1 + 0.2 * (page / 3))
        
        if progress_callback:
            progress_callback("正在篩選帖子", 0.3)
        
        filtered_items = [
            item for item in initial_threads
            if item.get("no_of_reply", 0) >= min_replies and
            int(item.get("like_count", 0)) >= min_likes and
            str(item["thread_id"]) not in previous_thread_ids
        ]
        
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
        
        if any(intent["type"] == "fetch_dates" for intent in intents):
            if progress_callback:
                progress_callback("正在處理日期相關資料", 0.4)
            sorted_items = sorted(
                filtered_items,
                key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                reverse=True
            )
            top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
        elif any(intent["type"] == "track_trends" for intent in intents):
            if progress_callback:
                progress_callback("正在處理趨勢資料", 0.4)
            sorted_items = sorted(
                filtered_items,
                key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                reverse=True
            )
            top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
        else:
            if not top_thread_ids and filtered_items:
                if progress_callback:
                    progress_callback("正在重新分析帖子選擇", 0.4)
                prioritization = await prioritize_threads_with_grok(user_question, filtered_items, selected_cat, cat_id)
                top_thread_ids = prioritization.get("top_thread_ids", [])
                if not top_thread_ids:
                    sorted_items = sorted(
                        filtered_items,
                        key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                        reverse=True
                    )
                    top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
        
        if progress_callback:
            progress_callback("正在抓取候選帖子內容", 0.5)
        
        candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
        if not candidate_threads and filtered_items:
            candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items)))
        
        for idx, item in enumerate(candidate_threads):
            thread_id = str(item["thread_id"])
            cache_key = thread_id
            cache_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
            
            if cache_data and cache_data.get("replies"):
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
                max_replies=reply_limit
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
        
        if analysis.get("needs_advanced_analysis", False) and thread_data:
            if progress_callback:
                progress_callback("正在進行進階分析", 0.8)
            advanced_analysis = await analyze_and_screen(
                user_query=user_question,
                cat_name=selected_cat,
                cat_id=cat_id,
                thread_titles=[item["title"] for item in thread_data],
                metadata=[{"thread_id": item["thread_id"], "title": item["title"]} for item in thread_data],
                thread_data={item["thread_id"]: item for item in thread_data},
                is_advanced=True,
                conversation_context=conversation_context
            )
            thread_data = [
                item for item in thread_data
                if str(item["thread_id"]) in map(str, advanced_analysis.get("top_thread_ids", []))
            ]
            analysis.update(advanced_analysis)
        
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
        raise AppError(f"Processing failed: {str(e)}", user_message=f"處理問題失敗：{str(e)}。請稍後重試或聯繫支持。")
