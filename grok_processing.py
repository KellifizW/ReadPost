"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
主要函數：
- analyze_and_screen：分析問題，動態識別意圖和篩選條件。
- stream_grok3_response：生成流式回應，支援結構化輸出。
- process_user_question：處理用戶問題，抓取帖子並生成回應。
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
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
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

class PromptBuilder:
    """
    提示詞生成器，從 prompts.json 載入模板並動態構建提示詞。
    """
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "prompts.json")
        
        logger.info(f"Loading prompts.json from: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"prompts.json not found at: {config_path}")
            raise FileNotFoundError(f"prompts.json not found at: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.loads(f.read())
                logger.info("Loaded prompts.json successfully")
        except Exception as e:
            logger.error(f"Failed to load prompts.json: {str(e)}")
            raise

    def build_dynamic_prompt(self, intent, query, cat_name, cat_id, metadata, thread_data, filters, conversation_context):
        intent_config = self.config["intents"]["specific_intents"].get(intent, self.config["intents"]["generic"])
        logger.info(f"Building prompt for intent: {intent}, config keys: {list(intent_config.keys())}")
        
        # 確保必要鍵存在，否則提供默認值
        defaults = {
            "task": "回答通用問題或提供功能說明",
            "data_requirements": ["none"],
            "filters": {},
            "output_format": "{\"response\": \"string\"}",
            "data_processing": "無需抓取數據，直接回答",
            "no_data_response": "問題與 LIHKG 數據無關或無數據，直接回答。"
        }
        for key, default in defaults.items():
            intent_config.setdefault(key, default)
        
        prompt = f"{intent_config['system']}\n"
        prompt += intent_config["context"].format(
            query=query, cat_name=cat_name, cat_id=cat_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        prompt += intent_config["data"].format(
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False),
            filters=json.dumps(filters or {}, ensure_ascii=False)
        )
        prompt += intent_config["instructions_template"].format(
            task=intent_config["task"],
            data_requirements=intent_config["data_requirements"],
            filters=json.dumps(filters or {}, ensure_ascii=False),
            output_format=intent_config["output_format"],
            data_processing=intent_config["data_processing"],
            no_data_response=intent_config["no_data_response"].format(cat_name=cat_name, filters=filters or {})
        )
        return prompt

    def build_analyze_prompt(self, query, cat_name, cat_id, conversation_context):
        intent_config = self.config["intents"]["generic"]
        available_intents = list(self.config["intents"]["specific_intents"].keys())
        prompt = f"{intent_config['system']}\n"
        prompt += intent_config["context"].format(
            query=query, cat_name=cat_name, cat_id=cat_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        prompt += f"任務：分析問題意圖，選擇最適合的意圖，僅從以下選項中選擇：{', '.join(available_intents)}。\n"
        prompt += "示例：\n- '你可以做什麼？' -> intent=ask_functionality\n- '分析熱門帖子' -> intent=analyze_popular_posts\n- '連登仔點睇美股' -> intent=search_opinions\n"
        prompt += "輸出格式：{\"intent\": \"string\", \"data_requirements\": [\"string\"], \"filters\": {\"min_replies\": number, \"min_likes\": number, \"sort\": \"string\"}, \"post_limit\": number, \"reply_limit\": number, \"task\": \"string\", \"output_format\": \"string\", \"theme\": \"string\", \"theme_keywords\": [\"string\"]}"
        return prompt

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

async def analyze_and_screen(user_query, cat_name, cat_id, conversation_context=None):
    """
    分析用戶問題，動態識別意圖和篩選條件。
    """
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
    prompt = prompt_builder.build_analyze_prompt(
        query=user_query,
        cat_name=cat_name,
        cat_id=cat_id,
        conversation_context=conversation_context
    )
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {
            "intent": "general_query",
            "data_requirements": ["none"],
            "filters": {},
            "post_limit": 0,
            "reply_limit": 0,
            "task": "回答通用問題",
            "output_format": "{\"response\": \"string\"}",
            "theme": "",
            "theme_keywords": []
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": prompt_builder.config["intents"]["generic"]["system"]},
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
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Intent analysis failed: missing choices, attempt={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    # 驗證意圖是否在 specific_intents 中
                    specific_intents = prompt_builder.config["intents"]["specific_intents"].keys()
                    if result.get("intent") not in specific_intents:
                        result["intent"] = "general_query"
                        result["data_requirements"] = ["none"]
                        result["filters"] = {}
                        result["post_limit"] = 0
                        result["reply_limit"] = 0
                        result["task"] = "回答通用問題"
                        result["output_format"] = "{\"response\": \"string\"}"
                    result.setdefault("post_limit", 5)
                    result.setdefault("reply_limit", 50)
                    result.setdefault("theme", "general")
                    result.setdefault("theme_keywords", [])
                    logger.info(f"Intent analysis completed: intent={result['intent']}")
                    return result
        except Exception as e:
            logger.warning(f"Intent analysis error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
    return {
        "intent": "general_query",
        "data_requirements": ["none"],
        "filters": {},
        "post_limit": 0,
        "reply_limit": 0,
        "task": "回答通用問題",
        "output_format": "{\"response\": \"string\"}",
        "theme": "",
        "theme_keywords": []
    }

async def stream_grok3_response(user_query, metadata, thread_data, intent, selected_cat, conversation_context=None, filters=None):
    """
    使用 Grok 3 API 生成流式回應，支援結構化輸出。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 20, "min_likes": 5}
    prompt_builder = PromptBuilder()
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        yield json.dumps({"error": "缺少 API 密鑰"})
        return
    
    max_replies_per_thread = 20
    filtered_thread_data = {}
    if not isinstance(thread_data, (dict, list)):
        logger.error(f"Invalid thread_data type: {type(thread_data)}, expected dict or list")
        yield json.dumps({"error": prompt_builder.config["intents"]["error_templates"]["data_invalid"]})
        return
    
    if isinstance(thread_data, list):
        thread_data = {item["thread_id"]: item for item in thread_data if isinstance(item, dict) and "thread_id" in item}
    
    for tid, data in thread_data.items():
        if not isinstance(data, dict):
            logger.warning(f"Skipping invalid thread_data entry for tid {tid}: {data}")
            continue
        replies = data.get("replies", [])
        if not isinstance(replies, list):
            logger.warning(f"Invalid replies format for tid {tid}: {replies}")
            replies = []
        sorted_replies = sorted(
            [r for r in replies if isinstance(r, dict) and r.get("msg")],
            key=lambda x: x.get("like_count", 0),
            reverse=True
        )[:max_replies_per_thread]
        filtered_thread_data[tid] = {
            "thread_id": data.get("thread_id", tid),
            "title": data.get("title", ""),
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", "0"),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": sorted_replies,
            "fetched_pages": data.get("fetched_pages", [])
        }
    
    prompt = prompt_builder.build_dynamic_prompt(
        intent=intent,
        query=user_query,
        cat_name=selected_cat,
        cat_id="",
        metadata=metadata,
        thread_data=filtered_thread_data,
        filters=filters,
        conversation_context=conversation_context
    )
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": prompt_builder.config["intents"]["generic"]["system"]},
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
        for attempt in range(3):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Response generation failed: status={response.status}, attempt={attempt + 1}")
                        continue
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
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid JSON chunk: {line_str}")
                                    continue
                    if response_content:
                        return
            except Exception as e:
                logger.warning(f"Response generation error: {str(e)}, attempt={attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(2)
        error_message = prompt_builder.config["intents"]["error_templates"]["api_failure"]
        yield json.dumps({"error": error_message})

def clean_cache(max_age=3600):
    """
    清理過期緩存數據。
    """
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]

def unix_to_readable(timestamp):
    """
    將 Unix 時間戳轉換為香港時區時間格式。
    """
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert timestamp {timestamp}: {str(e)}")
        return "1970-01-01 00:00:00"

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, conversation_context=None, progress_callback=None):
    """
    處理用戶問題，根據意圖抓取數據並生成回應。
    """
    try:
        logger.info(f"Processing user question: {user_question}, category: {selected_cat}")
        
        clean_cache()
        
        if rate_limit_until > time.time():
            logger.warning(f"Rate limit active until {rate_limit_until}")
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        intent = analysis.get("intent", "general_query")
        data_requirements = analysis.get("data_requirements", ["none"])
        filters = analysis.get("filters", {"min_replies": 20, "min_likes": 5})
        post_limit = min(analysis.get("post_limit", 5), 20)
        reply_limit = analysis.get("reply_limit", 50)
        
        thread_data = []
        rate_limit_info = []
        
        if "titles" in data_requirements:
            if progress_callback:
                progress_callback("正在抓取帖子列表", 0.1)
            for page in range(1, 6):
                result = await get_lihkg_topic_list(
                    cat_id=cat_id,
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
                if not isinstance(items, list):
                    logger.error(f"Invalid items format from API: {items}")
                    items = []
                for item in items:
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping invalid item: {item}")
                        continue
                    item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
                thread_data.extend(items)
                if len(thread_data) >= 150:
                    thread_data = thread_data[:150]
                    break
                if progress_callback:
                    progress_callback(f"已抓取第 {page}/5 頁帖子", 0.1 + 0.2 * (page / 5))
        
        if "replies" in data_requirements and thread_data:
            if progress_callback:
                progress_callback("正在抓取帖子內容", 0.3)
            filtered_items = [
                item for item in thread_data
                if isinstance(item, dict) and
                   item.get("no_of_reply", 0) >= filters.get("min_replies", 20) and
                   int(item.get("like_count", 0)) >= filters.get("min_likes", 5)
            ]
            sorted_items = sorted(
                filtered_items,
                key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                reverse=True
            )[:post_limit]
            
            final_thread_data = []
            for idx, item in enumerate(sorted_items):
                thread_id = str(item.get("thread_id", ""))
                if not thread_id:
                    logger.warning(f"Skipping item with missing thread_id: {item}")
                    continue
                cache_key = f"{intent}:{thread_id}"
                cache_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
                
                if cache_data and cache_data.get("replies"):
                    final_thread_data.append(cache_data)
                    continue
                
                if progress_callback:
                    progress_callback(f"正在抓取帖子 {idx + 1}/{len(sorted_items)}", 0.3 + 0.3 * ((idx + 1) / len(sorted_items)))
                
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
                        "title": content_result.get("title", item.get("title", "")),
                        "no_of_reply": item.get("no_of_reply", content_result.get("total_replies", 0)),
                        "last_reply_time": item.get("last_reply_time", "0"),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [
                            {
                                "post_id": reply.get("post_id", ""),
                                "msg": clean_html(reply.get("msg", "")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0),
                                "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                            } for reply in content_result.get("replies", []) if reply.get("msg")
                        ],
                        "fetched_pages": content_result.get("fetched_pages", [])
                    }
                    final_thread_data.append(thread_info)
                    st.session_state.thread_cache[cache_key] = {
                        "data": thread_info,
                        "timestamp": time.time()
                    }
        
            thread_data = final_thread_data
        
        if progress_callback:
            progress_callback("完成數據處理", 0.9)
        
        logger.info(f"Processed thread_data: {len(thread_data)} items")
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
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": rate_limit_info + [{"message": f"Processing failed: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }