"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
主要功能：
- IntentRegistry：模組化意圖管理，動態支援新意圖。
- quick_intent_classifier：快速意圖分類，減少 API 調用。
- PromptBuilder：動態生成細粒度提示詞。
- process_user_question：優化緩存和批量抓取。
- with_error_handling：統一錯誤處理。
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
import traceback
from string import Template
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content_batch
from typing import Dict, List, Optional, Callable, Any

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
API_TIMEOUT = 90  # 秒

class IntentRegistry:
    """
    意圖註冊器，管理意圖配置和處理邏輯。
    """
    def __init__(self):
        self.intents = {
            "list_titles": {
                "system_prompt": "你是 LIHKG 論壇的數據助手，以繁體中文回答，模擬論壇用戶的語氣。",
                "handler": self.list_titles,
                "filters": {"min_replies": 10, "min_likes": 2, "sort": "recent"},
                "data_type": "titles",
                "post_limit": 10,
                "reply_limit": 0
            },
            "summarize_posts": {
                "system_prompt": "你是 LIHKG 論壇的集體意見代表，以繁體中文回答，模擬論壇用戶的語氣。",
                "handler": self.summarize_posts,
                "filters": {"min_replies": 20, "min_likes": 5, "sort": "popular"},
                "data_type": "both",
                "post_limit": 10,
                "reply_limit": 100
            },
            "analyze_sentiment": {
                "system_prompt": "你是 LIHKG 論壇的集體意見代表，以繁體中文回答，模擬論壇用戶的語氣。",
                "handler": self.analyze_sentiment,
                "filters": {"min_replies": 20, "min_likes": 5, "sort": "popular"},
                "data_type": "both",
                "post_limit": 5,
                "reply_limit": 100
            },
            "fetch_dates": {
                "system_prompt": "你是 LIHKG 論壇的數據助手，以繁體中文回答，模擬論壇用戶的語氣。",
                "handler": self.fetch_dates,
                "filters": {"min_replies": 10, "min_likes": 2, "sort": "recent"},
                "data_type": "both",
                "post_limit": 5,
                "reply_limit": 50
            },
            "trend_prediction": {
                "system_prompt": "你是 LIHKG 論壇助手，預測討論區趨勢。",
                "handler": self.trend_prediction,
                "filters": {"min_replies": 50, "min_likes": 10, "sort": "recent"},
                "data_type": "titles",
                "post_limit": 20,
                "reply_limit": 0
            }
        }

    def register_intent(self, intent_name: str, config: Dict[str, Any]):
        self.intents[intent_name] = config

    def get_intent_config(self, intent_name: str) -> Dict[str, Any]:
        return self.intents.get(intent_name, self.intents["summarize_posts"])

    async def list_titles(self, query: str, data: List[Dict], **kwargs) -> str:
        if not data:
            return f"在 {kwargs.get('cat_name', '未知分類')} 中未找到符合條件的帖子。"
        complexity = kwargs.get("complexity", "simple")
        if complexity == "simple":
            return "\n".join(f"帖子 ID: {item['thread_id']} 標題: {item['title']}" for item in data[:10])
        else:
            return "\n".join(
                f"帖子 ID: {item['thread_id']} 標題: {item['title']}（回覆數：{item.get('no_of_reply', 0)}，點讚數：{item.get('like_count', 0)}）"
                for item in data[:10]
            )

    async def summarize_posts(self, query: str, data: Dict[str, Dict], **kwargs) -> str:
        if not data:
            return f"在 {kwargs.get('cat_name', '未知分類')} 中未找到符合條件的帖子。"
        # 簡化總結邏輯，交由 Grok 3 API 處理
        return ""

    async def analyze_sentiment(self, query: str, data: Dict[str, Dict], **kwargs) -> str:
        if not data:
            return f"在 {kwargs.get('cat_name', '未知分類')} 中未找到符合條件的帖子。"
        # 交由 Grok 3 API 處理
        return ""

    async def fetch_dates(self, query: str, data: Dict[str, Dict], **kwargs) -> str:
        if not data:
            return f"在 {kwargs.get('cat_name', '未知分類')} 中未找到符合條件的帖子。"
        complexity = kwargs.get("complexity", "simple")
        result = f"喺 {kwargs.get('cat_name', '未知分類')} 搵到以下帖文的日期資料：\n\n"
        for item in list(data.values())[:5]:
            title = item["title"]
            last_reply_time = item.get("last_reply_time", "1970-01-01 00:00:00")
            result += f"**標題**：{title}\n最後回覆時間：{last_reply_time}\n"
            if complexity == "detailed" and item.get("replies"):
                top_reply = max(item["replies"], key=lambda x: x.get("like_count", 0), default={})
                reply_time = top_reply.get("reply_time", "1970-01-01 00:00:00")
                result += f"熱門回覆發布時間：{reply_time}\n"
            result += "\n"
        result += "以上係最近幾個帖子的日期資料。"
        return result

    async def trend_prediction(self, query: str, data: List[Dict], **kwargs) -> str:
        if not data:
            return f"在 {kwargs.get('cat_name', '未知分類')} 中未找到符合條件的帖子。"
        sorted_threads = sorted(
            data,
            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
            reverse=True
        )[:5]
        result = f"喺 {kwargs.get('cat_name', '未知分類')} 預測以下話題將會熱門：\n\n"
        for item in sorted_threads:
            result += f"**標題**：{item['title']}\n回覆數：{item.get('no_of_reply', 0)}，點讚數：{item.get('like_count', 0)}\n"
        result += "\n**趨勢分析**：以上帖子因高回覆數和點讚數，預計將持續受到關注。"
        return result

class PromptBuilder:
    """
    提示詞生成器，動態構建細粒度提示詞。
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
                self.config = json.load(f)
                logger.info("Loaded prompts.json successfully")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in prompts.json: {e}")
            raise

    def get_system_prompt(self, mode: str) -> str:
        return self.config["system_prompts"].get(mode, "")

    def build_dynamic_prompt(self, intent: str, query: str, cat_name: str, data: Any, complexity: str = "simple") -> str:
        intent_config = self.config["intents"].get(intent, {})
        system_prompt = intent_config.get("system", "")
        template_str = intent_config.get("templates", {}).get(complexity, intent_config.get("templates", {}).get("simple", ""))
        
        # 根據複雜度調整數據量
        if complexity == "simple" and isinstance(data, dict):
            data = {k: {kk: vv[:5] if isinstance(vv, list) else vv for kk, vv in v.items()} for k, v in data.items()}
        elif complexity == "detailed" and isinstance(data, dict):
            data = {k: {kk: vv[:20] if isinstance(vv, list) else vv for kk, vv in v.items()} for k, v in data.items()}
        
        template = Template(template_str)
        return f"{system_prompt}\n{template.substitute(query=query, cat_name=cat_name, data=json.dumps(data, ensure_ascii=False))}\n{intent_config.get('instructions', '')}"

def clean_html(text: str) -> str:
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

def quick_intent_classifier(query: str) -> Optional[Dict[str, str]]:
    """
    快速意圖分類器，識別簡單查詢，減少 API 調用。
    """
    query_lower = query.lower()
    if "列出" in query_lower or "標題" in query_lower:
        return {"intent": "list_titles", "complexity": "simple"}
    if "分析" in query_lower or "總結" in query_lower or "on9" in query_lower:
        return {"intent": "summarize_posts", "complexity": "detailed" if "分析" in query_lower else "simple"}
    if "情緒" in query_lower or "態度" in query_lower:
        return {"intent": "analyze_sentiment", "complexity": "detailed"}
    if "日期" in query_lower or "時間" in query_lower or "最近" in query_lower:
        return {"intent": "fetch_dates", "complexity": "simple"}
    if "趨勢" in query_lower or "預測" in query_lower:
        return {"intent": "trend_prediction", "complexity": "simple"}
    return None

async def analyze_and_screen(user_query: str, cat_name: str, cat_id: str, thread_titles: List[str] = None, metadata: List[Dict] = None, thread_data: Dict[str, Dict] = None, is_advanced: bool = False, conversation_context: List[Dict] = None) -> Dict[str, Any]:
    """
    分析用戶問題，識別意圖，動態設置篩選條件。
    """
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    registry = IntentRegistry()

    # 快速意圖分類
    quick_result = quick_intent_classifier(user_query)
    if quick_result and not is_advanced:
        intent = quick_result["intent"]
        complexity = quick_result["complexity"]
        intent_config = registry.get_intent_config(intent)
        return {
            "direct_response": intent in ["list_titles", "fetch_dates", "trend_prediction"],
            "intent": intent,
            "theme": "general" if intent not in ["fetch_dates", "trend_prediction"] else intent,
            "category_ids": [cat_id],
            "data_type": intent_config["data_type"],
            "post_limit": intent_config["post_limit"],
            "reply_limit": intent_config["reply_limit"],
            "filters": intent_config["filters"],
            "processing": intent,
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": complexity == "detailed",
            "reason": f"Quick intent classification: {intent}",
            "theme_keywords": [],
            "complexity": complexity
        }

    # 深度分析
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
            "theme_keywords": [],
            "complexity": "simple"
        }

    prompt = prompt_builder.build_dynamic_prompt(
        intent="summarize_posts",
        query=user_query,
        cat_name=cat_name,
        data={"thread_titles": thread_titles or [], "metadata": metadata or [], "thread_data": thread_data or {}},
        complexity="simple"
    )
    
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
                            "reason": f"API request failed with status {response.status}",
                            "theme_keywords": [],
                            "complexity": "simple"
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
                            "theme_keywords": [],
                            "complexity": "simple"
                        }
                    
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intent = result.get("intent", "summarize_posts")
                    intent_config = registry.get_intent_config(intent)
                    result.setdefault("direct_response", False)
                    result.setdefault("intent", intent)
                    result.setdefault("theme", "general")
                    result.setdefault("category_ids", [cat_id])
                    result.setdefault("data_type", intent_config["data_type"])
                    result.setdefault("post_limit", intent_config["post_limit"])
                    result.setdefault("reply_limit", intent_config["reply_limit"])
                    result.setdefault("filters", intent_config["filters"])
                    result.setdefault("processing", intent)
                    result.setdefault("candidate_thread_ids", [])
                    result.setdefault("top_thread_ids", [])
                    result.setdefault("needs_advanced_analysis", False)
                    result.setdefault("reason", "")
                    result.setdefault("theme_keywords", [])
                    result.setdefault("complexity", "simple")
                    logger.info(f"Intent analysis completed: intent={result['intent']}, theme={result['theme']}")
                    return result
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
                "theme_keywords": [],
                "complexity": "simple"
            }

async def prioritize_threads_with_grok(user_query: str, threads: List[Dict], cat_name: str, cat_id: str) -> Dict[str, Any]:
    """
    使用 Grok 3 根據問題語義排序帖子。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"top_thread_ids": [], "reason": "Missing API key"}

    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_dynamic_prompt(
        intent="summarize_posts",
        query=user_query,
        cat_name=cat_name,
        data=[{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads],
        complexity="simple"
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
                        return {"top_thread_ids": [], "reason": f"API request failed with status {response.status}"}
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Thread prioritization failed: missing choices, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {"top_thread_ids": [], "reason": "Invalid API response: missing choices"}
                    
                    content = data["choices"][0]["message"]["content"]
                    try:
                        result = json.loads(content)
                        if not isinstance(result, dict) or "top_thread_ids" not in result:
                            logger.warning(f"Invalid prioritization result format: {content}")
                            return {"top_thread_ids": [], "reason": "Invalid result format"}
                        logger.info(f"Thread prioritization succeeded: {result}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse prioritization result: {content}, error: {str(e)}")
                        return {"top_thread_ids": [], "reason": f"Failed to parse API response: {str(e)}"}
        except Exception as e:
            logger.warning(f"Thread prioritization error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return {"top_thread_ids": [], "reason": f"Prioritization failed after {max_retries} attempts: {str(e)}"}

async def with_error_handling(coro: Callable, error_message: str = "處理失敗") -> Any:
    """
    統一錯誤處理中間件。
    """
    try:
        return await coro
    except Exception as e:
        logger.error(json.dumps({
            "event": "error",
            "message": str(e),
            "stack": traceback.format_exc()
        }, ensure_ascii=False))
        return {"error": f"{error_message}: {str(e)}"}

async def stream_grok3_response(user_query: str, metadata: List[Dict], thread_data: Dict[str, Dict], processing: Dict[str, Any], selected_cat: str, conversation_context: List[Dict] = None, needs_advanced_analysis: bool = False, reason: str = "", filters: Dict = None) -> str:
    """
    使用 Grok 3 API 生成流式回應。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 20, "min_likes": 5}
    prompt_builder = PromptBuilder()
    registry = IntentRegistry()
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    intent = processing.get("intent", "summarize_posts")
    complexity = processing.get("complexity", "simple")
    intent_config = registry.get_intent_config(intent)
    
    # 快速處理簡單意圖
    if intent in ["list_titles", "fetch_dates", "trend_prediction"] and not needs_advanced_analysis:
        result = await intent_config["handler"](user_query, thread_data if intent == "fetch_dates" else metadata, cat_name=selected_cat, complexity=complexity)
        yield result
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
    
    prompt = prompt_builder.build_dynamic_prompt(
        intent=intent,
        query=user_query,
        cat_name=selected_cat,
        data=filtered_thread_data,
        complexity=complexity
    )
    
    if len(prompt) > GROK3_TOKEN_LIMIT:
        max_replies_per_thread = max_replies_per_thread // 2
        filtered_thread_data = {
            tid: {k: v[:max_replies_per_thread] if k == "replies" else v for k, v in data.items()}
            for tid, data in filtered_thread_data.items()
        }
        prompt = prompt_builder.build_dynamic_prompt(
            intent=intent,
            query=user_query,
            cat_name=selected_cat,
            data=filtered_thread_data,
            complexity=complexity
        )
        logger.info(f"Truncated prompt: original_length={len(prompt)}, new_length={len(prompt)}")
    
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
        for attempt in range(3):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Response generation failed: status={response.status}, attempt={attempt + 1}")
                        if attempt < 2:
                            await asyncio.sleep(2 + attempt * 2)
                            continue
                        yield f"錯誤：API 請求失敗（狀態碼 {response.status}）。請稍後重試。"
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
                    if response_content:
                        logger.info(f"Response generation completed: length={len(response_content)}")
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                logger.warning(f"Response generation error: {str(e)}, attempt={attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(2 + attempt * 2)
                    continue
                yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試。"
                return
        yield "無法生成詳細總結，可能是數據不足。請試試其他問題！"
        return

def clean_cache(max_age: int = 3600, max_size: int = 1000):
    """
    清理過期緩存數據，限制緩存大小。
    """
    current_time = time.time()
    cache = st.session_state.thread_cache
    sorted_items = sorted(cache.items(), key=lambda x: x[1]["timestamp"], reverse=True)
    for key, value in sorted_items[max_size:]:
        del cache[key]
    for key, value in sorted_items:
        if current_time - value["timestamp"] > max_age:
            del cache[key]

def unix_to_readable(timestamp: str) -> str:
    """
    將 Unix 時間戳轉換為香港時區的時間格式。
    """
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert timestamp {timestamp}: {str(e)}")
        return "1970-01-01 00:00:00"

async def process_user_question(user_question: str, selected_cat: str, cat_id: str, analysis: Dict[str, Any], request_counter: int, last_reset: float, rate_limit_until: float, is_advanced: bool = False, previous_thread_ids: List[str] = None, previous_thread_data: Dict[str, Dict] = None, conversation_context: List[Dict] = None, progress_callback: Callable = None) -> Dict[str, Any]:
    """
    處理用戶問題，使用批量抓取和優化緩存。
    """
    start_time = time.time()
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
    
    if progress_callback:
        progress_callback("正在抓取帖子列表", 0.1)
    
    post_limit = min(analysis.get("post_limit", 5), 20)
    reply_limit = analysis.get("reply_limit", 50)
    filters = analysis.get("filters", {})
    min_replies = filters.get("min_replies", 20)
    min_likes = filters.get("min_likes", 5)
    sort_method = filters.get("sort", "popular")
    top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
    previous_thread_ids = previous_thread_ids or []
    intent = analysis.get("intent", "summarize_posts")
    
    thread_data = []
    rate_limit_info = []
    initial_threads = []
    
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
        for item in items:
            item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
        initial_threads.extend(items)
        if not items:
            logger.warning(f"No threads fetched for cat_id={cat_id}, page={page}")
        if len(initial_threads) >= 150:
            initial_threads = initial_threads[:150]
            break
        if progress_callback:
            progress_callback(f"已抓取第 {page}/5 頁帖子", 0.1 + 0.2 * (page / 5))
    
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
        if not top_thread_ids and filtered_items:
            if progress_callback:
                progress_callback("正在重新分析帖子選擇", 0.4)
            prioritization = await prioritize_threads_with_grok(user_question, filtered_items, selected_cat, cat_id)
            top_thread_ids = prioritization.get("top_thread_ids", [])
            if not top_thread_ids:
                logger.warning(f"Prioritization failed: {prioritization.get('reason', 'Unknown reason')}")
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                    reverse=True
                )
                top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
                logger.info(f"Fallback to popularity sorting: {top_thread_ids}")
            else:
                logger.info(f"Grok prioritized threads: {top_thread_ids}")
    
    if progress_callback:
        progress_callback("正在抓取候選帖子內容", 0.5)
    
    candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
    if not candidate_threads and filtered_items:
        candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items)))
        logger.info(f"No candidate threads, using random: {len(candidate_threads)} threads selected")
    
    content_result = await get_lihkg_thread_content_batch(
        thread_ids=[item["thread_id"] for item in candidate_threads],
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
    
    for idx, content in enumerate(content_result.get("results", [])):
        if progress_callback:
            progress_callback(f"正在處理帖子 {idx + 1}/{len(candidate_threads)}", 0.5 + 0.3 * ((idx + 1) / len(candidate_threads)))
        
        thread_id = content.get("replies", [{}])[0].get("thread_id", None)
        if not thread_id:
            continue
        item = next((i for i in candidate_threads if str(i["thread_id"]) == str(thread_id)), None)
        if not item:
            continue
        
        thread_info = {
            "thread_id": thread_id,
            "title": content.get("title", item["title"]),
            "no_of_reply": item.get("no_of_reply", content.get("total_replies", 0)),
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
                } for reply in content["replies"] if reply.get("msg")
            ],
            "fetched_pages": content.get("fetched_pages", [])
        }
        thread_data.append(thread_info)
        st.session_state.thread_cache[thread_id] = {
            "data": thread_info,
            "timestamp": time.time()
        }
    
    if analysis.get("needs_advanced_analysis", False) and thread_data:
        if progress_callback:
            progress_callback("正在進行進階分析", 0.8)
        advanced_analysis = await analyze_and_screen(
            user_query=user_query,
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
    
    if progress_callback:
        progress_callback("完成數據處理", 0.9)
    
    logger.info(json.dumps({
        "event": "query_processed",
        "query": user_question,
        "intent": analysis.get("intent"),
        "response_time": time.time() - start_time,
        "thread_count": len(thread_data)
    }, ensure_ascii=False))
    
    return {
        "selected_cat": selected_cat,
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until,
        "analysis": analysis
    }
