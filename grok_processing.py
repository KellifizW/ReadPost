import aiohttp
import asyncio
import json
import re
import datetime
import time
import logging
import streamlit as st
import os
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from logging_config import configure_logger

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "grok_processing.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90

class PromptBuilder:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
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

    def build_analyze(self, query, cat_name, cat_id, conversation_context=None):
        config = self.config["analyze"]
        context = config["context"].format(
            query=query, cat_name=cat_name, cat_id=cat_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        return f"{config['system']}\n{context}\n{config['instructions']}"

    def build_prioritize(self, query, cat_name, cat_id, threads):
        config = self.config.get("prioritize")
        if not config:
            raise ValueError("Prompt configuration for 'prioritize' not found")
        context = config["context"].format(query=query, cat_name=cat_name, cat_id=cat_id)
        data = config["data"].format(threads=json.dumps(threads, ensure_ascii=False))
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def build_response(self, intent, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        config = self.config["response"].get(intent, self.config["response"]["general"])
        context = config["context"].format(
            query=query, selected_cat=selected_cat,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        data = config["data"].format(
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or [], ensure_ascii=False),
            filters=json.dumps(filters, ensure_ascii=False)
        )
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def get_system_prompt(self, mode):
        return self.config["system"].get(mode, "")

def clean_html(text):
    if not isinstance(text, str):
        text = str(text)
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return "[表情符號]" if "hkgmoji" in text else "[圖片]" if any(ext in text.lower() for ext in ['.webp', '.jpg', '.png']) else "[無內容]"
    return text

def clean_response(response):
    if isinstance(response, str):
        return re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)
    return response

async def extract_keywords_with_grok(query, conversation_context=None):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        return {"keywords": [], "reason": "Missing API key"}

    prompt = f"""
    你是語義分析助手，專注提取繁體中文查詢中的1-3個核心關鍵詞（名詞或核心動詞，過濾停用詞及粵語俚語如「講D咩」「點解」）。
    查詢："{query}"
    對話歷史：{json.dumps(conversation_context or [], ensure_ascii=False)}
    返回：{{
      "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
      "reason": "提取邏輯（70字內）"
    }}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": "你是LIHKG論壇語義分析助手，專注提取關鍵詞。"},
            *conversation_context or [],
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        continue
                    data = await response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    return {"keywords": result.get("keywords", [])[:3], "reason": result.get("reason", "")[:70]}
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            return {"keywords": [], "reason": f"Extraction failed: {str(e)}"[:70]}

async def summarize_context(conversation_context):
    if not conversation_context:
        return {"theme": "general", "keywords": []}
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        return {"theme": "general", "keywords": []}

    prompt = f"""
    分析對話歷史，提煉主題及最多3個關鍵詞：
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    輸出：{{"theme": "主題", "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"]}}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    return {"theme": "general", "keywords": []}
                return json.loads((await response.json())["choices"][0]["message"]["content"])
    except Exception:
        return {"theme": "general", "keywords": []}

async def extract_relevant_thread(conversation_context, query):
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None

    query_keywords = set((await extract_keywords_with_grok(query, conversation_context))["keywords"])
    
    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: (\d+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keywords = set((await extract_keywords_with_grok(title, conversation_context))["keywords"])
                if query_keywords.intersection(title_keywords) or any(kw.lower() in title.lower() for kw in query_keywords):
                    return thread_id, title, message["content"]
    return None, None, None

async def analyze_and_screen(user_query, cat_name, cat_id, conversation_context=None):
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()

    id_match = re.search(r'(?:ID|帖子)\s*(\d+)', user_query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": "fetch_thread_by_id",
            "theme": "特定帖子查詢",
            "category_ids": [cat_id],
            "data_type": "replies",
            "post_limit": 1,
            "reply_limit": 200,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": []},
            "processing": {"intent": "fetch_thread_by_id", "top_thread_ids": [thread_id]},
            "top_thread_ids": [thread_id],
            "reason": f"Detected thread ID {thread_id}"
        }

    query_keywords = (await extract_keywords_with_grok(user_query, conversation_context))["keywords"]
    thread_id, thread_title, _ = await extract_relevant_thread(conversation_context, user_query)
    
    if thread_id:
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": "follow_up",
            "theme": thread_title or "追問相關主題",
            "category_ids": [cat_id],
            "data_type": "replies",
            "post_limit": 1,
            "reply_limit": 200,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": query_keywords},
            "processing": {"intent": "follow_up", "top_thread_ids": [thread_id]},
            "top_thread_ids": [thread_id],
            "reason": f"Matched follow-up to thread_id={thread_id}"
        }

    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "general")
    historical_keywords = context_summary.get("keywords", [])
    
    is_vague = len(query_keywords) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"])
    is_follow_up = False
    referenced_thread_ids = []

    if conversation_context and len(conversation_context) >= 2:
        last_response = conversation_context[-1].get("content", "")
        matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        referenced_thread_ids = matches
        last_query_keywords = (await extract_keywords_with_grok(conversation_context[-2].get("content", ""), conversation_context))["keywords"]
        if (set(query_keywords).intersection(last_query_keywords) or
                any(keyword in user_query for keyword in ["詳情", "更多", "進一步", "點解", "為什麼", "原因"])):
            is_follow_up = True

    if is_follow_up and not thread_id:
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": "search_keywords",
            "theme": query_keywords[0] if query_keywords else historical_theme,
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 2,
            "reply_limit": 200,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": query_keywords or historical_keywords},
            "processing": {"intent": "search_keywords", "top_thread_ids": referenced_thread_ids[:2]},
            "top_thread_ids": referenced_thread_ids[:2],
            "reason": "Follow-up intent, fallback to keyword search"
        }

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": historical_theme,
            "category_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "reply_limit": 0,
            "filters": {},
            "processing": {"intent": "general"},
            "top_thread_ids": [],
            "reason": "Missing API key"
        }

    semantic_prompt = f"""
    比較用戶問題與意圖描述，選擇最匹配意圖。若模糊，延續歷史主題（{historical_theme}）。
    若涉及追問（含「詳情」「更多」「點解」等或標題重疊），選「follow_up」。
    問題：{user_query}
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    意圖：{json.dumps({
        "list_titles": "列出帖子標題",
        "summarize_posts": "總結帖子內容",
        "analyze_sentiment": "分析情緒",
        "general_query": "模糊或無關問題",
        "find_themed": "尋找主題帖子",
        "fetch_dates": "提取日期",
        "search_keywords": "關鍵詞搜索",
        "recommend_threads": "推薦帖子",
        "follow_up": "追問帖子內容",
        "fetch_thread_by_id": "根據ID抓取"
    }, ensure_ascii=False, indent=2)}
    輸出：{{"intent": "意圖", "confidence": 0.0-1.0, "reason": "原因"}}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": prompt_builder.get_system_prompt("analyze")},
            *conversation_context,
            {"role": "user", "content": semantic_prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.5
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        continue
                    data = await response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intent = result.get("intent", "summarize_posts")
                    confidence = result.get("confidence", 0.7)
                    reason = result.get("reason", "語義匹配")

                    if is_vague:
                        intent = "summarize_posts"
                        reason = f"模糊問題，默認總結（歷史：{historical_theme})"

                    theme = historical_theme if is_vague else "general"
                    theme_keywords = historical_keywords if is_vague else query_keywords
                    post_limit = 10
                    reply_limit = 0
                    data_type = "both"
                    min_likes = 0 if cat_id in ["5", "15"] else 5

                    if intent in ["search_keywords", "find_themed"]:
                        theme = query_keywords[0] if query_keywords else historical_theme
                        theme_keywords = query_keywords or historical_keywords
                    elif intent == "follow_up":
                        theme = historical_theme
                        reply_limit = 200
                        data_type = "replies"
                        post_limit = min(len(referenced_thread_ids), 2) or 2
                    elif intent in ["general_query", "introduce"]:
                        reply_limit = 0
                        data_type = "none"

                    return {
                        "direct_response": intent in ["general_query", "introduce"],
                        "intent": intent,
                        "theme": theme,
                        "category_ids": [cat_id],
                        "data_type": data_type,
                        "post_limit": post_limit,
                        "reply_limit": reply_limit,
                        "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": theme_keywords},
                        "processing": {"intent": intent, "top_thread_ids": referenced_thread_ids[:2]},
                        "top_thread_ids": referenced_thread_ids[:2],
                        "reason": reason
                    }
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            min_likes = 0 if cat_id in ["5", "15"] else 5
            return {
                "direct_response": False,
                "intent": "summarize_posts",
                "theme": historical_theme,
                "category_ids": [cat_id],
                "data_type": "both",
                "post_limit": 5,
                "reply_limit": 0,
                "filters": {"min_replies": 0, "min_likes": min_likes, "keywords": historical_keywords},
                "processing": {"intent": "summarize"},
                "top_thread_ids": [],
                "reason": f"Analysis failed, defaulting to {historical_theme}"
            }

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent="summarize_posts"):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        return {"top_thread_ids": [], "reason": "Missing API key"}

    if intent == "follow_up":
        context = st.session_state.get("conversation_context", [])
        if context:
            matches = re.findall(r"\[帖子 ID: (\d+)\]", context[-1].get("content", ""))
            referenced_thread_ids = [int(tid) for tid in matches if any(t["thread_id"] == int(tid) for t in threads)]
            if referenced_thread_ids:
                return {"top_thread_ids": referenced_thread_ids[:2], "reason": "Using referenced thread IDs"}

    prompt = PromptBuilder().build_prioritize(
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

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        continue
                    data = await response.json()
                    return json.loads(data["choices"][0]["message"]["content"])
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            sorted_threads = sorted(threads, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
            return {"top_thread_ids": [t["thread_id"] for t in sorted_threads[:5]], "reason": "Fallback to popularity sorting"}

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, filters=None, cat_id=None):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5}
    prompt_builder = PromptBuilder()
    intent = processing.get('intent', 'summarize')

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        yield "錯誤: 缺少 API 密鑰"
        return

    reply_count_prompt = f"""
    根據問題和意圖決定回覆數量（0、25、50、100、200、250、500）：
    問題：{user_query}
    意圖：{intent}
    深入分析需200-500條，簡單查詢25-50條，無需數據0條，默認100條。
    輸出：{{"replies_per_thread": 100, "reason": "原因"}}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": reply_count_prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }

    max_replies_per_thread = 100
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status == 200:
                    result = json.loads((await response.json())["choices"][0]["message"]["content"])
                    max_replies_per_thread = min(result.get("replies_per_thread", 100), 500)
    except Exception:
        pass

    thread_data_dict = {str(data["thread_id"]): data for data in thread_data if isinstance(data, dict) and "thread_id" in data}

    if intent in ["follow_up", "fetch_thread_by_id"]:
        referenced_thread_ids = []
        if intent == "follow_up":
            thread_id, _, _ = await extract_relevant_thread(conversation_context, user_query)
            if thread_id:
                referenced_thread_ids = [thread_id]
            else:
                matches = re.findall(r"\[帖子 ID: (\d+)\]", conversation_context[-1].get("content", "") if conversation_context else "")
                referenced_thread_ids = [tid for tid in matches if any(str(t["thread_id"]) == tid for t in metadata)]
        else:
            referenced_thread_ids = [tid for tid in processing.get("top_thread_ids", []) if str(tid) in thread_data_dict]
        
        thread_data_dict = {tid: thread_data_dict[tid] for tid in map(str, referenced_thread_ids) if tid in thread_data_dict}

    filtered_thread_data = {}
    total_replies_count = 0

    for tid, data in thread_data_dict.items():
        replies = data.get("replies", [])
        filtered_replies = [r for r in replies if isinstance(r, dict) and r.get("msg") and len(r["msg"].strip()) > 7 and r["msg"].strip() not in ["[圖片]", "[無內容]"]]
        sorted_replies = sorted(filtered_replies, key=lambda x: x.get("like_count", 0), reverse=True)[:max_replies_per_thread]
        total_replies_count += len(sorted_replies)
        filtered_thread_data[tid] = {
            "thread_id": data.get("thread_id", tid),
            "title": data.get("title", ""),
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": sorted_replies,
            "fetched_pages": data.get("fetched_pages", []),
            "total_fetched_replies": len(sorted_replies)
        }

    if total_replies_count < max_replies_per_thread and intent in ["follow_up", "fetch_thread_by_id"]:
        for tid, data in filtered_thread_data.items():
            if data["total_fetched_replies"] < max_replies_per_thread:
                content_result = await get_lihkg_thread_content(
                    thread_id=tid, cat_id=cat_id, max_replies=max_replies_per_thread - data["total_fetched_replies"],
                    fetch_last_pages=2, specific_pages=[page + 1 for page in data["fetched_pages"]][-2:], start_page=max(data["fetched_pages"], default=0) + 1
                )
                if content_result.get("replies"):
                    cleaned_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                        }
                        for reply in content_result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                    ]
                    filtered_additional_replies = [r for r in cleaned_replies if len(r["msg"].strip()) > 7 and r["msg"].strip() not in ["[圖片]", "[無內容]"]]
                    filtered_thread_data[tid] = {
                        "thread_id": data.get("thread_id", tid),
                        "title": data.get("title", ""),
                        "no_of_reply": data.get("no_of_reply", 0),
                        "last_reply_time": data.get("last_reply_time", ""),
                        "like_count": data.get("like_count", 0),
                        "dislike_count": data.get("dislike_count", 0),
                        "replies": data.get("replies", []) + filtered_additional_replies,
                        "fetched_pages": list(set(data.get("fetched_pages", []) + content_result.get("fetched_pages", []))),
                        "total_fetched_replies": len(data.get("replies", []) + filtered_additional_replies)
                    }
                    total_replies_count += len(filtered_additional_replies)
                    st.session_state.thread_cache[tid] = {"data": filtered_thread_data[tid], "timestamp": time.time()}

    min_tokens = 1200
    max_tokens = 3600
    target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens) if total_replies_count else min_tokens
    target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)

    prompt = prompt_builder.build_response(
        intent=intent, query=user_query, selected_cat=selected_cat, conversation_context=conversation_context,
        metadata=metadata, thread_data=list(filtered_thread_data.values()), filters=filters
    ) + "\n請包含帖子ID，格式[帖子 ID: xxx]，禁止[post_id: ...]格式。"

    if len(prompt) > GROK3_TOKEN_LIMIT:
        max_replies_per_thread //= 2
        filtered_thread_data = {
            tid: {**data, "replies": data["replies"][:max_replies_per_thread], "total_fetched_replies": len(data["replies"][:max_replies_per_thread])}
            for tid, data in filtered_thread_data.items()
        }
        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
        prompt = prompt_builder.build_response(
            intent=intent, query=user_query, selected_cat=selected_cat, conversation_context=conversation_context,
            metadata=metadata, thread_data=list(filtered_thread_data.values()), filters=filters
        ) + "\n請包含帖子ID，格式[帖子 ID: xxx]，禁止[post_id: ...]格式。"
        target_tokens = min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": prompt_builder.get_system_prompt("response")},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": target_tokens,
        "temperature": 0.7,
        "stream": True
    }

    async with aiohttp.ClientSession() as session:
        try:
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        if response.status != 200:
                            if attempt < 2:
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            yield f"錯誤：API請求失敗（狀態碼 {response.status}）"
                            return
                        async for line in response.content:
                            if line and not line.isspace():
                                line_str = line.decode('utf-8').strip()
                                if line_str == "data: [DONE]":
                                    break
                                if line_str.startswith("data:"):
                                    try:
                                        chunk = json.loads(line_str[6:])
                                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if content and not ("###" in content and ("Content Moderation" in content or "Blocked" in content)):
                                            yield clean_response(content)
                                    except json.JSONDecodeError:
                                        continue
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt < 2:
                        max_replies_per_thread //= 2
                        filtered_thread_data = {
                            tid: {**data, "replies": data["replies"][:max_replies_per_thread], "total_fetched_replies": len(data["replies"][:max_replies_per_thread])}
                            for tid, data in filtered_thread_data.items()
                        }
                        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
                        prompt = prompt_builder.build_response(
                            intent=intent, query=user_query, selected_cat=selected_cat, conversation_context=conversation_context,
                            metadata=metadata, thread_data=list(filtered_thread_data.values()), filters=filters
                        ) + "\n請包含帖子ID，格式[帖子 ID: xxx]，禁止[post_id: ...]格式。"
                        payload["messages"][-1]["content"] = prompt
                        payload["max_tokens"] = min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    yield "錯誤：生成回應失敗"
                    return
        except Exception as e:
            yield f"錯誤：生成回應失敗（{str(e)}）"
        finally:
            await session.close()

def clean_cache(max_age=3600):
    current_time = time.time()
    for key in [k for k, v in st.session_state.thread_cache.items() if current_time - v["timestamp"] > max_age]:
        del st.session_state.thread_cache[key]

def unix_to_readable(timestamp):
    try:
        return datetime.datetime.fromtimestamp(int(timestamp), tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:MM:SS")
    except (ValueError, TypeError):
        return "1970-01-01 00:00:00"

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, conversation_context=None, progress_callback=None):
    configure_logger("lihkg_api", "lihkg_api.log")
    clean_cache()

    if rate_limit_until > time.time():
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }

    post_limit = min(analysis.get("post_limit", 5), 20)
    reply_limit = analysis.get("reply_limit", 0)
    filters = analysis.get("filters", {})
    min_replies = filters.get("min_replies", 0)
    min_likes = filters.get("min_likes", 0 if cat_id in ["5", "15"] else 5)
    top_thread_ids = analysis.get("top_thread_ids", [])
    intent = analysis.get("intent", "summarize_posts")

    thread_data = []
    rate_limit_info = []

    if intent in ["fetch_thread_by_id", "follow_up"] and top_thread_ids:
        candidate_threads = [{"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0} for tid in top_thread_ids]
        tasks = [
            get_lihkg_thread_content(thread_id=str(tid), cat_id=cat_id, max_replies=reply_limit, fetch_last_pages=0, specific_pages=[], start_page=1)
            for tid in top_thread_ids if str(tid) not in st.session_state.thread_cache or not st.session_state.thread_cache[str(tid)]["data"].get("replies")
        ]

        if tasks:
            content_results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(content_results):
                if isinstance(result, Exception):
                    continue
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                thread_id = str(candidate_threads[idx]["thread_id"])
                if result.get("title"):
                    cleaned_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                        }
                        for reply in result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                    ]
                    thread_info = {
                        "thread_id": thread_id,
                        "title": result.get("title"),
                        "no_of_reply": result.get("total_replies", 0),
                        "last_reply_time": unix_to_readable(result.get("last_reply_time", "0")),
                        "like_count": result.get("like_count", 0),
                        "dislike_count": result.get("dislike_count", 0),
                        "replies": cleaned_replies,
                        "fetched_pages": result.get("fetched_pages", []),
                        "total_fetched_replies": len(cleaned_replies)
                    }
                    thread_data.append(thread_info)
                    st.session_state.thread_cache[thread_id] = {"data": thread_info, "timestamp": time.time()}

        for tid in top_thread_ids:
            if str(tid) in st.session_state.thread_cache and st.session_state.thread_cache[str(tid)]["data"].get("replies"):
                thread_data.append(st.session_state.thread_cache[str(tid)]["data"])

        if len(thread_data) == 1 and intent == "follow_up":
            keyword_result = await extract_keywords_with_grok(user_query, conversation_context)
            supplemental_result = await get_lihkg_topic_list(cat_id=cat_id, start_page=1, max_pages=2)
            supplemental_threads = supplemental_result.get("items", [])
            filtered_supplemental = [
                item for item in supplemental_threads
                if str(item["thread_id"]) not in top_thread_ids and any(kw.lower() in item["title"].lower() for kw in keyword_result["keywords"])
            ][:1]
            request_counter = supplemental_result.get("request_counter", request_counter)
            last_reset = supplemental_result.get("last_reset", last_reset)
            rate_limit_until = supplemental_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(supplemental_result.get("rate_limit_info", []))

            tasks = [
                get_lihkg_thread_content(thread_id=str(item["thread_id"]), cat_id=cat_id, max_replies=reply_limit, fetch_last_pages=0, specific_pages=[], start_page=1)
                for item in filtered_supplemental
            ]
            if tasks:
                supplemental_results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(supplemental_results):
                    if isinstance(result, Exception):
                        continue
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    thread_id = str(filtered_supplemental[idx]["thread_id"])
                    if result.get("title"):
                        cleaned_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0),
                                "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                            }
                            for reply in result.get("replies", [])
                            if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                        ]
                        thread_info = {
                            "thread_id": thread_id,
                            "title": result.get("title"),
                            "no_of_reply": result.get("total_replies", 0),
                            "last_reply_time": unix_to_readable(result.get("last_reply_time", "0")),
                            "like_count": filtered_supplemental[idx].get("like_count", 0),
                            "dislike_count": filtered_supplemental[idx].get("dislike_count", 0),
                            "replies": cleaned_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(cleaned_replies)
                        }
                        thread_data.append(thread_info)
                        st.session_state.thread_cache[thread_id] = {"data": thread_info, "timestamp": time.time()}

        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }

    initial_threads = []
    for page in range(1, 6):
        result = await get_lihkg_topic_list(cat_id=cat_id, start_page=page, max_pages=1)
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(result.get("rate_limit_info", []))
        items = result.get("items", [])
        for item in items:
            item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
        initial_threads.extend(items)
        if len(initial_threads) >= 150:
            initial_threads = initial_threads[:150]
            break
        if progress_callback:
            progress_callback(f"已抓取第 {page}/5 頁帖子", 0.1 + 0.2 * (page / 5))

    filtered_items = [
        item for item in initial_threads
        if item.get("no_of_reply", 0) >= min_replies and (cat_id in ["5", "15"] or int(item.get("like_count", 0)) >= min_likes)
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

    candidate_threads = []
    if top_thread_ids:
        candidate_threads = [{"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0} for tid in top_thread_ids]
    else:
        prioritization = await prioritize_threads_with_grok(user_query, filtered_items, selected_cat, cat_id, intent)
        top_thread_ids = prioritization.get("top_thread_ids", [])
        candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit] or \
                           sorted(filtered_items, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)[:post_limit]

    if progress_callback:
        progress_callback("正在抓取帖子內容", 0.3)

    tasks = [
        get_lihkg_thread_content(thread_id=str(item["thread_id"]), cat_id=cat_id, max_replies=reply_limit, fetch_last_pages=0, specific_pages=[], start_page=1)
        for item in candidate_threads if str(item["thread_id"]) not in st.session_state.thread_cache or not st.session_state.thread_cache[str(item["thread_id"])]["data"].get("replies")
    ]

    if tasks:
        content_results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(content_results):
            if isinstance(result, Exception):
                continue
            request_counter = result.get("request_counter", request_counter)
            last_reset = result.get("last_reset", last_reset)
            rate_limit_until = result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(result.get("rate_limit_info", []))
            thread_id = str(candidate_threads[idx]["thread_id"])
            if result.get("title"):
                cleaned_replies = [
                    {
                        "reply_id": reply.get("reply_id"),
                        "msg": clean_html(reply.get("msg", "[無內容]")),
                        "like_count": reply.get("like_count", 0),
                        "dislike_count": reply.get("dislike_count", 0),
                        "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                    }
                    for reply in result.get("replies", [])
                    if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                ]
                thread_info = {
                    "thread_id": thread_id,
                    "title": result.get("title"),
                    "no_of_reply": result.get("total_replies", 0),
                    "last_reply_time": unix_to_readable(result.get("last_reply_time", "0")),
                    "like_count": candidate_threads[idx].get("like_count", 0),
                    "dislike_count": candidate_threads[idx].get("dislike_count", 0),
                    "replies": cleaned_replies,
                    "fetched_pages": result.get("fetched_pages", []),
                    "total_fetched_replies": len(cleaned_replies)
                }
                thread_data.append(thread_info)
                st.session_state.thread_cache[thread_id] = {"data": thread_info, "timestamp": time.time()}

    for item in candidate_threads:
        thread_id = str(item["thread_id"])
        if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
            thread_data.append(st.session_state.thread_cache[thread_id]["data"])

    if progress_callback:
        progress_callback("正在準備數據", 0.5)

    return {
        "selected_cat": selected_cat,
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until,
        "analysis": analysis
    }
