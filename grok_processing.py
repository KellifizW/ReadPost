import aiohttp
import asyncio
import json
import re
import datetime
import time
import logging
import streamlit as st
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from reddit_api import get_reddit_topic_list, get_reddit_thread_content
from logging_config import configure_logger
from dynamic_prompt_utils import build_dynamic_prompt, parse_query, extract_keywords, CONFIG

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "grok_processing.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 120000
API_TIMEOUT = 120
MAX_CACHE_SIZE = 100

cache_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(5)

async def call_grok3_api(payload, headers, retries=3, timeout=API_TIMEOUT, stream=False):
    """Unified API call handler with retries and error handling."""
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        logger.debug(f"API call failed: status={response.status}, attempt={attempt + 1}")
                        continue
                    if stream:
                        return response
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"API call failed: no choices, attempt={attempt + 1}")
                        continue
                    return data
        except (aiohttp.ClientConnectionError, aiohttp.ClientResponseError, asyncio.TimeoutError) as e:
            logger.debug(f"API call error: {str(e)}, attempt={attempt + 1}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
                continue
            return {"error": str(e), "data": None}
    return {"error": "Max retries exceeded", "data": None}

def clean_content(content, is_response=False):
    """Unified content cleaning for HTML and response formatting."""
    if not isinstance(content, str):
        content = str(content)
    try:
        if is_response:
            content = re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', content)
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        if not content:
            return "[表情符號]" if "hkgmoji" in content else "[圖片]" if any(ext in content.lower() for ext in ['.webp', '.jpg', '.png']) else "[無內容]"
        return content
    except Exception as e:
        logger.error(f"Content cleaning failed: {str(e)}")
        return content

def unix_to_readable(timestamp, context="unknown"):
    """Convert timestamp to readable format."""
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        elif isinstance(timestamp, str):
            try:
                dt = datetime.datetime.fromtimestamp(int(timestamp), tz=HONG_KONG_TZ)
            except ValueError:
                dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                dt = HONG_KONG_TZ.localize(dt)
        else:
            raise TypeError(f"Invalid timestamp type: {type(timestamp)}")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"Timestamp conversion failed: value={timestamp}, context={context}, error={str(e)}")
        return "1970-01-01 00:00:00"

def normalize_selected_source(selected_source, source_type):
    """Normalize selected_source to dict format."""
    if isinstance(selected_source, str):
        return {"source_name": selected_source, "source_type": source_type}
    if not isinstance(selected_source, dict) or "source_name" not in selected_source or "source_type" not in selected_source:
        logger.warning(f"Invalid selected_source: {selected_source}, using defaults")
        return {"source_name": "未知", "source_type": source_type}
    return selected_source

async def get_or_update_cache(thread_id, source_type, source_id, max_replies, fetch_last_pages=0, max_comments=100):
    """Unified cache management for thread data."""
    thread_id = str(thread_id)
    async with cache_lock:
        if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
            logger.debug(f"Cache hit: thread_id={thread_id}")
            return st.session_state.thread_cache[thread_id]["data"]
    
    async with request_semaphore:
        if source_type == "lihkg":
            result = await get_lihkg_thread_content(thread_id=thread_id, cat_id=source_id, max_replies=max_replies, fetch_last_pages=fetch_last_pages, start_page=1)
        else:
            result = await get_reddit_thread_content(post_id=thread_id, subreddit=source_id, max_comments=max_comments)
    
    if not result.get("title"):
        logger.warning(f"Failed to fetch thread: {thread_id}")
        return None
    
    filtered_replies = [
        {
            "reply_id": reply.get("reply_id"),
            "msg": clean_content(reply.get("msg", "[無內容]")),
            "like_count": reply.get("like_count", 0),
            "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
            "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"reply in thread {thread_id}")
        }
        for reply in result.get("replies", [])
        if reply.get("msg") and clean_content(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"] and len(clean_content(reply.get("msg")).strip()) > 7
    ]
    
    thread_info = {
        "thread_id": thread_id,
        "title": result.get("title"),
        "no_of_reply": result.get("total_replies", 0),
        "last_reply_time": unix_to_readable(result.get("last_reply_time", "0"), context=f"thread {thread_id}"),
        "like_count": result.get("like_count", 0),
        "dislike_count": result.get("dislike_count", 0) if source_type == "lihkg" else 0,
        "replies": filtered_replies,
        "fetched_pages": result.get("fetched_pages", []),
        "total_fetched_replies": len(filtered_replies)
    }
    
    async with cache_lock:
        st.session_state.thread_cache[thread_id] = {"data": thread_info, "timestamp": time.time()}
    return thread_info

async def summarize_context(conversation_context):
    """Summarize conversation context for theme and keywords."""
    if not conversation_context:
        return {"theme": "一般", "keywords": []}
    
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Missing Grok 3 API key")
        return {"theme": "一般", "keywords": []}
    
    prompt = f"""
你是對話摘要助手，請分析以下對話歷史，提煉主要主題和關鍵詞（最多3個）。
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
輸出格式：{{"theme": "主要主題", "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"]}}
"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100, "temperature": 0.5}
    
    result = await call_grok3_api(payload, headers)
    if result.get("error"):
        logger.warning(f"Context summarization failed: {result['error']}")
        return {"theme": "一般", "keywords": []}
    
    try:
        return json.loads(result["data"]["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse summary: {str(e)}")
        return {"theme": "一般", "keywords": []}

async def analyze_and_screen(user_query, source_name, source_id, source_type="lihkg", conversation_context=None):
    """Analyze query and screen for intents and threads."""
    conversation_context = conversation_context or []
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Missing Grok 3 API key")
        return {"direct_response": True, "intents": [{"intent": "general_query", "confidence": 0.5, "reason": "Missing API key"}], "theme": "一般", "source_type": source_type, "source_ids": [], "data_type": "none", "post_limit": 5, "filters": {}, "processing": {"intents": ["general_query"]}, "candidate_thread_ids": [], "top_thread_ids": [], "needs_advanced_analysis": False, "reason": "Missing API key", "theme_keywords": []}
    
    parsed_query = await parse_query(user_query, conversation_context, api_key, source_type)
    intents = parsed_query["intents"] or [{"intent": "recommend_threads", "confidence": 0.5, "reason": "No valid intent detected"}]
    query_keywords = parsed_query["keywords"]
    top_thread_ids = parsed_query["thread_ids"]
    reason = parsed_query["reason"]
    confidence = parsed_query["confidence"]
    
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "一般")
    historical_keywords = context_summary.get("keywords", [])
    
    is_vague = len(query_keywords) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"]) and not any(i["intent"] == "list_titles" and i["confidence"] >= 0.9 for i in intents)
    if is_vague:
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": f"Vague query, using historical theme: {historical_theme}" if historical_theme != "一般" else "Vague query, default to summarize"}]
        reason = intents[0]["reason"]
    
    theme = historical_theme if is_vague else (query_keywords[0] if query_keywords else "一般")
    theme_keywords = historical_keywords if is_vague else query_keywords
    post_limit = 15 if any(i["intent"] == "list_titles" for i in intents) else 20 if any(i["intent"] in ["search_keywords", "find_themed"] for i in intents) else 5
    data_type = "both" if not all(i["intent"] in ["general_query", "introduce"] for i in intents) else "none"
    
    if any(i["intent"] == "follow_up" for i in intents):
        post_limit = len(top_thread_ids) or 2
        data_type = "replies"
    
    return {
        "direct_response": all(i["intent"] in ["general_query", "introduce"] for i in intents),
        "intents": intents,
        "theme": theme,
        "source_type": source_type,
        "source_ids": [source_id],
        "data_type": data_type,
        "post_limit": post_limit,
        "filters": {"min_replies": 10, "min_likes": 0, "sort": "popular", "keywords": theme_keywords},
        "processing": {"intents": [i["intent"] for i in intents], "top_thread_ids": top_thread_ids, "analysis": parsed_query},
        "candidate_thread_ids": top_thread_ids,
        "top_thread_ids": top_thread_ids,
        "needs_advanced_analysis": confidence < 0.7,
        "reason": reason,
        "theme_keywords": theme_keywords
    }

async def prioritize_threads_with_grok(user_query, threads, source_name, source_id, source_type="lihkg", intents=["summarize_posts"]):
    """Prioritize threads based on query and intents."""
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Missing Grok 3 API key")
        return {"top_thread_ids": [], "reason": "Missing API key", "intent_breakdown": []}
    
    if any(intent == "follow_up" for intent in intents):
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\d+)\]", st.session_state.get("conversation_context", [])[-1].get("content", ""))
        valid_ids = [tid for tid in referenced_thread_ids if any(str(t["thread_id"]) == tid for t in threads)]
        if valid_ids:
            return {"top_thread_ids": valid_ids[:5], "reason": "Using follow-up referenced IDs", "intent_breakdown": [{"intent": "follow_up", "thread_ids": valid_ids[:5]}]}
    
    threads = [{"thread_id": str(t["thread_id"]), **t} for t in threads]
    prompt = f"""
你是帖子優先級排序助手，請根據用戶查詢和多個意圖，從提供的帖子中選出最多20個最相關的帖子。
查詢：{user_query}
意圖：{json.dumps(intents, ensure_ascii=False)}
討論區：{source_name} (ID: {source_id})
來源類型：{source_type}
帖子數據：
{json.dumps([{"thread_id": str(t["thread_id"]), "title": clean_content(t["title"]), "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads], ensure_ascii=False)}
輸出格式：{{
  "top_thread_ids": ["id1", "id2", ...],
  "reason": "排序原因",
  "intent_breakdown": [
    {{"intent": "意圖1", "thread_ids": ["id1", "id2"]}},
    ...
  ]
}}
"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.7}
    
    result = await call_grok3_api(payload, headers)
    if result.get("error"):
        sorted_threads = sorted(threads, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
        return {"top_thread_ids": [str(t["thread_id"]) for t in sorted_threads[:20]], "reason": f"API call failed: {result['error']}", "intent_breakdown": []}
    
    try:
        response = json.loads(result["data"]["choices"][0]["message"]["content"])
        top_thread_ids = [str(tid) for tid in response.get("top_thread_ids", []) if str(tid) in [str(t["thread_id"]) for t in threads]]
        return {"top_thread_ids": top_thread_ids, "reason": response.get("reason", "No reason provided"), "intent_breakdown": response.get("intent_breakdown", [])}
    except (json.JSONDecodeError, KeyError):
        match = re.search(r'"top_thread_ids":\s*\[(.*?)\]', result["data"]["choices"][0]["message"]["content"], re.DOTALL)
        if match:
            ids = [str(id.strip().strip('"')) for id in match.group(1).split(',') if id.strip() and str(id.strip().strip('"')) in [str(t["thread_id"]) for t in threads]]
            return {"top_thread_ids": ids, "reason": "Extracted from partial response", "intent_breakdown": []}
        return {"top_thread_ids": [], "reason": "Failed to parse response", "intent_breakdown": []}

async def filter_replies(replies, max_replies, source_type, thread_id):
    """Filter and sort thread replies."""
    if not isinstance(replies, list):
        logger.warning(f"Invalid replies format for thread {thread_id}: expected list, got {type(replies)}")
        return []
    
    filtered_replies = [
        {
            "reply_id": r.get("reply_id"),
            "msg": clean_content(r["msg"]),
            "like_count": r.get("like_count", 0),
            "dislike_count": r.get("dislike_count", 0) if source_type == "lihkg" else 0,
            "reply_time": unix_to_readable(r.get("reply_time", "0"), context=f"reply in thread {thread_id}")
        }
        for r in replies
        if isinstance(r, dict) and r.get("msg") and len(clean_content(r["msg"]).strip()) > 7 and clean_content(r["msg"]) not in ["[圖片]", "[無內容]", "[表情符號]"]
    ]
    return sorted(filtered_replies, key=lambda x: x.get("like_count", 0), reverse=True)[:max_replies]

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_source, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, source_id=None, source_type="lihkg"):
    """Stream Grok3 response for query processing."""
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 10, "min_likes": 0}
    selected_source = normalize_selected_source(selected_source, source_type)
    
    if not thread_data:
        error_msg = f"No matching threads found in {selected_source['source_name']} (filters: {json.dumps(filters, ensure_ascii=False)})."
        logger.warning(error_msg)
        yield error_msg
        return
    
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Missing Grok 3 API key")
        yield "Error: Missing API key"
        return
    
    intents_info = processing.get('analysis', {}).get('intents', [{"intent": "summarize_posts", "confidence": 0.7, "reason": "Default intent"}])
    intents = [i['intent'] for i in intents_info]
    primary_intent = max(intents_info, key=lambda x: x["confidence"])["intent"]
    
    max_replies_per_thread = 100
    if any(intent in ["follow_up", "fetch_thread_by_id"] for intent in intents):
        max_replies_per_thread = 300
    elif any(intent == "analyze_sentiment" for intent in intents):
        max_replies_per_thread = 200
    elif any(intent == "list_titles" for intent in intents):
        max_replies_per_thread = 10
    
    filtered_thread_data = {}
    total_replies_count = 0
    for tid, data in (thread_data.items() if isinstance(thread_data, dict) else {str(d["thread_id"]): d for d in thread_data}.items()):
        replies = await filter_replies(data.get("replies", []), max_replies_per_thread, source_type, tid)
        total_replies_count += len(replies)
        filtered_thread_data[tid] = {
            **data,
            "replies": replies,
            "total_fetched_replies": len(replies),
            "last_reply_time": unix_to_readable(data.get("last_reply_time", "0"), context=f"thread {tid}")
        }
    
    prompt = await build_dynamic_prompt(user_query, conversation_context, metadata, list(filtered_thread_data.values()), filters, primary_intent, selected_source, api_key)
    prompt_length = len(prompt)
    
    if prompt_length > GROK3_TOKEN_LIMIT:
        logger.error(f"Prompt length {prompt_length} exceeds limit {GROK3_TOKEN_LIMIT}")
        yield "Error: Prompt too large. Please reduce query scope."
        return
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是社交媒體數據助手，以繁體中文回答，語氣客觀輕鬆，使用 [帖子 ID: {thread_id}] 格式引用帖子。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": min(8000, int((500 + 1500 * (prompt_length / GROK3_TOKEN_LIMIT)) / 0.8)),
        "temperature": 0.7,
        "stream": True
    }
    
    response = await call_grok3_api(payload, headers, stream=True)
    if isinstance(response, dict) and response.get("error"):
        yield f"Error: API call failed ({response['error']})."
        return
    
    response_content = ""
    async for line in response.content:
        if line and not line.isspace():
            line_str = line.decode('utf-8').strip()
            if line_str == "data: [DONE]":
                break
            if line_str.startswith("data:"):
                try:
                    chunk = json.loads(line_str[6:])
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                         content = clean_content(content, is_response=True)
                    response_content += content
                    yield content
                except json.JSONDecodeError:
                    continue

async def process_user_question(user_query, selected_source, source_id, source_type="lihkg", analysis=None, request_counter=0, last_reset=0, rate_limit_until=0, conversation_context=None, progress_callback=None):
    """Process user query and fetch relevant threads."""
    selected_source = normalize_selected_source(selected_source, source_type)
    clean_cache()
    
    if rate_limit_until > time.time():
        return {"selected_source": selected_source, "thread_data": [], "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}], "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until, "analysis": analysis}
    
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        return {"selected_source": selected_source, "thread_data": [], "rate_limit_info": [{"message": "Missing API key"}], "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until, "analysis": analysis}
    
    analysis = analysis or await analyze_and_screen(user_query, selected_source["source_name"], source_id, source_type, conversation_context)
    post_limit = min(analysis.get("post_limit", 5), 20)
    top_thread_ids = list(set(analysis.get("top_thread_ids", [])))
    intents = [i["intent"] for i in analysis.get("intents", [{"intent": "summarize_posts", "confidence": 0.7}])]
    
    keyword_result = await extract_keywords(user_query, conversation_context, api_key)
    sort = "new" if keyword_result.get("time_sensitive", False) else "best"
    max_replies = 300 if any(i in ["follow_up", "analyze_sentiment"] for i in intents) else 100
    max_comments = 300 if source_type == "reddit" and "follow_up" in intents else 100
    
    thread_data = []
    rate_limit_info = []
    processed_thread_ids = set()
    
    if top_thread_ids and any(i in ["fetch_thread_by_id", "follow_up", "analyze_sentiment"] for i in intents):
        for thread_id in top_thread_ids:
            if str(thread_id) in processed_thread_ids:
                continue
            processed_thread_ids.add(str(thread_id))
            thread_info = await get_or_update_cache(thread_id, source_type, source_id, max_replies, 1 if keyword_result.get("time_sensitive", False) else 0, max_comments)
            if thread_info:
                thread_data.append(thread_info)
                rate_limit_info.extend(thread_info.get("rate_limit_info", []))
    
    if len(thread_data) < 5 and "follow_up" in intents:
        supplemental_result = await (get_lihkg_topic_list(cat_id=source_id, start_page=1, max_pages=2) if source_type == "lihkg" else get_reddit_topic_list(subreddit=source_id, start_page=1, max_pages=2, sort=sort))
        supplemental_threads = [item for item in supplemental_result.get("items", []) if str(item["thread_id"]) not in top_thread_ids and any(kw.lower() in item["title"].lower() for kw in keyword_result["keywords"])][:5 - len(thread_data)]
        for item in supplemental_threads:
            thread_id = str(item["thread_id"])
            if thread_id in processed_thread_ids:
                continue
            processed_thread_ids.add(thread_id)
            thread_info = await get_or_update_cache(thread_id, source_type, source_id, max_replies, 1 if keyword_result.get("time_sensitive", False) else 0, max_comments)
            if thread_info:
                thread_data.append(thread_info)
                rate_limit_info.extend(thread_info.get("rate_limit_info", []))
    
    return {
        "selected_source": selected_source,
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until,
        "analysis": analysis
    }

def clean_cache(max_age=3600):
    """Clean expired or excess cache entries."""
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]
    
    if len(st.session_state.thread_cache) > MAX_CACHE_SIZE:
        sorted_keys = sorted(st.session_state.thread_cache.items(), key=lambda x: x[1]["timestamp"])[:len(st.session_state.thread_cache) - MAX_CACHE_SIZE]
        for key, _ in sorted_keys:
            del st.session_state.thread_cache[key]
        logger.info(f"Cleaned cache, removed {len(sorted_keys)} entries, current size: {len(st.session_state.thread_cache)}")

def configure_lihkg_api_logger():
    configure_logger("lihkg_api", "lihkg_api.log")

def configure_reddit_api_logger():
    configure_logger("reddit_api", "reddit_api.log")
