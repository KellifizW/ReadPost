"""
核心處理模組，分析 LIHKG 數據並生成回應，內嵌提示詞模板。
"""

import aiohttp
import asyncio
import json
import re
import datetime
import time
import logging
import streamlit as st
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
import pytz

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_KONG")

# 日誌配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("grok_processing.log")
    ]
)
logger = logging.getLogger(__name__)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90

# 簡化提示詞模板（內嵌）
PROMPTS = {
    "system": {
        "analyze": "你是 LIHKG 論壇的語義分析助手，以繁體中文回答，專注於理解用戶意圖並篩選討論區數據。",
        "response": "你是 LIHKG 論壇的數據助手，以繁體中文回答，語氣客觀輕鬆，引用帖子使用 [帖子 ID: {thread_id}] 格式。",
        "prioritize": "你是 LIHKG 論壇的帖子排序助手，以繁體中文回答，專注於根據問題語義排序帖子，引用帖子使用 [帖子 ID: {thread_id}] 格式。"
    },
    "analyze": {
        "context": "問題：{query}\n分類：{cat_name}（cat_id={cat_id})\n對話歷史：{conversation_context}",
        "data": "帖子標題：{thread_titles}\n元數據：{metadata}\n帖子數據：{thread_data}",
        "instructions": """步驟：
1. 分析問題意圖，動態分類：
   - list_titles：問題包含「列出」「標題」。
   - summarize_posts：問題包含「分析」「總結」「討論」或提及版塊/主題。
   - analyze_sentiment：問題包含「情緒」「態度」。
   - general_query：問題與 LIHKG 無關或模糊。
   - find_themed：問題指定主題，如「時事」「搞笑」。
   - search_keywords：問題包含「搜索」「關鍵詞」。
   - recommend_threads：問題包含「推薦」「熱門」。
   - follow_up：問題包含「詳情」「更多」「為什麼」或與前問題/回應標題有語義重疊。
2. 若問題模糊，延續對話歷史主題，默認 summarize_posts。
3. 輸出 JSON：{{
  "direct_response": true/false,
  "intent": "意圖",
  "theme": "主題詞",
  "category_ids": [cat_id],
  "data_type": "titles|replies|both|none",
  "post_limit": 5-20,
  "reply_limit": 0-500,
  "filters": {{"min_replies": 0, "min_likes": 0, "sort": "popular", "keywords": []}},
  "top_thread_ids": [],
  "needs_advanced_analysis": false,
  "reason": "分析原因",
  "theme_keywords": []
}}"""
    },
    "prioritize": {
        "context": "問題：{query}\n分類：{cat_name}（cat_id={cat_id})",
        "data": "帖子數據：{threads}",
        "instructions": """任務：
1. 按相關性排序帖子，選最多10個 thread_id，格式 [帖子 ID: {thread_id}]。
2. 若問題提及「熱門」，優先 no_of_reply 和 like_count。
3. 若問題提及「最新」，優先帖子時間。
4. 否則按加權平均（0.6 * no_of_reply + 0.4 * like_count）排序。
5. 輸出 JSON：{{"top_thread_ids": [], "reason": "排序原因"}}"""
    },
    "response": {
        "general": {
            "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
            "data": "帖子元數據：{metadata}\n篩選條件：{filters}",
            "instructions": """任務：
1. 若問題與 LIHKG 相關，生成簡化總結，基於元數據推測話題，字數200-300。
2. 若問題無關 LIHKG，回答上下文相關內容，字數200-400。
3. 若無數據，回應：「在 {selected_cat} 中未找到符合條件的帖子。」"""
        },
        "summarize": {
            "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
            "data": "帖子元數據：{metadata}\n帖子內容：{thread_data}\n篩選條件：{filters}",
            "instructions": """任務：
1. 總結最多5個帖子，引用高點讚回覆，聚焦問題主題。
2. 格式：[帖子 ID: {thread_id}] {標題}。
3. 字數：300-500。
4. 若無數據，回應：「在 {selected_cat} 中未找到符合條件的帖子。」"""
        },
        "follow_up": {
            "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
            "data": "帖子元數據：{metadata}\n帖子內容：{thread_data}\n篩選條件：{filters}",
            "instructions": """任務：
1. 針對歷史帖子 ID 深入分析，引用高點讚或最新回覆。
2. 格式：[帖子 ID: {thread_id}] {標題}。
3. 字數：500-1500。
4. 若無數據，回應：「未能找到您提到的帖子。」"""
        }
    }
}

def clean_html(text):
    """清理 HTML 標籤"""
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            if "hkgmoji" in text.lower():
                return "[表情符號]"
            if any(ext in text.lower() for ext in ['.webp', '.jpg', '.png']):
                return "[圖片]"
            return "[無內容]"
        return text
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}")
        return text

def clean_response(response):
    """清理 [post_id: ...] 字串"""
    if not isinstance(response, str):
        return response
    return re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)

def extract_keywords(query):
    """提取關鍵詞"""
    stop_words = {"的", "是", "在", "有", "什麼", "嗎", "請問"}
    words = re.findall(r'\w+', query)
    return [word for word in words if word not in stop_words][:3]

async def summarize_context(conversation_context):
    """提煉對話歷史主題"""
    if not conversation_context:
        return {"theme": "general", "keywords": []}
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {"theme": "general", "keywords": []}
    
    prompt = f"""
    分析對話歷史，提煉主題和關鍵詞（最多3個）。
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
                    logger.warning(f"Context summarization failed: status={response.status}")
                    return {"theme": "general", "keywords": []}
                data = await response.json()
                return json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:
        logger.warning(f"Context summarization error: {str(e)}")
        return {"theme": "general", "keywords": []}

class PromptBuilder:
    """提示詞生成器"""
    def build_analyze(self, query, cat_name, cat_id, conversation_context=None, thread_titles=None, metadata=None, thread_data=None):
        config = PROMPTS["analyze"]
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
        return f"{PROMPTS['system']['analyze']}\n{context}\n{data}\n{config['instructions']}"

    def build_prioritize(self, query, cat_name, cat_id, threads):
        config = PROMPTS["prioritize"]
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id
        )
        data = config["data"].format(
            threads=json.dumps(threads, ensure_ascii=False)
        )
        return f"{PROMPTS['system']['prioritize']}\n{context}\n{data}\n{config['instructions']}"

    def build_response(self, intent, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        config = PROMPTS["response"].get(intent, PROMPTS["response"]["general"])
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
        return f"{PROMPTS['system']['response']}\n{context}\n{data}\n{config['instructions']}"

async def analyze_and_screen(user_query, cat_name, cat_id, conversation_context=None):
    """分析問題意圖"""
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "general")
    historical_keywords = context_summary.get("keywords", [])
    
    query_words = set(extract_keywords(user_query))
    is_vague = len(query_words) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題"])
    
    is_follow_up = False
    referenced_thread_ids = []
    if conversation_context and len(conversation_context) >= 2:
        last_response = conversation_context[-1].get("content", "")
        matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        referenced_thread_ids = matches
        common_words = query_words.intersection(set(extract_keywords(last_response)))
        explicit_follow_up = any(keyword in user_query for keyword in ["詳情", "更多", "為什麼"])
        if len(common_words) >= 1 or explicit_follow_up:
            is_follow_up = True
            logger.info(f"Follow-up detected: {referenced_thread_ids}, common_words: {common_words}")
    
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
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "Missing API key",
            "theme_keywords": historical_keywords
        }
    
    prompt = prompt_builder.build_analyze(
        query=user_query,
        cat_name=cat_name,
        cat_id=cat_id,
        conversation_context=conversation_context
    )
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    logger.warning(f"Intent analysis failed: status={response.status}")
                    return {
                        "direct_response": False,
                        "intent": "summarize_posts",
                        "theme": historical_theme,
                        "category_ids": [cat_id],
                        "data_type": "both",
                        "post_limit": 5,
                        "reply_limit": 100,
                        "filters": {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5, "keywords": historical_keywords},
                        "top_thread_ids": [],
                        "needs_advanced_analysis": False,
                        "reason": "API failure",
                        "theme_keywords": historical_keywords
                    }
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                intent = result.get("intent", "summarize_posts")
                if is_vague:
                    intent = "summarize_posts"
                    result["reason"] = f"問題模糊，延續主題：{historical_theme}"
                if is_follow_up:
                    intent = "follow_up"
                    result["reason"] = "檢測到追問"
                    result["top_thread_ids"] = referenced_thread_ids
                result["needs_advanced_analysis"] = False
                return result
    except Exception as e:
        logger.warning(f"Intent analysis error: {str(e)}")
        return {
            "direct_response": False,
            "intent": "summarize_posts",
            "theme": historical_theme,
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 100,
            "filters": {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5, "keywords": historical_keywords},
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": f"Analysis failed: {str(e)}",
            "theme_keywords": historical_keywords
        }

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent):
    """排序帖子"""
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {"top_thread_ids": [], "reason": "Missing API key"}
    
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
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    logger.warning(f"Prioritization failed: status={response.status}")
                    sorted_threads = sorted(threads, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
                    return {"top_thread_ids": [t["thread_id"] for t in sorted_threads[:5]], "reason": "Fallback to popularity"}
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                return result
    except Exception as e:
        logger.warning(f"Prioritization error: {str(e)}")
        sorted_threads = sorted(threads, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
        return {"top_thread_ids": [t["thread_id"] for t in sorted_threads[:5]], "reason": f"Prioritization failed: {str(e)}"}

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, cat_id=None):
    """生成流式回應"""
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5}
    prompt_builder = PromptBuilder()
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        yield "錯誤: 缺少 API 密鑰"
        return
    
    intent = processing.get('intent', 'summarize') if isinstance(processing, dict) else processing
    max_replies_per_thread = 100 if intent in ["summarize", "follow_up"] else 25
    
    filtered_thread_data = {}
    total_replies_count = 0
    for tid, data in thread_data.items():
        replies = sorted(
            [r for r in data.get("replies", []) if r.get("msg") and r.get("msg") != "[無內容]"],
            key=lambda x: x.get("like_count", 0),
            reverse=True
        )[:max_replies_per_thread]
        total_replies_count += len(replies)
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
    
    min_tokens = 1200
    max_tokens = 3600
    target_tokens = min_tokens if total_replies_count == 0 else min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)
    
    prompt = prompt_builder.build_response(
        intent=intent,
        query=user_query,
        selected_cat=selected_cat,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=filtered_thread_data,
        filters=filters
    ) + "\n回應需包含 [帖子 ID: xxx]，禁止 [post_id: ...]。"
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": PROMPTS["system"]["response"]}, *conversation_context, {"role": "user", "content": prompt}],
        "max_tokens": target_tokens,
        "temperature": 0.7,
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    yield f"錯誤：API 請求失敗（狀態碼 {response.status}）。"
                    return
                response_content = ""
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
                                    cleaned_content = clean_response(content)
                                    response_content += cleaned_content
                                    yield cleaned_content
                            except json.JSONDecodeError:
                                continue
                if not response_content and metadata:
                    fallback_response = f"以下是 {selected_cat} 的概述：討論涵蓋多主題。[帖子 ID: {list(thread_data.keys())[0] if thread_data else '無'}]"
                    yield clean_response(fallback_response)
        except Exception as e:
            yield f"錯誤：生成回應失敗（{str(e)}）。"
        finally:
            await session.close()

def unix_to_readable(timestamp):
    """轉換時間戳"""
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "1970-01-01 00:00:00"

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, conversation_context=None, progress_callback=None):
    """處理用戶問題"""
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
    
    post_limit = min(analysis.get("post_limit", 5), 10)
    reply_limit = analysis.get("reply_limit", 100)
    filters = analysis.get("filters", {})
    min_replies = filters.get("min_replies", 0)
    min_likes = filters.get("min_likes", 0 if cat_id in ["5", "15"] else 5)
    top_thread_ids = analysis.get("top_thread_ids", [])
    intent = analysis.get("intent", "summarize_posts")
    
    thread_data = []
    rate_limit_info = []
    
    if top_thread_ids:
        candidate_threads = [{"thread_id": tid, "title": "", "no_of_reply": 0, "like_count": 0} for tid in top_thread_ids]
    else:
        initial_threads = []
        for page in range(1, 3):
            result = await get_lihkg_topic_list(cat_id=cat_id, start_page=page, max_pages=1)
            request_counter = result.get("request_counter", request_counter)
            last_reset = result.get("last_reset", last_reset)
            rate_limit_until = result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(result.get("rate_limit_info", []))
            items = result.get("items", [])
            for item in items:
                item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
            initial_threads.extend(items)
            if progress_callback:
                progress_callback(f"已抓取第 {page}/2 頁帖子", 0.1 + 0.2 * (page / 2))
        
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
        
        prioritization = await prioritize_threads_with_grok(user_query, filtered_items, selected_cat, cat_id, intent)
        top_thread_ids = prioritization.get("top_thread_ids", [])
        candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
    
    tasks = []
    for item in candidate_threads:
        thread_id = str(item["thread_id"])
        if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
            thread_data.append(st.session_state.thread_cache[thread_id]["data"])
            continue
        tasks.append(get_lihkg_thread_content(thread_id=thread_id, cat_id=cat_id, max_replies=reply_limit))
    
    if tasks:
        content_results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(content_results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch thread {candidate_threads[idx]['thread_id']}: {str(result)}")
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
                    "last_reply_time": unix_to_readable(candidate_threads[idx].get("last_reply_time", "0")),
                    "like_count": candidate_threads[idx].get("like_count", 0),
                    "dislike_count": candidate_threads[idx].get("dislike_count", 0),
                    "replies": cleaned_replies,
                    "fetched_pages": result.get("fetched_pages", [])
                }
                thread_data.append(thread_info)
                st.session_state.thread_cache[thread_id] = {
                    "data": thread_info,
                    "timestamp": time.time()
                }
    
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
