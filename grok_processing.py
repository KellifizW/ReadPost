"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
包含數據處理邏輯（進階分析、緩存管理）和輔助函數。
主要函數：
- analyze_and_screen：分析問題，識別細粒度意圖，動態設置篩選條件。
- stream_grok3_response：生成流式回應，根據意圖和分類動態選擇模板。
- process_user_question：處理用戶問題，抓取並分析帖子。
- clean_html：清理 HTML 標籤。
"""

import aiohttp
import asyncio
import json
import re
import random
import math
import time
import logging
import streamlit as st
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(function)s - %(message)s")

# 檔案處理器：寫入 app.log
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台處理器：輸出到 stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 30  # 秒

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
        logger.error(f"HTML cleaning failed: {str(e)}", extra={"function": "clean_html"})
        return text

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，識別細粒度意圖，動態設置篩選條件。
    依賴 Grok 3 進行意圖分類和參數設置，減少硬編碼。
    """
    conversation_context = conversation_context or []
    prompt = f"""
    你是 LIHKG 論壇的集體意見代表，根據用戶問題和提供的數據，以繁體中文回覆，模擬論壇用戶的語氣。輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else ''}
    {'元數據：' + json.dumps(metadata, ensure_ascii=False) if metadata else ''}
    {'回覆數據：' + json.dumps(thread_data, ensure_ascii=False) if thread_data else ''}

    步驟：
    1. 分析問題意圖，動態分類（例如：列出帖子、總結內容、分析情緒、一般問題，或其他自定義意圖）。
    2. 若問題不需抓取帖子（例如一般問題），設置 direct_response=True，category_ids=[]。
    3. 若問題涉及「熱門」，根據回覆數、點讚數等標準，選最多10個帖子ID（top_thread_ids）。
    4. 動態確定帖子數量（post_limit，1-20）、回覆數量（reply_limit，0-500）。
    5. 設置篩選條件（filters，例如 min_replies, min_likes），根據問題語義動態調整。
    6. 若為進階分析（is_advanced=True），檢查是否需要更多帖子或回覆，設置 needs_advanced_analysis。
    7. 提供候選帖子ID（candidate_thread_ids）和處理方式（processing，例如 list, summarize, sentiment）。
    8. 若無法確定參數，提供原因（reason）。

    輸出格式：
    {{
        "direct_response": boolean,
        "intent": string,
        "theme": string,
        "category_ids": array,
        "data_type": string (title_only, replies, both),
        "post_limit": integer,
        "reply_limit": integer,
        "filters": object,
        "processing": string,
        "candidate_thread_ids": array,
        "top_thread_ids": array,
        "needs_advanced_analysis": boolean,
        "reason": string
    }}
    示例：
    {{
        "direct_response": false,
        "intent": "summarize_posts",
        "theme": "熱門",
        "category_ids": ["{cat_id}"],
        "data_type": "both",
        "post_limit": 5,
        "reply_limit": 100,
        "filters": {{"min_replies": 50, "min_likes": 20}},
        "processing": "summarize",
        "candidate_thread_ids": [],
        "top_thread_ids": [],
        "needs_advanced_analysis": false,
        "reason": ""
    }}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}", extra={"function": "analyze_and_screen"})
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
            "reason": "Missing API key"
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": "你是由 xAI 創建的 Grok 3，代表 LIHKG 論壇的集體意見，以繁體中文回答。根據問題語義和提供數據直接回應，無需提及身份或語氣。"},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 300,  # 增加 token 以支持更詳細的分析
        "temperature": 0.7
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    data = await response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    # 確保必要字段存在，設置合理默認值
                    result.setdefault("direct_response", False)
                    result.setdefault("intent", "general_query")
                    result.setdefault("theme", "")
                    result.setdefault("category_ids", [cat_id] if not result.get("direct_response") else [])
                    result.setdefault("data_type", "both")
                    result.setdefault("post_limit", 5)
                    result.setdefault("reply_limit", 100)
                    result.setdefault("filters", {"min_replies": 50, "min_likes": 20})
                    result.setdefault("processing", "summarize")
                    result.setdefault("candidate_thread_ids", [])
                    result.setdefault("top_thread_ids", [])
                    result.setdefault("needs_advanced_analysis", False)
                    result.setdefault("reason", "")
                    logger.info(
                        json.dumps({
                            "event": "grok3_api_call",
                            "function": "analyze_and_screen",
                            "query": user_query,
                            "status": "success",
                            "intent": result["intent"],
                            "needs_advanced_analysis": result["needs_advanced_analysis"],
                            "filters": result["filters"]
                        }, ensure_ascii=False),
                        extra={"function": "analyze_and_screen"}
                    )
                    return result
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.warning(
                json.dumps({
                    "event": "grok3_api_call",
                    "function": "analyze_and_screen",
                    "query": user_query,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error": str(e)
                }, ensure_ascii=False),
                extra={"function": "analyze_and_screen"}
            )
            if attempt < max_retries - 1:
                # 使用備用提示詞，簡化要求
                simplified_prompt = f"""
                你是 LIHKG 論壇助手，分析用戶問題，輸出 JSON。
                問題：{user_query}
                分類：{cat_name}（cat_id={cat_id})
                步驟：
                1. 判斷意圖（例如：列出帖子、總結、情緒分析、一般問題）。
                2. 設置帖子數量（post_limit，1-20）、回覆數量（reply_limit，0-500）。
                3. 設置篩選條件（filters，例如 min_replies, min_likes）。
                輸出：
                {{"direct_response": boolean, "intent": string, "theme": string, "category_ids": array, "data_type": string, "post_limit": integer, "reply_limit": integer, "filters": object, "processing": string, "candidate_thread_ids": array, "top_thread_ids": array, "needs_advanced_analysis": boolean, "reason": string}}
                """
                messages[-1]["content"] = simplified_prompt
                payload["max_tokens"] = 200
                await asyncio.sleep(2)
                continue
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
                "reason": f"Analysis failed after {max_retries} attempts: {str(e)}"
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None):
    """
    使用 Grok 3 API 生成流式回應，根據意圖和分類動態選擇模板。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 50, "min_likes": 20}
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}", extra={"function": "stream_grok3_response"})
        yield "錯誤: 缺少 API 密鑰"
        return
    
    filtered_thread_data = {
        tid: {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": [r for r in data.get("replies", []) if r.get("like_count", 0) >= 5][:25],
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items()
    }
    
    prompt_templates = {
        "list": f"""
        你是 LIHKG 論壇的數據助手，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子元數據：{json.dumps(metadata, ensure_ascii=False)}
        篩選條件：{json.dumps(filters, ensure_ascii=False)}
        僅列出帖子標題，格式為：
        - 帖子 ID: [thread_id] 標題: [title]
        若無符合條件的帖子，回答：「無符合條件的帖子（篩選：回覆數≥{filters.get('min_replies', 0)}，點讚數≥{filters.get('min_likes', 0)}）。」並列出最多5個最新帖子標題（無篩選）。
        若無任何帖子，回答：「目前無可用帖子標題。」
        輸出：標題列表
        """,
        "summarize": f"""
        你是 LIHKG 論壇的集體意見代表，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        引用高關注回覆（like_count≥5），總結帖子內容，300-500字。
        輸出：總結
        """,
        "sentiment": f"""
        你是 LIHKG 論壇的集體意見代表，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        判斷情緒分佈（正面、負面、中立），聚焦高關注回覆（like_count≥5）。
        輸出：情緒分析：正面XX%，負面XX%，中立XX%\n依據：...
        """,
        "general": f"""
        你是 Grok 3，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        根據問題語義提供直接回應，聚焦問題核心（50-100 字）。
        輸出：直接回應
        """
    }
    
    if not metadata and not filtered_thread_data:
        prompt = f"""
        你是 Grok 3，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        若問題要求列出數據類型，回答：「我可以抓取 LIHKG 討論區的帖子標題、帖子 ID、回覆數量、最後回覆時間、點贊數、踩數，及部分回覆內容、回覆點贊與踩數、回覆時間。若需具體帖子分析，請提供更多細節！」（100-150字）。
        若問題為一般問題，直接回應。
        若無帖子數據且問題要求帖子，回答：「目前無可用帖子標題。」
        輸出：直接回應
        """
    else:
        prompt = prompt_templates.get(processing, prompt_templates["general"])
    
    if len(prompt) > GROK3_TOKEN_LIMIT:
        for tid in filtered_thread_data:
            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:10]
        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
        logger.info(f"Truncated prompt: {len(prompt)} characters", extra={"function": "stream_grok3_response"})
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": "你是由 xAI 創建的 Grok 3，代表 LIHKG 論壇的集體意見，以繁體中文回答。根據問題語義和提供數據直接回應，無需提及身份或語氣。"},
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
    
    response_content = ""
    async with aiohttp.ClientSession() as session:
        try:
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
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
                                            if "###" in content:
                                                logger.warning(f"Detected ### in response, retrying", extra={"function": "stream_grok3_response"})
                                                raise ValueError("Content moderation detected")
                                            response_content += content
                                            yield content
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON decode error in stream chunk: {str(e)}", extra={"function": "stream_grok3_response"})
                                        continue
                        if needs_advanced_analysis and metadata and filtered_thread_data:
                            yield f"\n建議：為確保分析全面，建議抓取更多帖子頁數。{reason}\n"
                        logger.info(
                            json.dumps({
                                "event": "grok3_api_call",
                                "function": "stream_grok3_response",
                                "query": user_query,
                                "status": "success"
                            }, ensure_ascii=False),
                            extra={"function": "stream_grok3_response"}
                        )
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Grok 3 request failed, attempt {attempt+1}: {str(e)}", extra={"function": "stream_grok3_response"})
                    if attempt < 2:
                        for tid in filtered_thread_data:
                            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:5]
                        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
                        payload["messages"][-1]["content"] = prompt
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    raise
        except Exception as e:
            logger.error(
                json.dumps({
                    "event": "grok3_api_call",
                    "function": "stream_grok3_response",
                    "query": user_query,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error": str(e)
                }, ensure_ascii=False),
                extra={"function": "stream_grok3_response"}
            )
            yield f"錯誤：生成回應失敗（{str(e)}），請稍後重試"
        finally:
            await session.close()

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None):
    """
    處理用戶問題，抓取並分析 LIHKG 帖子。
    """
    try:
        logger.info(f"Processing query: {user_question}, category: {selected_cat}, cat_id: {cat_id}", extra={"function": "process_user_question"})
        
        if rate_limit_until > time.time():
            logger.warning(f"Rate limit active until {rate_limit_until}", extra={"function": "process_user_question"})
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        if analysis.get("direct_response", True) or analysis.get("intent") == "list_titles":
            thread_data = []
            rate_limit_info = []
            if analysis.get("intent") == "list_titles":
                initial_threads = []
                for page in range(1, 4):
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
                    initial_threads.extend(items)
                    logger.info(
                        json.dumps({
                            "event": "thread_fetch",
                            "function": "process_user_question",
                            "cat_id": cat_id,
                            "page": page,
                            "items_fetched": len(items)
                        }, ensure_ascii=False),
                        extra={"function": "process_user_question"}
                    )
                    if not items:
                        logger.warning(f"No threads fetched for cat_id={cat_id}, page={page}", extra={"function": "process_user_question"})
                    if len(initial_threads) >= 90:
                        initial_threads = initial_threads[:90]
                        break
                
                filters = analysis.get("filters", {})
                min_replies = filters.get("min_replies", 0)
                min_likes = filters.get("min_likes", 0)
                previous_thread_ids = previous_thread_ids or []
                
                filtered_items = [
                    item for item in initial_threads
                    if item.get("no_of_reply", 0) >= min_replies
                    and int(item.get("like_count", 0)) >= min_likes
                    and str(item["thread_id"]) not in previous_thread_ids
                ]
                
                post_limit = min(analysis.get("post_limit", 5), 20)
                if not filtered_items:
                    logger.warning(
                        f"No threads meet filters: min_replies={min_replies}, min_likes={min_likes}, trying without filters",
                        extra={"function": "process_user_question"}
                    )
                    filtered_items = [
                        item for item in initial_threads
                        if str(item["thread_id"]) not in previous_thread_ids
                    ]
                
                thread_data = [
                    {
                        "thread_id": str(item["thread_id"]),
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item.get("last_reply_time", 0),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [],
                        "fetched_pages": []
                    } for item in filtered_items[:post_limit]
                ]
                for item in thread_data:
                    thread_id = item["thread_id"]
                    if thread_id not in st.session_state.thread_cache:
                        st.session_state.thread_cache[thread_id] = {
                            "data": item,
                            "timestamp": time.time()
                        }
                logger.info(
                    json.dumps({
                        "event": "thread_preparation",
                        "function": "process_user_question",
                        "query": user_question,
                        "initial_threads": len(initial_threads),
                        "filtered_items": len(filtered_items),
                        "thread_data": len(thread_data),
                        "filters": filters
                    }, ensure_ascii=False),
                    extra={"function": "process_user_question"}
                )
            
            return {
                "selected_cat": selected_cat,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        post_limit = min(analysis.get("post_limit", 5), 20)
        reply_limit = analysis.get("reply_limit", 100)  # 完全依賴 analyze_and_screen
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 0)
        min_likes = filters.get("min_likes", 0)
        candidate_thread_ids = analysis.get("candidate_thread_ids", [])
        top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
        
        thread_data = []
        rate_limit_info = []
        previous_thread_ids = previous_thread_ids or []
        
        if is_advanced and previous_thread_ids:
            for thread_id in previous_thread_ids:
                cached_data = previous_thread_data.get(thread_id) if previous_thread_data else None
                if not cached_data:
                    continue
                fetched_pages = cached_data.get("fetched_pages", [])
                existing_replies = cached_data.get("replies", [])
                total_replies = cached_data.get("no_of_reply", 0)
                
                like_count = cached_data.get("like_count", 0)
                no_of_reply = total_replies
                if like_count >= 500 or no_of_reply >= 500:
                    threshold = 0.6
                elif like_count >= 100 or no_of_reply >= 100:
                    threshold = 0.4
                else:
                    threshold = 0.2
                total_pages = (total_replies + 24) // 25
                target_pages = math.ceil(total_pages * threshold)
                remaining_pages = max(0, target_pages - len(fetched_pages))
                
                if remaining_pages <= 0:
                    logger.info(f"Thread {thread_id} meets {int(threshold*100)}% page threshold: {len(fetched_pages)}/{target_pages}", extra={"function": "process_user_question"})
                    thread_data.append({
                        "thread_id": str(thread_id),
                        "title": cached_data.get("title", "未知標題"),
                        "no_of_reply": total_replies,
                        "last_reply_time": cached_data.get("last_reply_time", 0),
                        "like_count": cached_data.get("like_count", 0),
                        "dislike_count": cached_data.get("dislike_count", 0),
                        "replies": existing_replies,
                        "fetched_pages": fetched_pages
                    })
                    continue
                
                start_page = max(fetched_pages, default=1) + 1 if fetched_pages else 1
                thread_result = await get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=cat_id,
                    request_counter=request_counter,
                    last_reset=last_reset,
                    rate_limit_until=rate_limit_until,
                    max_replies=reply_limit,
                    fetch_last_pages=remaining_pages,
                    start_page=start_page
                )
                
                request_counter = thread_result.get("request_counter", request_counter)
                last_reset = thread_result.get("last_reset", last_reset)
                rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(thread_result.get("rate_limit_info", []))
                
                replies = thread_result.get("replies", [])
                all_replies = existing_replies + [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in replies]
                sorted_replies = sorted(all_replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
                all_fetched_pages = sorted(set(fetched_pages + thread_result.get("fetched_pages", [])))
                
                thread_data.append({
                    "thread_id": str(thread_id),
                    "title": thread_result.get("title", cached_data.get("title", "未知標題")),
                    "no_of_reply": thread_result.get("total_replies", total_replies),
                    "last_reply_time": thread_result.get("last_reply_time", cached_data.get("last_reply_time", 0)),
                    "like_count": thread_result.get("like_count", cached_data.get("like_count", 0)),
                    "dislike_count": thread_result.get("dislike_count", cached_data.get("dislike_count", 0)),
                    "replies": sorted_replies,
                    "fetched_pages": all_fetched_pages
                })
                logger.info(f"Advanced thread {thread_id}: replies={len(sorted_replies)}, pages={len(all_fetched_pages)}/{target_pages}", extra={"function": "process_user_question"})
                await asyncio.sleep(1)
            
            logger.info(f"Advanced processing completed: {len(thread_data)} threads", extra={"function": "process_user_question"})
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
        for page in range(1, 4):
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
            initial_threads.extend(items)
            logger.info(
                json.dumps({
                    "event": "thread_fetch",
                    "function": "process_user_question",
                    "cat_id": cat_id,
                    "page": page,
                    "items_fetched": len(items)
                }, ensure_ascii=False),
                extra={"function": "process_user_question"}
            )
            if not items:
                logger.warning(f"No threads fetched for cat_id={cat_id}, page={page}", extra={"function": "process_user_question"})
            if len(initial_threads) >= 90:
                initial_threads = initial_threads[:90]
                break
        
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 0)
        min_likes = filters.get("min_likes", 0)
        filtered_items = [
            item for item in initial_threads
            if item.get("no_of_reply", 0) >= min_replies and int(item.get("like_count", 0)) >= min_likes
            and str(item["thread_id"]) not in previous_thread_ids
        ]
        logger.info(
            json.dumps({
                "event": "thread_filtering",
                "function": "process_user_question",
                "initial_threads": len(initial_threads),
                "filtered_items": len(filtered_items),
                "filters": filters,
                "excluded_thread_ids": previous_thread_ids
            }, ensure_ascii=False),
            extra={"function": "process_user_question"}
        )
        
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            if thread_id not in st.session_state.thread_cache:
                st.session_state.thread_cache[thread_id] = {
                    "data": {
                        "thread_id": thread_id,
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item.get("last_reply_time", 0),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [],
                        "fetched_pages": []
                    },
                    "timestamp": time.time()
                }
        
        if not top_thread_ids and filtered_items:
            # 備用邏輯：若 Grok 3 未提供 top_thread_ids，按熱門程度排序
            sorted_items = sorted(
                filtered_items,
                key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                reverse=True
            )
            top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
            logger.info(f"Generated top_thread_ids based on popularity: {top_thread_ids}", extra={"function": "process_user_question"})
        
        candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
        if not candidate_threads:
            candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items))) if filtered_items else []
            logger.info(f"No candidate threads, using random: {len(candidate_threads)}", extra={"function": "process_user_question"})
        
        for item in candidate_threads:
            thread_id = str(item["thread_id"])
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit,
                fetch_last_pages=0
            )
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
            thread_data.append({
                "thread_id": thread_id,
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
                "fetched_pages": thread_result.get("fetched_pages", [1])
            })
            st.session_state.thread_cache[thread_id]["data"].update({
                "replies": thread_data[-1]["replies"],
                "fetched_pages": thread_data[-1]["fetched_pages"]
            })
            st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
            logger.info(f"Fetched candidate thread {thread_id}: replies={len(replies)}", extra={"function": "process_user_question"})
            await asyncio.sleep(1)
        
        final_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
        if not final_threads:
            final_threads = candidate_threads[:post_limit]
        
        for item in final_threads:
            thread_id = str(item["thread_id"])
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit,
                fetch_last_pages=2
            )
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
            thread_data.append({
                "thread_id": thread_id,
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
                "fetched_pages": thread_result.get("fetched_pages", [1])
            })
            st.session_state.thread_cache[thread_id]["data"].update({
                "replies": thread_data[-1]["replies"],
                "fetched_pages": thread_data[-1]["fetched_pages"]
            })
            st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
            logger.info(f"Fetched final thread {thread_id}: replies={len(replies)}", extra={"function": "process_user_question"})
            await asyncio.sleep(1)
        
        logger.info(f"Processing completed: {len(thread_data)} threads for query: {user_question}", extra={"function": "process_user_question"})
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
        logger.error(
            json.dumps({
                "event": "processing_error",
                "function": "process_user_question",
                "query": user_question,
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e)
            }, ensure_ascii=False),
            extra={"function": "process_user_question"}
        )
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": [{"message": f"Processing failed: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis or {}
        }