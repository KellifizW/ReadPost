"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
包含數據處理邏輯（進階分析、緩存管理）和輔助函數。
主要函數：
- analyze_and_screen：分析問題並篩選帖子。
- stream_grok3_response：生成流式回應。
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
from datetime import datetime
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 日誌過濾器：完全過濾 in-event 日誌
class InEventFilter(logging.Filter):
    def filter(self, record):
        # 完全過濾掉所有 in-event 日誌
        if "in-event" in record.msg.lower():
            return False
        return True

# 配置日誌記錄器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("grok_processing.log")  # 保存日誌到文件
    ]
)
logger = logging.getLogger(__name__)
logger.addFilter(InEventFilter())

# 確保全局日誌級別為 INFO
logging.getLogger().setLevel(logging.INFO)

# 設置日誌時間為香港時區
logging.Formatter.converter = lambda *args: datetime.now(HONG_KONG_TZ).timetuple()

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000

def clean_html(text):
    """
    清理 HTML 標籤，規範化文本。
    Args:
        text (str): 包含 HTML 的文本。
    Returns:
        str: 清理後的純文本。
    """
    if not isinstance(text, str):
        logger.debug(f"Invalid input to clean_html: {text}")
        return ""
    original = text
    clean = re.compile(r'<[^>]+>|[\u200B-\u200F\u202A-\u202E]|[^\w\s.,!?，。！？]')
    text = clean.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    logger.debug(f"clean_html input: {original[:50]}... output: {text[:50]}...")
    return text

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False):
    """
    分析用戶問題並篩選 LIHKG 帖子。
    Args:
        user_query (str): 用戶問題。
        cat_name (str): 分類名稱。
        cat_id (str): 分類 ID。
        thread_titles (list): 帖子標題列表。
        metadata (list): 帖子元數據。
        thread_data (dict): 帖子回覆數據。
        is_advanced (bool): 是否進行進階分析。
    Returns:
        dict: 分析結果，包含主題、篩選參數等。
    """
    prompt = f"""
    你是一個智能助手，分析用戶問題並篩選 LIHKG 帖子。以繁體中文回覆，輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else ''}
    {'元數據：' + json.dumps(metadata, ensure_ascii=False) if metadata else ''}
    {'回覆數據：' + json.dumps(thread_data, ensure_ascii=False) if thread_data else ''}

    步驟：
    1. 識別主題（感動、搞笑、財經等），標記為 theme。
    2. 判斷意圖（總結、情緒分析、幽默總結）。
    3. {'檢查帖子是否達60%頁數（總頁數*0.6，向上取整），若未達標，設置 needs_advanced_analysis=True。' if is_advanced else '篩選帖子：'}
       {'- 若無標題，設置初始抓取（30-90個標題）。' if not thread_titles else '- 從標題選10個候選（candidate_thread_ids），再選top_thread_ids。'}
    4. 設置參數：
       - theme：問題主題。
       - category_ids：[cat_id]。
       - data_type："title"、"replies"、"both"。
       - post_limit：從問題提取（默認2，最大10）。
       - reply_limit：{200 if is_advanced else 75}。
       - filters：根據主題（感動：like_count≥5；搞笑：like_count≥10；財經：like_count≥10；其他：min_replies≥20，min_likes≥10）。
       - processing：emotion_focused_summary、humor_focused_summary、professional_summary、summarize、sentiment。
       - candidate_thread_ids：10個候選ID。
       - top_thread_ids：最終選定ID。
    5. 若無關LIHKG，返回空category_ids。

    輸出：
    {"{ \"needs_advanced_analysis\": false, \"suggestions\": { \"theme\": \"\", \"category_ids\": [], \"data_type\": \"\", \"post_limit\": 0, \"reply_limit\": 0, \"filters\": {}, \"processing\": \"\", \"candidate_thread_ids\": [], \"top_thread_ids\": [] }, \"reason\": \"\" }" if is_advanced else "{ \"theme\": \"\", \"category_ids\": [], \"data_type\": \"\", \"post_limit\": 0, \"reply_limit\": 0, \"filters\": {}, \"processing\": \"\", \"candidate_thread_ids\": [], \"top_thread_ids\": [], \"category_suggestion\": \"\" }"}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {
            "theme": "未知", "category_ids": [cat_id], "data_type": "both", "post_limit": 2,
            "reply_limit": 200 if is_advanced else 75, "filters": {"min_replies": 20, "min_likes": 10},
            "processing": "summarize", "candidate_thread_ids": [], "top_thread_ids": [],
            "category_suggestion": "Missing API key"
        } if not is_advanced else {
            "needs_advanced_analysis": False, "suggestions": {}, "reason": "Missing API key"
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=30) as response:
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                if is_advanced:
                    result["suggestions"]["category_ids"] = [cat_id]
                else:
                    result["category_ids"] = [cat_id]
                logger.info(f"Analysis result: {str(result)[:50]}...")
                return result
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {
            "theme": "未知", "category_ids": [cat_id], "data_type": "both", "post_limit": 2,
            "reply_limit": 200 if is_advanced else 75, "filters": {"min_replies": 20, "min_likes": 10},
            "processing": "summarize", "candidate_thread_ids": [], "top_thread_ids": [],
            "category_suggestion": f"Analysis failed: {str(e)}"
        } if not is_advanced else {
            "needs_advanced_analysis": False, "suggestions": {"category_ids": [cat_id]},
            "reason": f"Analysis failed: {str(e)}"
        }

async def stream_grok3_response(user_query, metadata, thread_data, processing):
    """
    使用 Grok 3 API 生成流式回應。
    Args:
        user_query (str): 用戶問題。
        metadata (list): 帖子元數據。
        thread_data (dict): 帖子回覆數據。
        processing (str): 處理類型（總結、情緒分析等）。
    Yields:
        str: 回應片段。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        yield "錯誤: 缺少 API 密鑰"
        return

    filtered_thread_data = {
        tid: {
            "thread_id": data["thread_id"], "title": clean_html(data["title"]), "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0), "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": [
                {
                    "msg": clean_html(r["msg"]),
                    "like_count": r.get("like_count", 0),
                    "dislike_count": r.get("dislike_count", 0),
                    "reply_time": r.get("reply_time", 0)
                }
                for r in data.get("replies", []) if r.get("like_count", 0) != 0 or r.get("dislike_count", 0) != 0
            ][:25],
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items()
    }

    prompt_templates = {
        "summarize": f"""
        總結 LIHKG 帖子, 300-500字, 結構清晰. 問題: {user_query}
        帖子: {json.dumps(metadata, ensure_ascii=False)}
        回覆: {json.dumps(filtered_thread_data, ensure_ascii=False)}
        要求:
        - 開頭介紹帖子標題、ID和主題.
        - 概述回覆數、點贊數和討論熱度.
        - 引用1-2條高關注回覆（like_count或dislike_count>0）, 說明其內容.
        - 若回覆數據不足, 註明限制並簡化總結.
        - 適配分類語氣（吹水台輕鬆, 創意台溫馨, 財經台專業等）.
        - 結尾總結討論的主要觀點或趨勢.
        - 確保無特殊字符或亂碼.
        輸出: 總結
        """,
        "emotion_focused_summary": f"""
        總結 LIHKG 感動或溫馨帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦感動情緒，引用高關注回覆，適配分類語氣（吹水台輕鬆，創意台溫馨）。
        輸出：總結
        """,
        "humor_focused_summary": f"""
        總結 LIHKG 幽默或搞笑帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦幽默情緒，引用高關注回覆，適配分類語氣（吹水台輕鬆，成人台大膽）。
        輸出：總結
        """,
        "professional_summary": f"""
        總結 LIHKG 財經或時事帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦專業觀點，引用高關注回覆，適配分類語氣（財經台專業，時事台嚴肅）。
        輸出：總結
        """,
        "sentiment": f"""
        分析 LIHKG 帖子情緒。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        判斷情緒分佈（正面、負面、中立），聚焦高關注回覆。
        輸出：情緒分析：正面XX%，負面XX%，中立XX%\n依據：...
        """
    }

    prompt = prompt_templates.get(processing, f"""
    直接回答問題, 50-100字, 結構清晰. 問題: {user_query}
    要求:
    - 簡潔回答用戶問題.
    - 若涉及帖子, 提及標題和簡要觀點.
    - 適配分類語氣.
    - 確保無特殊字符或亂碼.
    輸出: 回應
    """)
    if len(prompt) > GROK3_TOKEN_LIMIT:
        for tid in filtered_thread_data:
            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:10]
        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
        logger.info(f"Truncated prompt: {len(prompt)} characters")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答, 僅基於提供數據, 確保內容連貫無特殊字符."}, {"role": "user", "content": prompt}],
        "max_tokens": 5000,  # 增加 max_tokens 至 5000
        "temperature": 0.3,
        "stream": True
    }

    response_buffer = ""
    response_chunks = []
    start_time = time.time()
    chunk_count = 0
    input_tokens = 0
    output_tokens = 0

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=30) as response:
                    async for line in response.content:
                        if line and not line.isspace():
                            line_str = line.decode('utf-8', errors='replace').strip()
                            if line_str == "data: [DONE]":
                                break
                            if line_str.startswith("data: "):
                                try:
                                    chunk = json.loads(line_str[6:])
                                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content and content not in response_chunks:
                                        content = clean_html(content)  # 清理特殊字符
                                        response_buffer += content
                                        response_chunks.append(content)
                                        chunk_count += 1

                                        # 放寬條件：每 100 字或遇到句號時 yield
                                        if len(response_buffer) >= 100 or content.endswith(('。', '！', '？')):
                                            if chunk_count == 1:
                                                logger.debug(f"Stream chunk started: {response_buffer[:50]}...")
                                            else:
                                                logger.debug(f"Stream chunk: {response_buffer[:50]}...")
                                            yield response_buffer
                                            response_buffer = ""

                                except json.JSONDecodeError:
                                    logger.debug(f"Invalid JSON chunk: {line_str}")
                                    continue
                    # 記錄剩餘的緩衝內容
                    if response_buffer:
                        logger.debug(f"Stream chunk ended: {response_buffer[:50]}...")
                        yield response_buffer
                    # 記錄總結統計，包括 tokens 使用
                    total_response = "".join(response_chunks)
                    duration = time.time() - start_time
                    # 簡單估計 tokens（每 4 字符約 1 token）
                    input_tokens = len(prompt) // 4
                    output_tokens = len(total_response) // 4
                    logger.info(f"Stream response completed: total_chars={len(total_response)}, duration={duration:.3f}s, input_tokens={input_tokens}, output_tokens={output_tokens}")
                    return
        except Exception as e:
            logger.warning(f"Grok 3 request failed, attempt {attempt+1}: {str(e)}")
            if attempt < 2:
                await asyncio.sleep(5 * attempt)
                continue
            yield f"錯誤: 無法連接到 Grok 3 API，請稍後重試 ({str(e)})"
            return

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None):
    """
    處理用戶問題，抓取並分析 LIHKG 帖子。
    Args:
        user_question (str): 用戶問題。
        selected_cat (str): 分類名稱。
        cat_id (str): 分類 ID。
        analysis (dict): 問題分析結果。
        request_counter (int): 請求計數。
        last_reset (float): 最後重置時間。
        rate_limit_until (float): 速率限制解除時間。
        is_advanced (bool): 是否進階分析。
        previous_thread_ids (list): 前次帖子 ID。
        previous_thread_data (dict): 前次帖子數據。
    Returns:
        dict: 處理結果，包含帖子數據、速率限制信息等。
    """
    post_limit = min(analysis.get("post_limit", 2), 10)
    reply_limit = 200 if is_advanced else min(analysis.get("reply_limit", 75), 75)
    
    # 設置唯一的過濾條件：min_replies = 25
    min_replies = 25
    
    thread_data = []
    rate_limit_info = []
    
    # 提取 top_thread_ids
    top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else (previous_thread_ids or [])
    
    if is_advanced and top_thread_ids:
        for thread_id in top_thread_ids:
            cached_data = previous_thread_data.get(thread_id) if previous_thread_data else None
            fetched_pages = cached_data.get("fetched_pages", []) if cached_data else []
            existing_replies = cached_data.get("replies", []) if cached_data else []
            total_replies = cached_data.get("no_of_reply", 0) if cached_data else 0
            
            total_pages = (total_replies + 24) // 25
            target_pages = math.ceil(total_pages * 0.6)
            remaining_pages = max(0, target_pages - len(fetched_pages))
            
            if remaining_pages <= 0:
                logger.info(f"Thread {thread_id} meets 60% page threshold: {len(fetched_pages)}/{target_pages}")
                thread_data.append({
                    "thread_id": str(thread_id), "title": cached_data.get("title", "未知標題"),
                    "no_of_reply": total_replies, "last_reply_time": cached_data.get("last_reply_time", 0),
                    "like_count": cached_data.get("like_count", 0), "dislike_count": cached_data.get("dislike_count", 0),
                    "replies": existing_replies, "fetched_pages": fetched_pages
                })
                continue
            
            start_page = max(fetched_pages, default=1) + 1 if fetched_pages else 1
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id, cat_id=cat_id, request_counter=request_counter, last_reset=last_reset,
                rate_limit_until=rate_limit_until, max_replies=reply_limit, fetch_last_pages=remaining_pages, start_page=start_page
            )
            
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"Invalid thread: {thread_id}")
                continue
            
            all_replies = existing_replies + [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in replies]
            sorted_replies = sorted(all_replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
            all_fetched_pages = sorted(set(fetched_pages + thread_result.get("fetched_pages", [])))
            
            thread_data.append({
                "thread_id": str(thread_id), "title": thread_result.get("title", cached_data.get("title", "未知標題") if cached_data else "未知標題"),
                "no_of_reply": thread_result.get("total_replies", total_replies), "last_reply_time": thread_result.get("last_reply_time", cached_data.get("last_reply_time", 0) if cached_data else 0),
                "like_count": thread_result.get("like_count", cached_data.get("like_count", 0) if cached_data else 0),
                "dislike_count": thread_result.get("dislike_count", cached_data.get("dislike_count", 0) if cached_data else 0),
                "replies": sorted_replies, "fetched_pages": all_fetched_pages
            })
            logger.info(f"Advanced thread {thread_id}: replies={len(sorted_replies)}, pages={len(all_fetched_pages)}/{target_pages}")
            await asyncio.sleep(1)
        
        return {
            "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    initial_threads = []
    for page in range(1, 4):
        result = await get_lihkg_topic_list(cat_id=cat_id, start_page=page, max_pages=1, request_counter=request_counter, last_reset=last_reset, rate_limit_until=rate_limit_until)
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(result.get("rate_limit_info", []))
        initial_threads.extend(result.get("items", []))
        logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(result.get('items', []))}")
        if len(initial_threads) >= 90:
            initial_threads = initial_threads[:90]
            break
    
    # 僅使用 min_replies = 25 作為過濾條件
    filtered_items = [
        item for item in initial_threads
        if item.get("no_of_reply", 0) >= min_replies
    ]
    logger.info(f"Filtered items: {len(filtered_items)} from {len(initial_threads)}")
    
    for item in initial_threads:
        thread_id = str(item["thread_id"])
        st.session_state.thread_cache[thread_id] = {
            "data": {
                "thread_id": thread_id, "title": item["title"], "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0), "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0), "replies": [], "fetched_pages": []
            },
            "timestamp": time.time()
        }
    
    analysis = await analyze_and_screen(user_query=user_question, cat_name=selected_cat, cat_id=cat_id, thread_titles=filtered_items[:90], metadata=None, thread_data=None)
    top_thread_ids = analysis.get("top_thread_ids", [])
    if not top_thread_ids and filtered_items:
        top_thread_ids = [item["thread_id"] for item in random.sample(filtered_items, min(post_limit, len(filtered_items)))]
        logger.warning(f"No top_thread_ids, randomly selected: {top_thread_ids}")
    
    candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
    if not candidate_threads:
        candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items))) if filtered_items else []
        logger.info(f"No candidate threads, using random: {len(candidate_threads)}")
    
    for item in candidate_threads:
        thread_id = str(item["thread_id"])
        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id, cat_id=cat_id, request_counter=request_counter, last_reset=last_reset,
            rate_limit_until=rate_limit_until, max_replies=25, fetch_last_pages=0
        )
        request_counter = thread_result.get("request_counter", request_counter)
        last_reset = thread_result.get("last_reset", last_reset)
        rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(thread_result.get("rate_limit_info", []))
        
        replies = thread_result.get("replies", [])
        if not replies and thread_result.get("total_replies", 0) >= min_replies:
            logger.warning(f"Invalid thread: {thread_id}")
            continue
        
        sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:25]
        thread_data.append({
            "thread_id": thread_id, "title": item["title"], "no_of_reply": item.get("no_of_reply", 0),
            "last_reply_time": item.get("last_reply_time", 0), "like_count": item.get("like_count", 0),
            "dislike_count": item.get("dislike_count", 0),
            "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
            "fetched_pages": thread_result.get("fetched_pages", [1])
        })
        st.session_state.thread_cache[thread_id]["data"].update({
            "replies": thread_data[-1]["replies"], "fetched_pages": thread_data[-1]["fetched_pages"]
        })
        st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
        logger.info(f"Fetched candidate thread {thread_id}: replies={len(replies)}")
        await asyncio.sleep(1)
    
    final_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
    if not final_threads:
        final_threads = candidate_threads[:post_limit]
    
    for item in final_threads:
        thread_id = str(item["thread_id"])
        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id, cat_id=cat_id, request_counter=request_counter, last_reset=last_reset,
            rate_limit_until=rate_limit_until, max_replies=reply_limit, fetch_last_pages=2
        )
        request_counter = thread_result.get("request_counter", request_counter)
        last_reset = thread_result.get("last_reset", last_reset)
        rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(thread_result.get("rate_limit_info", []))
        
        replies = thread_result.get("replies", [])
        if not replies and thread_result.get("total_replies", 0) >= min_replies:
            logger.warning(f"Invalid thread: {thread_id}")
            continue
        
        sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
        thread_data.append({
            "thread_id": thread_id, "title": item["title"], "no_of_reply": item.get("no_of_reply", 0),
            "last_reply_time": item.get("last_reply_time", 0), "like_count": item.get("like_count", 0),
            "dislike_count": item.get("dislike_count", 0),
            "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
            "fetched_pages": thread_result.get("fetched_pages", [1])
        })
        st.session_state.thread_cache[thread_id]["data"].update({
            "replies": thread_data[-1]["replies"], "fetched_pages": thread_data[-1]["fetched_pages"]
        })
        st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
        logger.info(f"Fetched final thread {thread_id}: replies={len(replies)}")
        await asyncio.sleep(1)
    
    return {
        "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
    }
