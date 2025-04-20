"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000

def clean_html(text):
    """
    清理 HTML 標籤，規範化文本。
    """
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None):
    """
    分析用戶問題並篩選 LIHKG 帖子。
    """
    prompt = f"""
    你是一個智能助手，分析用戶問題並篩選 LIHKG 帖子。以繁體中文回覆，輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else ''}

    步驟：
    1. 識別主題（感動、搞笑、財經等），標記為 theme。
    2. 判斷意圖（總結、情緒分析、幽默總結）。
    3. 篩選帖子：
       - 若無標題，設置初始抓取（30-90個標題）。
       - 從標題選10個候選（candidate_thread_ids），再選top_thread_ids。
    4. 設置參數：
       - theme：問題主題。
       - category_ids：[cat_id]。
       - data_type："both"。
       - post_limit：從問題提取（默認2，最大10）。
       - reply_limit：75。
       - filters：根據主題（感動：like_count≥5；搞笑：like_count≥10；財經：like_count≥10；其他：min_replies≥20，min_likes≥10）。
       - processing：emotion_focused_summary、humor_focused_summary、professional_summary、summarize、sentiment。
       - candidate_thread_ids：10個候選ID。
       - top_thread_ids：最終選定ID。
    5. 若無關LIHKG，返回空category_ids。

    輸出：
    { "theme": "", "category_ids": [], "data_type": "", "post_limit": 0, "reply_limit": 0, "filters": {}, "processing": "", "candidate_thread_ids": [], "top_thread_ids": [], "category_suggestion": "" }
    """

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {
            "theme": "未知", "category_ids": [cat_id], "data_type": "both", "post_limit": 2,
            "reply_limit": 75, "filters": {"min_replies": 20, "min_likes": 10},
            "processing": "summarize", "candidate_thread_ids": [], "top_thread_ids": [],
            "category_suggestion": "Missing API key"
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
                result["category_ids"] = [cat_id]
                logger.info(f"Analysis result: theme={result['theme']}, top_thread_ids={result['top_thread_ids']}")
                return result
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {
            "theme": "未知", "category_ids": [cat_id], "data_type": "both", "post_limit": 2,
            "reply_limit": 75, "filters": {"min_replies": 20, "min_likes": 10},
            "processing": "summarize", "candidate_thread_ids": [], "top_thread_ids": [],
            "category_suggestion": f"Analysis failed: {str(e)}"
        }

async def stream_grok3_response(user_query, metadata, thread_data, processing):
    """
    使用 Grok 3 API 生成流式回應。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        yield "錯誤: 缺少 API 密鑰"
        return

    filtered_thread_data = {
        tid: {
            "thread_id": data["thread_id"], "title": data["title"], "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0), "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": [r for r in data.get("replies", []) if r.get("like_count", 0) != 0 or r.get("dislike_count", 0) != 0][:25],
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items()
    }

    prompt_templates = {
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
        "summarize": f"""
        總結 LIHKG 帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        引用高關注回覆，適配分類語氣。
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

    prompt = prompt_templates.get(processing, f"直接回答問題，50-100字。問題：{user_query}\n輸出：回應")
    if len(prompt) > GROK3_TOKEN_LIMIT:
        for tid in filtered_thread_data:
            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:10]
        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
        logger.info(f"Truncated prompt: {len(prompt)} characters")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": True
    }

    response_chunks = []
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=30) as response:
                    async for line in response.content:
                        if line and not line.isspace():
                            line_str = line.decode('utf-8').strip()
                            if line_str == "data: [DONE]":
                                break
                            if line_str.startswith("data: "):
                                try:
                                    chunk = json.loads(line_str[6:])
                                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content and content not in response_chunks:
                                        response_chunks.append(content)
                                        yield content
                                except json.JSONDecodeError:
                                    continue
                    logger.info("Stream response completed")
                    return
        except Exception as e:
            logger.warning(f"Grok 3 request failed, attempt {attempt+1}: {str(e)}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            yield f"錯誤: 連線失敗，請稍後重試"
            return

async def process_user_question(user_question, selected_cat, cat_id, analysis, rate_limit_info):
    """
    處理用戶問題，抓取並分析 LIHKG 帖子。
    """
    post_limit = min(analysis.get("post_limit", 2), 10)
    reply_limit = min(analysis.get("reply_limit", 75), 75)
    filters = analysis.get("filters", {})
    min_replies = filters.get("min_replies", 20)
    min_likes = filters.get("min_likes", 10)
    top_thread_ids = analysis.get("top_thread_ids", [])

    thread_data = []
    rate_limit_info = rate_limit_info.copy()

    # 抓取帖子列表
    initial_threads = []
    for page in range(1, 4):
        result = await get_lihkg_topic_list(
            cat_id=cat_id, start_page=page, max_pages=1,
            request_counter=rate_limit_info["counter"], last_reset=rate_limit_info["last_reset"], rate_limit_info=rate_limit_info["until"]
        )
        rate_limit_info["counter"] = result.get("request_counter", rate_limit_info["counter"])
        rate_limit_info["last_reset"] = result.get("last_reset", rate_limit_info["last_reset"])
        rate_limit_info["until"] = result.get("rate_limit_until", rate_limit_info["until"])
        initial_threads.extend(result.get("items", []))
        logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(result.get('items', []))}")
        if len(initial_threads) >= 90:
            initial_threads = initial_threads[:90]
            break

    filtered_items = [
        item for item in initial_threads
        if item.get("no_of_reply", 0) >= min_replies and int(item.get("like_count", 0)) >= min_likes
    ]
    logger.info(f"Filtered items: {len(filtered_items)} from {len(initial_threads)}")

    # 更新緩存
    for item in initial_threads:
        thread_id = str(item["thread_id"])
        if thread_id not in st.session_state.thread_cache or time.time() - st.session_state.thread_cache[thread_id]["timestamp"] > 3600:
            st.session_state.thread_cache[thread_id] = {
                "data": {
                    "thread_id": thread_id, "title": item["title"], "no_of_reply": item.get("no_of_reply", 0),
                    "last_reply_time": item.get("last_reply_time", 0), "like_count": item.get("like_count", 0),
                    "dislike_count": item.get("dislike_count", 0), "replies": [], "fetched_pages": []
                },
                "timestamp": time.time()
            }

    # 選擇帖子
    candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
    if not candidate_threads and filtered_items:
        candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items)))
        logger.info(f"No top_thread_ids, randomly selected: {[item['thread_id'] for item in candidate_threads]}")

    # 抓取帖子內容
    for item in candidate_threads:
        thread_id = str(item["thread_id"])
        cached_data = st.session_state.thread_cache.get(thread_id, {}).get("data", {})
        if cached_data.get("replies") and len(cached_data["replies"]) >= reply_limit:
            thread_data.append(cached_data)
            logger.info(f"Cache hit for thread {thread_id}: replies={len(cached_data['replies'])}")
            continue

        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id, cat_id=cat_id,
            request_counter=rate_limit_info["counter"], last_reset=rate_limit_info["last_reset"], rate_limit_until=rate_limit_info["until"],
            max_replies=reply_limit, fetch_last_pages=2
        )
        rate_limit_info["counter"] = thread_result.get("request_counter", rate_limit_info["counter"])
        rate_limit_info["last_reset"] = thread_result.get("last_reset", rate_limit_info["last_reset"])
        rate_limit_info["until"] = thread_result.get("rate_limit_until", rate_limit_info["until"])

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
        logger.info(f"Fetched thread {thread_id}: replies={len(replies)}")
        await asyncio.sleep(1)

    return {
        "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info
    }
