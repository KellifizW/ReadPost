"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
包含數據處理邏輯（進階分析、緩存管理）和輔助函數。
主要函數：
- analyze_and_screen：分析問題並生成篩選策略。
- stream_grok3_response：生成流式回應。
- process_user_question：處理用戶問題，抓取並分析帖子。
- clean_html：清理 HTML 標籤。
硬編碼參數（優化建議：移至配置文件或介面）：
- post_limit=max=20
- reply_limit=75/200
- min_replies=20/50
- min_likes=10/20
- max_tokens=600/1000
- temperature=0.7
- timeout=30
- retries=3
- sleep=0.5/1
"""

import aiohttp
import asyncio
import json
import re
import math
import time
import logging
import streamlit as st
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

logger = logging.getLogger(__name__)
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000

def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False):
    prompt = f"""
    你是一個智能助手，分析用戶問題並為 LIHKG 論壇數據生成篩選和處理策略。以繁體中文回覆，輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else '無帖子標題數據'}
    {'元數據：' + json.dumps(metadata, ensure_ascii=False) if metadata else ''}
    {'回覆數據：' + json.dumps(thread_data, ensure_ascii=False) if thread_data else ''}

    任務：
    1. 理解用戶問題的意圖（例如列出帖子、總結討論、分析情緒等）。
    2. 根據問題語義，生成篩選策略：
       - 若要求列出帖子，指定篩選條件（如所有帖子、熱門帖子、含特定主題的帖子）。
       - 若要求總結或分析，選擇相關帖子並指定處理方式（如摘要、情緒分析）。
    3. 若有帖子標題數據，選擇與問題最相關的帖子 ID（最多20個），並說明篩選依據。
    4. 若無帖子標題數據，建議初始抓取（30-180個帖子）並設置寬鬆篩選條件。
    5. 若問題與分類無關，設置 category_ids 為空，並建議直接回答。

    輸出：
    {{
      {'"needs_advanced_analysis": false, "reason": "",' if is_advanced else ''}
      "strategy": {{
        "intent": "描述問題意圖（如 list_threads, summarize, analyze_sentiment）",
        "filters": {{
          "min_replies": 0,
          "min_likes": 0,
          "keywords": [],
          "sub_theme": ""
        }},
        "post_limit": 0,
        "reply_limit": 0,
        "processing": "描述處理方式（如 list, summarize, sentiment）",
        "top_thread_ids": []
      }},
      "category_ids": [],
      "category_suggestion": ""
    }}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {
            "strategy": {
                "intent": "summarize",
                "filters": {"min_replies": 20, "min_likes": 10, "keywords": [], "sub_theme": ""},
                "post_limit": 5,
                "reply_limit": 75,
                "processing": "summarize",
                "top_thread_ids": []
            },
            "category_ids": [cat_id],
            "category_suggestion": "Missing API key"
        } if not is_advanced else {
            "needs_advanced_analysis": False,
            "reason": "Missing API key",
            "strategy": {"intent": "summarize", "filters": {}, "post_limit": 0, "reply_limit": 0, "processing": "", "top_thread_ids": []},
            "category_ids": [cat_id],
            "category_suggestion": ""
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.7
    }
    
    for attempt in range(3):
        try:
            logger.info(f"Grok 3 API request: url={GROK3_API_URL}, prompt_length={len(prompt)}, payload={json.dumps(payload, ensure_ascii=False)}")
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=30) as response:
                    data = await response.json()
                    logger.info(f"Grok 3 API response: status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                    result = json.loads(data["choices"][0]["message"]["content"])
                    result["category_ids"] = [cat_id] if not is_advanced else result.get("category_ids", [cat_id])
                    logger.info(f"Analysis result: {json.dumps(result, ensure_ascii=False)}")
                    return result
        except Exception as e:
            logger.error(f"Grok 3 API failed: attempt={attempt+1}, error={str(e)}, prompt_summary={prompt[:50]}...")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "strategy": {
                    "intent": "summarize",
                    "filters": {"min_replies": 20, "min_likes": 10, "keywords": [], "sub_theme": ""},
                    "post_limit": 5,
                    "reply_limit": 75,
                    "processing": "summarize",
                    "top_thread_ids": []
                },
                "category_ids": [cat_id],
                "category_suggestion": f"Analysis failed: {str(e)}"
            } if not is_advanced else {
                "needs_advanced_analysis": False,
                "reason": f"Analysis failed: {str(e)}",
                "strategy": {"intent": "summarize", "filters": {}, "post_limit": 0, "reply_limit": 0, "processing": "", "top_thread_ids": []},
                "category_ids": [cat_id],
                "category_suggestion": ""
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, strategy=None):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
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
    
    needs_advanced_analysis = False
    reason = ""
    for data in filtered_thread_data.values():
        total_pages = (data["no_of_reply"] + 24) // 25
        target_pages = math.ceil(total_pages * 0.6)
        if len(data["fetched_pages"]) < target_pages:
            needs_advanced_analysis = True
            reason += f"帖子 {data['thread_id']} 僅抓取 {len(data['fetched_pages'])}/{total_pages} 頁，未達60%。"
    
    strategy = strategy or {"intent": "summarize", "filters": {}, "processing": "summarize"}
    prompt_templates = {
        "list": f"""
        列出 LIHKG 帖子。問題：{user_query}
        篩選策略：{json.dumps(strategy, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        步驟：
        1. 根據篩選策略（如熱門程度、關鍵詞），整理帖子清單，格式為「帖子 ID: {thread_id}\n標題: {title}\n」。
        2. 若無帖子數據，說明「未找到任何帖子」。
        輸出：帖子清單
        """,
        "summarize": f"""
        總結 LIHKG 帖子，300-500字。問題：{user_query}
        篩選策略：{json.dumps(strategy, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        步驟：
        1. 根據篩選策略，總結相關帖子內容，引用高關注回覆，適配分類語氣。
        2. 若無相關帖子，直接回答問題，基於一般知識。
        3. 若數據不足，建議用戶提供更具體問題。
        輸出：總結\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "sentiment": f"""
        分析 LIHKG 帖子情緒。問題：{user_query}
        篩選策略：{json.dumps(strategy, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        步驟：
        1. 根據篩選策略，判斷情緒分佈（正面、負面、中立），聚焦高關注回覆。
        2. 若無相關帖子，直接回答問題，基於一般知識。
        3. 若數據不足，建議用戶提供更具體問題。
        輸出：情緒分析：正面XX%，負面XX%，中立XX%\n依據：...\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "direct_answer": f"""
        直接回答用戶問題，50-200字。問題：{user_query}
        篩選策略：{json.dumps(strategy, ensure_ascii=False)}
        步驟：
        1. 基於問題和篩選策略，提供簡潔、專業的回答。
        2. 若需要外部數據，說明「當前數據不足，建議查閱最新資訊」。
        輸出：回應
        """
    }
    
    prompt = prompt_templates.get(processing, prompt_templates["direct_answer"])
    if len(prompt) > GROK3_TOKEN_LIMIT:
        for tid in filtered_thread_data:
            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:10]
        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
        logger.info(f"Truncated prompt: length={len(prompt)}")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": True
    }
    
    for attempt in range(3):
        try:
            logger.info(f"Grok 3 API request: url={GROK3_API_URL}, attempt={attempt+1}, prompt_length={len(prompt)}, payload={json.dumps(payload, ensure_ascii=False)}")
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
                                    if content and "進階分析建議：" not in content:
                                        yield content
                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON decode error: error={str(e)}, line={line_str}")
                                    continue
                    logger.info(f"Grok 3 API response: status={response.status}, stream_completed=True")
                    return
        except Exception as e:
            logger.error(f"Grok 3 API failed: attempt={attempt+1}, error={str(e)}, prompt_summary={prompt[:50]}...")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            yield f"錯誤: 連線失敗，請稍後重試"
            return

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None):
    strategy = analysis.get("strategy", {
        "intent": "summarize",
        "filters": {"min_replies": 20, "min_likes": 10, "keywords": [], "sub_theme": ""},
        "post_limit": 5,
        "reply_limit": 75,
        "processing": "summarize",
        "top_thread_ids": []
    })
    
    post_limit = min(strategy.get("post_limit", 5), 20)
    reply_limit = 200 if is_advanced else min(strategy.get("reply_limit", 75), 75)
    filters = strategy.get("filters", {})
    min_replies = filters.get("min_replies", 20)
    min_likes = filters.get("min_likes", 10)
    keywords = filters.get("keywords", [])
    sub_theme = filters.get("sub_theme", "")
    top_thread_ids = list(set(strategy.get("top_thread_ids", [])))
    
    thread_data = []
    rate_limit_info = []
    
    # 進階分析：處理已有帖子數據
    if is_advanced and previous_thread_ids:
        for thread_id in previous_thread_ids:
            cached_data = previous_thread_data.get(thread_id) if previous_thread_data else None
            if not cached_data:
                continue
            fetched_pages = cached_data.get("fetched_pages", [])
            existing_replies = cached_data.get("replies", [])
            total_replies = cached_data.get("no_of_reply", 0)
            
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
            
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id, cat_id=cat_id, request_counter=request_counter, last_reset=last_reset,
                rate_limit_until=rate_limit_until, max_replies=reply_limit, fetch_last_pages=remaining_pages
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
                "thread_id": str(thread_id), "title": thread_result.get("title", cached_data.get("title", "未知標題")),
                "no_of_reply": thread_result.get("total_replies", total_replies), "last_reply_time": thread_result.get("last_reply_time", cached_data.get("last_reply_time", 0)),
                "like_count": thread_result.get("like_count", cached_data.get("like_count", 0)),
                "dislike_count": thread_result.get("dislike_count", cached_data.get("dislike_count", 0)),
                "replies": sorted_replies, "fetched_pages": all_fetched_pages
            })
            logger.info(f"Advanced thread {thread_id}: replies={len(sorted_replies)}, pages={len(all_fetched_pages)}/{target_pages}")
            await asyncio.sleep(0.5)
        
        return {
            "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    # 初始抓取帖子
    initial_threads = []
    max_pages = 6
    for page in range(1, max_pages + 1):
        result = await get_lihkg_topic_list(cat_id=cat_id, start_page=page, max_pages=1, request_counter=request_counter, last_reset=last_reset, rate_limit_until=rate_limit_until)
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(result.get("rate_limit_info", []))
        initial_threads.extend(result.get("items", []))
        logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(result.get('items', []))}")
        if len(initial_threads) >= 180:
            initial_threads = initial_threads[:180]
            break
    
    # 篩選帖子
    filtered_items = [
        item for item in initial_threads
        if item.get("no_of_reply", 0) >= min_replies and int(item.get("like_count", 0)) >= min_likes
    ]
    logger.info(f"Filtered items: {len(filtered_items)} from {len(initial_threads)}")
    
    # 緩存帖子數據
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
    
    # 若無 top_thread_ids，重新分析
    if not top_thread_ids:
        logger.info(f"No top_thread_ids, re-analyzing with thread_titles")
        analysis = await analyze_and_screen(
            user_query=user_question, cat_name=selected_cat, cat_id=cat_id,
            thread_titles=[{"thread_id": item["thread_id"], "title": item["title"]} for item in filtered_items[:180]]
        )
        strategy = analysis.get("strategy", strategy)
        top_thread_ids = list(set(strategy.get("top_thread_ids", [])))
    
    # 處理帖子
    valid_threads = filtered_items[:post_limit] if not top_thread_ids else [
        item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)
    ]
    
    if not valid_threads:
        logger.warning(f"No valid threads found for cat_id={cat_id}")
        direct_answer = ""
        async for content in stream_grok3_response(
            user_query=user_question, metadata={}, thread_data={}, processing="direct_answer", strategy=strategy
        ):
            direct_answer += content
        return {
            "selected_cat": selected_cat, "thread_data": [], "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until,
            "direct_answer": direct_answer
        }
    
    for item in valid_threads[:post_limit]:
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
        logger.info(f"Fetched thread {thread_id}: replies={len(replies)}")
        await asyncio.sleep(0.5)
    
    return {
        "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
    }
