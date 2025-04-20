"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
包含數據處理邏輯（進階分析、緩存管理）和輔助函數。
主要函數：
- analyze_and_screen：分析問題並生成篩選策略。
- stream_grok3_response：生成流式回應。
- process_user_question：處理用戶問題，抓取並分析帖子。
- clean_html：清理 HTML 標籤。
硬編碼參數：
- post_limit=max=20
- reply_limit=75/200
- min_replies=10/20
- max_tokens=300/1500
- temperature=0.5/0.9
- timeout=60
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
    max_tokens = 300 if len(user_query) < 50 else 1500
    temperature = 0.5 if "列出" in user_query or "熱門" in user_query else 0.9

    slim_thread_data = {
        tid: {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "fetched_pages": data.get("fetched_pages", []),
            "total_pages": math.ceil(data.get("no_of_reply", 0) / 25),
            "replies": [{"msg": r["msg"], "reply_time": r.get("reply_time", 0)} for r in data.get("replies", [])[:10]]
        } for tid, data in (thread_data or {}).items() if "thread_id" in data
    } if thread_data else {}
    slim_metadata = [
        {"thread_id": m["thread_id"], "title": m["title"], "no_of_reply": m.get("no_of_reply", 0)}
        for m in (metadata or [])[:20] if "thread_id" in m
    ] if metadata else []

    prompt = f"""
    你是一個智能助手，分析用戶問題並為 LIHKG 論壇數據生成篩選和處理策略。以繁體中文回覆，輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else '無帖子標題數據'}
    {'元數據：' + json.dumps(slim_metadata, ensure_ascii=False) if slim_metadata else ''}
    {'回覆數據：' + json.dumps(slim_thread_data, ensure_ascii=False) if slim_thread_data else ''}

    任務：
    1. 理解用戶問題的意圖（例如列出熱門帖子、總結討論、表達個人偏好等）。
    2. 根據問題語義，生成篩選策略：
       - 若要求「熱門帖子」，選擇回覆數高（min_replies=20）或近期活躍的帖子，關鍵詞為可選。
       - 若要求搞笑話題，優先選擇標題或回覆含幽默元素（迷因、搞笑詞彙）的帖子，關鍵詞可放寬。
       - 若要求總結或分析，選擇相關帖子並指定處理方式（如摘要、情緒分析）。
       - 篩選優先基於回覆數和活躍度，關鍵詞為輔助條件，若關鍵詞匹配失敗，則放寬至高回覆數帖子。
    3. 若有帖子標題數據，選擇與問題最相關的帖子 ID（最多10個），按回覆數降序排序，並說明篩選依據。
    4. 若無帖子標題數據，建議初始抓取（30-180個帖子）並設置寬鬆篩選條件，post_limit 不超過 10。
    5. 若問題與分類無關，設置 category_ids 為空，並建議直接回答。
    6. 若 is_advanced=True，檢查帖子是否達到 60% 總頁數（total_pages * 0.6，向上取整），若未達標，設置 needs_advanced_analysis=True。
    7. 根據問題複雜度，動態建議 post_limit（簡單問題 5，複雜問題 10）和 reply_limit（簡單問題 50，複雜問題 200）。

    輸出：
    {{
      {'"needs_advanced_analysis": false, "reason": "",' if is_advanced else ''}
      "strategy": {{
        "intent": "描述問題意圖（如 list_hot_threads, summarize, analyze_sentiment, express_preference）",
        "filters": {{
          "min_replies": 0,
          "keywords": [],
          "sub_theme": ""
        }},
        "post_limit": 0,
        "reply_limit": 0,
        "processing": "描述處理方式（如 list_with_summary, summarize, sentiment, express_preference）",
        "top_thread_ids": []
      }},
      "category_ids": [],
      "category_suggestion": ""
    }}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing. Falling back to direct answer.")
        return {
            "strategy": {
                "intent": "direct_answer",
                "filters": {"min_replies": 20, "keywords": [], "sub_theme": ""},
                "post_limit": 5,
                "reply_limit": 50,
                "processing": "direct_answer",
                "top_thread_ids": []
            },
            "category_ids": [cat_id],
            "category_suggestion": "API key missing, using direct answer."
        } if not is_advanced else {
            "needs_advanced_analysis": False,
            "reason": "Missing API key",
            "strategy": {"intent": "direct_answer", "filters": {}, "post_limit": 0, "reply_limit": 0, "processing": "direct_answer", "top_thread_ids": []},
            "category_ids": [cat_id],
            "category_suggestion": ""
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    for attempt in range(3):
        try:
            logger.info(f"Grok 3 API request: url={GROK3_API_URL}, prompt_length={len(prompt)}")
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 429:
                        wait_time = int(response.headers.get("Retry-After", 5 * (2 ** attempt)))
                        wait_time = min(wait_time, 15)
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                        continue
                    data = await response.json()
                    logger.info(f"Grok 3 API response: status={response.status}")
                    result = json.loads(data["choices"][0]["message"]["content"])
                    # 確保 filters 的 keywords 為列表
                    if "strategy" in result and "filters" in result["strategy"]:
                        if result["strategy"]["filters"].get("keywords") is None:
                            result["strategy"]["filters"]["keywords"] = []
                    result["category_ids"] = [cat_id] if not is_advanced else result.get("category_ids", [cat_id])
                    logger.debug(f"Analysis result: {json.dumps(result, ensure_ascii=False)}")
                    return result
        except Exception as e:
            logger.error(f"Grok 3 API failed: attempt={attempt+1}, error={str(e)}")
            if attempt < 2:
                await asyncio.sleep(min(5 * (2 ** attempt), 15))
                continue
            return {
                "strategy": {
                    "intent": "direct_answer",
                    "filters": {"min_replies": 20, "keywords": [], "sub_theme": ""},
                    "post_limit": 5,
                    "reply_limit": 50,
                    "processing": "direct_answer",
                    "top_thread_ids": []
                },
                "category_ids": [cat_id],
                "category_suggestion": f"Analysis failed: {str(e)}"
            } if not is_advanced else {
                "needs_advanced_analysis": False,
                "reason": f"Analysis failed: {str(e)}",
                "strategy": {"intent": "direct_answer", "filters": {}, "post_limit": 0, "reply_limit": 0, "processing": "direct_answer", "top_thread_ids": []},
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
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": [
                {
                    "msg": clean_html(r["msg"]),
                    "like_count": r.get("like_count", 0),
                    "dislike_count": r.get("dislike_count", 0),
                    "reply_time": r.get("reply_time", 0)
                } for r in data.get("replies", []) if r.get("like_count", 0) != 0 or r.get("dislike_count", 0) != 0
            ][:25],
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items() if "thread_id" in data
    }
    
    needs_advanced_analysis = False
    reason = ""
    for tid, data in filtered_thread_data.items():
        total_pages = (data["no_of_reply"] + 24) // 25
        target_pages = math.ceil(total_pages * 0.6)
        if len(data["fetched_pages"]) < target_pages:
            needs_advanced_analysis = True
            reason += f"帖子 {tid} 僅抓取 {len(data['fetched_pages'])}/{target_pages} 頁，未達60%。 "
    
    prompt_templates = {
        "list_with_summary": f"""
        列出並總結 LIHKG 帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        按回覆數排序，聚焦高關注帖子，適配分類語氣（吹水台輕鬆，財經台專業）。
        輸出：列出帖子（標題、ID）並總結內容\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "summarize": f"""
        總結 LIHKG 帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦高關注回覆，適配分類語氣（吹水台輕鬆，財經台專業）。
        輸出：總結\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "sentiment": f"""
        分析 LIHKG 帖子情緒，200-300字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        判斷情緒分佈（正面、負面、中立），聚焦高關注回覆。
        輸出：情緒分析：正面XX%，負面XX%，中立XX%\n依據：...\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "express_preference": f"""
        根據用戶偏好生成回應，200-300字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦用戶表達的偏好，引用相關回覆，適配分類語氣。
        輸出：回應\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "direct_answer": f"""
        直接回答問題，100-200字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        若無相關帖子，基於問題語義回答，適配分類語氣。
        輸出：回應
        """
    }
    
    prompt = prompt_templates.get(processing, f"直接回答問題，100-200字。問題：{user_query}\n輸出：回應")
    if len(prompt) > GROK3_TOKEN_LIMIT:
        for tid in filtered_thread_data:
            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:10]
        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
        logger.info(f"Prompt truncated: length={len(prompt)}")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.9,
        "stream": True
    }
    
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 429:
                        wait_time = int(response.headers.get("Retry-After", 5 * (2 ** attempt)))
                        wait_time = min(wait_time, 15)
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds")
                        await asyncio.sleep(wait_time)
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
                                    if content and "進階分析建議：" not in content:
                                        yield content
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse chunk: {line_str}")
                                    continue
                    return
        except Exception as e:
            logger.error(f"Streaming failed: attempt={attempt+1}, error={str(e)}")
            if attempt < 2:
                await asyncio.sleep(min(5 * (2 ** attempt), 15))
                continue
            yield f"錯誤: 連線失敗，請稍後重試"
            return

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None):
    strategy = analysis.get("strategy", {
        "intent": "list_hot_threads",
        "filters": {"min_replies": 20, "keywords": [], "sub_theme": ""},
        "post_limit": 10,
        "reply_limit": 50,
        "processing": "list_with_summary",
        "top_thread_ids": []
    })
    
    post_limit = min(strategy.get("post_limit", 10), 20)
    reply_limit = 200 if is_advanced else min(strategy.get("reply_limit", 50), 50)
    filters = strategy.get("filters", {})
    min_replies = filters.get("min_replies", 20)
    keywords = filters.get("keywords", []) or []  # 確保 keywords 為列表
    sub_theme = filters.get("sub_theme", "")
    top_thread_ids = list(set(strategy.get("top_thread_ids", [])))
    
    thread_data = []
    rate_limit_info = []
    
    if is_advanced and previous_thread_ids:
        for thread_id in previous_thread_ids:
            cached_data = previous_thread_data.get(thread_id) if previous_thread_data else None
            if not cached_data or "thread_id" not in cached_data:
                logger.warning(f"Invalid cached data for thread_id={thread_id}")
                continue
            fetched_pages = cached_data.get("fetched_pages", [])
            existing_replies = cached_data.get("replies", [])
            total_replies = cached_data.get("no_of_reply", 0)
            
            total_pages = (total_replies + 24) // 25
            target_pages = math.ceil(total_pages * 0.6)
            remaining_pages = max(0, target_pages - len(fetched_pages))
            
            if remaining_pages <= 0:
                logger.info(f"Thread fetch complete: thread_id={thread_id}, pages={len(fetched_pages)}/{target_pages}, replies={len(existing_replies)}")
                thread_data.append({
                    "thread_id": str(thread_id), 
                    "title": cached_data.get("title", "未知標題"),
                    "no_of_reply": total_replies, 
                    "last_reply_time": cached_data.get("last_reply_time", 0),
                    "like_count": cached_data.get("like_count", 0),
                    "dislike_count": cached_data.get("dislike_count", 0),
                    "replies": existing_replies[:50], 
                    "fetched_pages": fetched_pages
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
                logger.warning(f"Failed to fetch thread content: thread_id={thread_id}")
                continue
            
            all_replies = existing_replies + [
                {
                    "msg": clean_html(r["msg"]),
                    "like_count": r.get("like_count", 0),
                    "dislike_count": r.get("dislike_count", 0),
                    "reply_time": r.get("reply_time", 0)
                } for r in replies
            ]
            sorted_replies = sorted(all_replies, key=lambda x: x.get("reply_time", 0), reverse=True)[:reply_limit]
            all_fetched_pages = sorted(set(fetched_pages + thread_result.get("fetched_pages", [])))
            
            thread_data.append({
                "thread_id": str(thread_id), 
                "title": thread_result.get("title", cached_data.get("title", "未知標題")),
                "no_of_reply": thread_result.get("total_replies", total_replies), 
                "last_reply_time": thread_result.get("last_reply_time", cached_data.get("last_reply_time", 0)),
                "like_count": thread_result.get("like_count", cached_data.get("like_count", 0)),
                "dislike_count": thread_result.get("dislike_count", cached_data.get("dislike_count", 0)),
                "replies": sorted_replies[:50], 
                "fetched_pages": all_fetched_pages
            })
            logger.info(f"Advanced thread fetch: thread_id={thread_id}, pages={len(all_fetched_pages)}/{target_pages}, replies={len(sorted_replies)}")
            await asyncio.sleep(0.5)
        
        return {
            "selected_cat": selected_cat, 
            "thread_data": thread_data, 
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, 
            "last_reset": last_reset, 
            "rate_limit_until": rate_limit_until
        }
    
    initial_threads = []
    max_pages = 6
    for page in range(1, max_pages + 1):
        result = await get_lihkg_topic_list(cat_id=cat_id, start_page=page, max_pages=1, request_counter=request_counter, last_reset=last_reset, rate_limit_until=rate_limit_until)
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(result.get("rate_limit_info", []))
        items = result.get("items", []) or []
        initial_threads.extend([item for item in items if "thread_id" in item])
        logger.info(f"Fetched page: cat_id={cat_id}, page={page}, items={len(items)}")
        if not items:
            logger.warning(f"No items fetched for cat_id={cat_id}, page={page}")
            break
        if len(initial_threads) >= 180:
            initial_threads = initial_threads[:180]
            break
    
    filtered_items = [
        item for item in initial_threads
        if item.get("no_of_reply", 0) >= min_replies and
           item.get("title") is not None and isinstance(item.get("title"), str) and
           (
               not keywords or
               any(keyword.lower() in item["title"].lower() for keyword in keywords) or
               re.search(r'\(\d+\)|搞笑|哈哈|笑死|好笑|迷因', item["title"], re.IGNORECASE)
           ) and "thread_id" in item
    ] if initial_threads else []
    filtered_items.sort(key=lambda x: (x.get("no_of_reply", 0), x.get("last_reply_time", 0)), reverse=True)
    logger.info(f"Filtered threads: cat_id={cat_id}, initial_count={len(initial_threads)}, filtered_count={len(filtered_items)}")
    logger.debug(f"Filtered thread IDs: {[item['thread_id'] for item in filtered_items]}")
    
    # 記錄無效的 item
    for item in initial_threads:
        if item.get("title") is None or not isinstance(item.get("title"), str):
            logger.warning(f"Invalid thread title: thread_id={item.get('thread_id', 'unknown')}, title={item.get('title')}")
    
    for item in initial_threads:
        if "thread_id" not in item:
            logger.warning(f"Missing thread_id in item: cat_id={cat_id}")
            continue
        thread_id = str(item["thread_id"])
        st.session_state.thread_cache[thread_id] = {
            "data": {
                "thread_id": thread_id, 
                "title": item.get("title", "未知標題"), 
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0), 
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [], 
                "fetched_pages": []
            },
            "timestamp": time.time()
        }
    
    if not filtered_items:
        logger.warning(f"No threads passed initial filter for cat_id={cat_id}. Falling back to top reply count threads.")
        filtered_items = sorted(
            [item for item in initial_threads if item.get("no_of_reply", 0) >= min_replies and "thread_id" in item and item.get("title") is not None and isinstance(item.get("title"), str)],
            key=lambda x: (x.get("no_of_reply", 0), x.get("last_reply_time", 0)),
            reverse=True
        )[:post_limit] if initial_threads else []
        logger.info(f"Fallback filtered threads: cat_id={cat_id}, fallback_count={len(filtered_items)}")
    
    if not top_thread_ids and filtered_items:
        logger.info(f"Re-analyzing with thread titles: cat_id={cat_id}, thread_count={len(filtered_items)}")
        analysis = await analyze_and_screen(
            user_query=user_question, 
            cat_name=selected_cat, 
            cat_id=cat_id,
            thread_titles=[{"thread_id": item["thread_id"], "title": item["title"]} for item in filtered_items[:180] if "thread_id" in item]
        )
        strategy = analysis.get("strategy", strategy)
        top_thread_ids = list(set(strategy.get("top_thread_ids", [])))
    
    valid_threads = [
        item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)
    ] if top_thread_ids else filtered_items[:post_limit]
    
    if not valid_threads:
        logger.warning(f"No valid threads after re-analysis for cat_id={cat_id}. Using top reply count threads.")
        valid_threads = filtered_items[:post_limit] if filtered_items else []
    
    if not valid_threads:
        logger.warning(f"No threads available for cat_id={cat_id}. Returning direct answer with metadata.")
        metadata = [
            {"thread_id": item["thread_id"], "title": item["title"], "no_of_reply": item.get("no_of_reply", 0)}
            for item in sorted(initial_threads, key=lambda x: x.get("no_of_reply", 0), reverse=True)[:5]
            if "thread_id" in item and item.get("title") is not None and isinstance(item.get("title"), str)
        ] if initial_threads else []
        direct_answer = ""
        async for content in stream_grok3_response(
            user_query=user_question, 
            metadata=metadata, 
            thread_data={}, 
            processing="direct_answer", 
            strategy=strategy
        ):
            direct_answer += content
        return {
            "selected_cat": selected_cat, 
            "thread_data": [], 
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, 
            "last_reset": last_reset, 
            "rate_limit_until": rate_limit_until,
            "direct_answer": direct_answer
        }
    
    for item in valid_threads[:post_limit]:
        if "thread_id" not in item:
            logger.warning(f"Missing thread_id in valid thread: cat_id={cat_id}")
            continue
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
        
        replies = thread_result.get("replies", []) or []
        if not replies and thread_result.get("total_replies", 0) >= min_replies:
            logger.warning(f"Failed to fetch thread content: thread_id={thread_id}")
            continue
        
        sorted_replies = sorted(replies, key=lambda x: x.get("reply_time", 0), reverse=True)[:reply_limit]
        thread_data.append({
            "thread_id": thread_id, 
            "title": item.get("title", "未知標題"), 
            "no_of_reply": item.get("no_of_reply", 0),
            "last_reply_time": item.get("last_reply_time", 0),
            "like_count": item.get("like_count", 0),
            "dislike_count": item.get("dislike_count", 0),
            "replies": [
                {
                    "msg": clean_html(r["msg"]),
                    "like_count": r.get("like_count", 0),
                    "dislike_count": r.get("dislike_count", 0),
                    "reply_time": r.get("reply_time", 0)
                } for r in sorted_replies
            ][:50],
            "fetched_pages": thread_result.get("fetched_pages", [1])
        })
        st.session_state.thread_cache[thread_id]["data"].update({
            "replies": thread_data[-1]["replies"], 
            "fetched_pages": thread_data[-1]["fetched_pages"],
            "like_count": thread_data[-1]["like_count"],
            "dislike_count": thread_data[-1]["dislike_count"]
        })
        st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
        logger.info(f"Fetched thread: thread_id={thread_id}, replies={len(replies)}, pages={len(thread_result.get('fetched_pages', []))}")
        await asyncio.sleep(0.5)
    
    return {
        "selected_cat": selected_cat, 
        "thread_data": thread_data, 
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter, 
        "last_reset": last_reset, 
        "rate_limit_until": rate_limit_until
    }
