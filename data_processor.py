import aiohttp
import random
import asyncio
import time
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from grok3_client import screen_thread_titles
from utils import clean_html
import streamlit.logger
from datetime import datetime

logger = streamlit.logger.get_logger(__name__)

async def process_user_question(user_question, cat_id_map, selected_cat, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None):
    """處理用戶問題，實現分階段篩選和回覆抓取"""
    category_ids = [cat_id_map[selected_cat]]
    data_type = analysis.get("data_type", "both")
    post_limit = min(analysis.get("post_limit", 2), 10)
    reply_limit = 200 if is_advanced else min(analysis.get("reply_limit", 75), 75)
    filters = analysis.get("filters", {})
    
    # 針對搞笑主題放寬篩選條件
    min_replies = 20 if analysis.get("theme") == "搞笑" else filters.get("min_replies", 50)
    min_likes = 10 if analysis.get("theme") == "搞笑" else filters.get("min_likes", 20)
    recent_only = filters.get("recent_only", False)
    exclude_thread_ids = filters.get("exclude_thread_ids", [])
    
    candidate_thread_ids = analysis.get("candidate_thread_ids", [])
    top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else (previous_thread_ids or [])
    
    cat_id = cat_id_map[selected_cat]
    for cat_name, cat_id_val in cat_id_map.items():
        if cat_name in user_question:
            selected_cat = cat_name
            cat_id = cat_id_val
            break
    if cat_id not in category_ids:
        category_ids = [cat_id]
    
    thread_data = []
    rate_limit_info = []
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today.timestamp())
    
    if is_advanced and top_thread_ids:
        logger.info(f"進階分析：抓取初始帖子剩餘回覆: top_thread_ids={top_thread_ids}")
        for thread_id in top_thread_ids:
            logger.info(f"開始抓取進階帖子回覆: thread_id={thread_id}")
            # 檢查緩存中的帖子數據
            cached_data = previous_thread_data.get(thread_id) if previous_thread_data else None
            fetched_pages = cached_data.get("fetched_pages", []) if cached_data else []
            existing_replies = cached_data.get("replies", []) if cached_data else []
            
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit,
                fetch_last_pages=2,
                start_page=max(fetched_pages, default=1) + 1 if fetched_pages else 1
            )
            
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"帖子無效或無權訪問: thread_id={thread_id}")
                continue
            
            # 合併現有回覆和新抓取的回覆
            all_replies = existing_replies + [
                {
                    "msg": clean_html(reply["msg"]),
                    "like_count": reply.get("like_count", 0),
                    "dislike_count": reply.get("dislike_count", 0),
                    "reply_time": reply.get("reply_time", 0)
                }
                for reply in replies
            ]
            sorted_replies = sorted(
                all_replies,
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:reply_limit]
            
            # 更新已抓取頁面
            new_fetched_pages = thread_result.get("fetched_pages", [])
            all_fetched_pages = list(set(fetched_pages + new_fetched_pages))
            
            thread_data.append({
                "thread_id": str(thread_id),
                "title": thread_result.get("title", cached_data.get("title", "未知標題") if cached_data else "未知標題"),
                "no_of_reply": thread_result.get("total_replies", cached_data.get("no_of_reply", 0) if cached_data else 0),
                "last_reply_time": thread_result.get("last_reply_time", cached_data.get("last_reply_time", 0) if cached_data else 0),
                "like_count": thread_result.get("like_count", cached_data.get("like_count", 0) if cached_data else 0),
                "dislike_count": thread_result.get("dislike_count", cached_data.get("dislike_count", 0) if cached_data else 0),
                "replies": sorted_replies,
                "fetched_pages": all_fetched_pages
            })
            
            logger.info(f"進階帖子數據: thread_id={thread_id}, 回覆數={len(sorted_replies)}, 已抓取頁面={all_fetched_pages}")
            await asyncio.sleep(5)
        
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    # 階段 1：抓取 30-90 個標題
    initial_threads = []
    total_fetched = 0
    for page in range(1, 4):
        result = await get_lihkg_topic_list(
            cat_id=cat_id,
            sub_cat_id=0,
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
        
        page_items = result.get("items", [])
        initial_threads.extend(page_items)
        total_fetched += len(page_items)
        logger.info(f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(page_items)}")
        
        if len(initial_threads) >= 90:
            initial_threads = initial_threads[:90]
            break
    
    # 移除合併邏輯，保留所有帖子
    logger.info(f"初始抓取: cat_id={cat_id}, 總抓取數={total_fetched}, 帖子數={len(initial_threads)}")
    
    # 階段 2：篩選候選帖子
    filtered_items = []
    filter_debug = {"min_replies_failed": 0, "min_likes_failed": 0, "recent_only_failed": 0, "excluded_failed": 0}
    for item in initial_threads:
        thread_id = str(item["thread_id"])
        no_of_reply = item.get("no_of_reply", 0)
        like_count = int(item.get("like_count", 0))
        last_reply_time = int(item.get("last_reply_time", 0))
        
        if no_of_reply < min_replies:
            filter_debug["min_replies_failed"] += 1
            continue
        if like_count < min_likes:
            filter_debug["min_likes_failed"] += 1
            continue
        if recent_only and last_reply_time < today_timestamp:
            filter_debug["recent_only_failed"] += 1
            continue
        if thread_id in exclude_thread_ids:
            filter_debug["excluded_failed"] += 1
            continue
        filtered_items.append(item)
    
    logger.info(f"本地篩選候選帖子: 總數={len(initial_threads)}, 符合條件={len(filtered_items)}, "
                f"篩選失敗詳情: 回覆數不足={filter_debug['min_replies_failed']}, "
                f"點讚數不足={filter_debug['min_likes_failed']}, "
                f"非近期帖子={filter_debug['recent_only_failed']}, "
                f"被排除={filter_debug['excluded_failed']}")
    
    # 更新緩存
    for item in initial_threads:
        thread_id = str(item["thread_id"])
        st.session_state.thread_content_cache[thread_id] = {
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
    
    # 階段 2.5：Grok 3 標題篩選
    title_screening = await screen_thread_titles(
        user_query=user_question,
        thread_titles=filtered_items[:90],
        post_limit=post_limit
    )
    top_thread_ids = title_screening.get("top_thread_ids", [])
    need_replies = title_screening.get("need_replies", True)
    screening_reason = title_screening.get("reason", "未知")
    logger.info(f"Grok 3 標題篩選: top_thread_ids={top_thread_ids}, need_replies={need_replies}, 理由={screening_reason}")
    
    if not top_thread_ids and filtered_items:
        theme = analysis.get("theme", "未知")
        keywords = {
            "感動": ["溫馨", "感人", "互助", "愛", "緣分"],
            "搞笑": ["幽默", "搞亂", "on9", "爆笑"],
            "財經": ["股票", "投資", "經濟"],
            "時事": ["新聞", "政治", "事件"]
        }.get(theme, [])
        
        theme_items = [
            item for item in filtered_items
            if any(keyword in item["title"] for keyword in keywords) or not keywords
        ]
        if theme_items:
            top_thread_ids = [item["thread_id"] for item in random.sample(theme_items, min(post_limit, len(theme_items)))]
        else:
            top_thread_ids = [item["thread_id"] for item in random.sample(filtered_items, min(post_limit, len(filtered_items)))]
        logger.warning(f"標題篩選無結果，隨機選擇帖子: top_thread_ids={top_thread_ids}, 主題={theme}")
    
    # 階段 3：抓取候選帖子的首頁回覆
    candidate_data = {}
    candidate_threads = [
        item for item in filtered_items
        if str(item["thread_id"]) in map(str, top_thread_ids)
    ][:post_limit]
    
    if not candidate_threads and top_thread_ids:
        logger.warning(f"無帖子匹配top_thread_ids: {top_thread_ids}")
        candidate_threads = filtered_items[:post_limit]
    
    if not candidate_threads:
        candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items))) if filtered_items else []
        logger.info(f"無候選帖子，隨機選擇備用: 數量={len(candidate_threads)}")
    
    if need_replies:
        for item in candidate_threads:
            thread_id = str(item["thread_id"])
            logger.info(f"開始抓取候選帖子首頁回覆: thread_id={thread_id}")
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=25,
                fetch_last_pages=0
            )
            
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"帖子無效或無權訪問: thread_id={thread_id}")
                continue
            
            sorted_replies = sorted(
                replies,
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:25]
            
            candidate_data[thread_id] = {
                "thread_id": thread_id,
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [
                    {
                        "msg": clean_html(reply["msg"]),
                        "like_count": reply.get("like_count", 0),
                        "dislike_count": reply.get("dislike_count", 0),
                        "reply_time": reply.get("reply_time", 0)
                    }
                    for reply in sorted_replies
                ],
                "fetched_pages": thread_result.get("fetched_pages", [1])
            }
            logger.info(f"抓取候選回覆: thread_id={thread_id}, 回覆數={len(replies)}")
            
            # 更新緩存
            st.session_state.thread_content_cache[thread_id]["data"].update({
                "replies": candidate_data[thread_id]["replies"],
                "fetched_pages": candidate_data[thread_id]["fetched_pages"]
            })
            st.session_state.thread_content_cache[thread_id]["timestamp"] = time.time()
    
    # 階段 4：選取最終帖子並抓取首 1 頁和末 2 頁回覆
    final_threads = [
        item for item in filtered_items
        if str(item["thread_id"]) in map(str, top_thread_ids)
    ][:post_limit]
    
    if not final_threads:
        final_threads = candidate_threads[:post_limit]
        logger.info(f"無匹配top_thread_ids，使用候選帖子: 數量={len(final_threads)}")
    
    for item in final_threads:
        thread_id = str(item["thread_id"])
        logger.info(f"開始抓取最終帖子回覆: thread_id={thread_id}")
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
        if not replies and thread_result.get("total_replies", 0) >= min_replies:
            logger.warning(f"帖子無效或無權訪問: thread_id={thread_id}")
            continue
        
        sorted_replies = sorted(
            replies,
            key=lambda x: x.get("like_count", 0),
            reverse=True
        )[:reply_limit]
        
        thread_data.append({
            "thread_id": thread_id,
            "title": item["title"],
            "no_of_reply": item.get("no_of_reply", 0),
            "last_reply_time": item.get("last_reply_time", 0),
            "like_count": item.get("like_count", 0),
            "dislike_count": item.get("dislike_count", 0),
            "replies": [
                {
                    "msg": clean_html(reply["msg"]),
                    "like_count": reply.get("like_count", 0),
                    "dislike_count": reply.get("dislike_count", 0),
                    "reply_time": reply.get("reply_time", 0)
                }
                for reply in sorted_replies
            ],
            "fetched_pages": thread_result.get("fetched_pages", [1])
        })
        
        # 更新緩存
        st.session_state.thread_content_cache[thread_id]["data"].update({
            "replies": thread_data[-1]["replies"],
            "fetched_pages": thread_data[-1]["fetched_pages"]
        })
        st.session_state.thread_content_cache[thread_id]["timestamp"] = time.time()
        
        logger.info(f"最終帖子數據: thread_id={thread_id}, 回覆數={len(replies)}")
        await asyncio.sleep(5)
    
    if not thread_data:
        logger.warning(f"最終無有效帖子: 問題={user_question}, 分類={selected_cat}, "
                      f"篩選條件={filters}, candidate_thread_ids={candidate_thread_ids}, top_thread_ids={top_thread_ids}")
    
    return {
        "selected_cat": selected_cat,
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }
