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

async def process_user_question(user_question, cat_id_map, selected_cat, analysis, request_counter, last_reset, rate_limit_until):
    """處理用戶問題，實現分階段篩選和回覆抓取"""
    # 提取分析參數
    category_ids = [cat_id_map[selected_cat]]  # 僅使用用戶選擇的分類
    data_type = analysis.get("data_type", "both")
    post_limit = min(analysis.get("post_limit", 2), 10)  # 最大 10 個帖子
    reply_limit = min(analysis.get("reply_limit", 150), 150)  # 最大 150 條回覆
    filters = analysis.get("filters", {})
    candidate_thread_ids = analysis.get("candidate_thread_ids", [])
    top_thread_ids = analysis.get("top_thread_ids", [])
    
    # 放寬過濾條件
    min_replies = filters.get("min_replies", 10)
    min_likes = filters.get("min_likes", 5)
    dislike_count_max = filters.get("dislike_count_max", 50)
    recent_only = filters.get("recent_only", False)
    exclude_thread_ids = filters.get("exclude_thread_ids", [])
    
    # 預設分類
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
    
    # 計算今日時間範圍
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today.timestamp())
    
    # 階段 1：抓取 30-90 個標題
    initial_threads = []
    result = await get_lihkg_topic_list(
        cat_id=cat_id,
        sub_cat_id=0,
        start_page=1,
        max_pages=3,
        request_counter=request_counter,
        last_reset=last_reset,
        rate_limit_until=rate_limit_until
    )
    
    request_counter = result.get("request_counter", request_counter)
    last_reset = result.get("last_reset", last_reset)
    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
    rate_limit_info.extend(result.get("rate_limit_info", []))
    
    initial_threads = result.get("items", [])
    logger.info(f"初始抓取: cat_id={cat_id}, 帖子數={len(initial_threads)}")
    
    # 階段 2：篩選候選帖子（本地篩選）
    filtered_items = []
    filter_debug = {"min_replies_failed": 0, "min_likes_failed": 0, "dislike_count_failed": 0, "recent_only_failed": 0, "excluded_failed": 0}
    for item in initial_threads:
        thread_id = str(item["thread_id"])
        no_of_reply = item.get("no_of_reply", 0)
        like_count = int(item.get("like_count", 0))
        dislike_count = int(item.get("dislike_count", 0))
        last_reply_time = int(item.get("last_reply_time", 0))
        
        if no_of_reply < min_replies:
            filter_debug["min_replies_failed"] += 1
            continue
        if like_count < min_likes:
            filter_debug["min_likes_failed"] += 1
            continue
        if dislike_count > dislike_count_max:
            filter_debug["dislike_count_failed"] += 1
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
                f"負評過多={filter_debug['dislike_count_failed']}, "
                f"非近期帖子={filter_debug['recent_only_failed']}, "
                f"被排除={filter_debug['excluded_failed']}")
    
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
    
    # 若標題篩選失敗，隨機選擇備用帖子
    if not top_thread_ids and filtered_items:
        top_thread_ids = [item["thread_id"] for item in random.sample(filtered_items, min(post_limit, len(filtered_items)))]
        logger.warning(f"標題篩選無結果，隨機選擇帖子: top_thread_ids={top_thread_ids}")
    
    # 階段 3：抓取候選帖子的首頁回覆（若需要）
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
                ]
            }
            logger.info(f"抓取候選回覆: thread_id={thread_id}, 回覆數={len(replies)}")
    
    # 階段 4：選取最終帖子並抓取首 3 頁和末 3 頁回覆
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
            fetch_last_pages=3
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
            ]
        })
        
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
