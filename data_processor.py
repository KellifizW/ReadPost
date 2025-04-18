import aiohttp
import random
import asyncio
import time
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from utils import clean_html
import streamlit.logger
from datetime import datetime

logger = streamlit.logger.get_logger(__name__)

async def process_user_question(user_question, cat_id_map, selected_cat, analysis, request_counter, last_reset, rate_limit_until):
    """處理用戶問題，實現分階段篩選和回覆抓取"""
    # 提取分析參數
    category_ids = analysis.get("category_ids", [cat_id_map[selected_cat]])
    data_type = analysis.get("data_type", "both")
    post_limit = min(analysis.get("post_limit", 2), 10)  # 最大 10 個帖子
    reply_limit = min(analysis.get("reply_limit", 150), 150)  # 最大 150 條回覆
    filters = analysis.get("filters", {})
    candidate_thread_ids = analysis.get("candidate_thread_ids", [])
    top_thread_ids = analysis.get("top_thread_ids", [])
    
    min_replies = filters.get("min_replies", 50)
    min_likes = filters.get("min_likes", 10)
    dislike_count_max = filters.get("dislike_count_max", 20)
    recent_only = filters.get("recent_only", True)
    exclude_thread_ids = filters.get("exclude_thread_ids", [])
    
    # 預設分類
    cat_id = cat_id_map[selected_cat]
    for cat_name, cat_id_val in cat_id_map.items():
        if cat_name in user_question:
            selected_cat = cat_name
            cat_id = cat_id_val
            break
    if cat_id not in category_ids:
        category_ids = [cat_id] + [c for c in category_ids if c != cat_id]
    
    thread_data = []
    rate_limit_info = []
    
    # 計算今日時間範圍
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today.timestamp())
    
    # 階段 1：抓取 30-90 個標題
    initial_threads = []
    for cat_id in category_ids[:2]:  # 限制最多 2 個分類
        result = await get_lihkg_topic_list(
            cat_id=cat_id,
            sub_cat_id=0,
            start_page=1,
            max_pages=3,  # 2-3 頁，約 30-90 個標題
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
        logger.info(f"初始抓取: cat_id={cat_id}, 帖子數={len(items)}")
    
    # 階段 2：篩選候選帖子（10 個）
    filtered_items = [
        item for item in initial_threads
        if item.get("no_of_reply", 0) >= min_replies and
           int(item.get("like_count", 0)) >= min_likes and
           int(item.get("dislike_count", 0)) <= dislike_count_max and
           (not recent_only or int(item.get("last_reply_time", 0)) >= today_timestamp) and
           str(item["thread_id"]) not in exclude_thread_ids
    ]
    candidate_threads = [
        item for item in filtered_items
        if str(item["thread_id"]) in candidate_thread_ids or not candidate_thread_ids
    ][:10]
    logger.info(f"篩選候選帖子: 總數={len(initial_threads)}, 符合條件={len(filtered_items)}, 候選數={len(candidate_threads)}")
    
    # 階段 3：抓取候選帖子的首頁回覆（25 條）
    candidate_data = {}
    for item in candidate_threads:
        thread_id = str(item["thread_id"])
        logger.info(f"開始抓取候選帖子首頁回覆: thread_id={thread_id}")
        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id,
            cat_id=cat_id,
            request_counter=request_counter,
            last_reset=last_reset,
            rate_limit_until=rate_limit_until,
            max_replies=25,  # 首頁回覆
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
        if str(item["thread_id"]) in top_thread_ids or not top_thread_ids
    ][:post_limit]
    
    for item in final_threads:
        thread_id = str(item["thread_id"])
        logger.info(f"開始抓取最終帖子回覆: thread_id={thread_id}")
        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id,
            cat_id=cat_id,
            request_counter=request_counter,
            last_reset=last_reset,
            rate_limit_until=rate_limit_until,
            max_replies=reply_limit,  # 最多 150 條
            fetch_last_pages=3  # 末 3 頁
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
        
        await asyncio.sleep(5)
    
    return {
        "selected_cat": selected_cat,
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }