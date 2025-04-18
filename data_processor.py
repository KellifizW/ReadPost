import aiohttp
import random
import asyncio
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from utils import clean_html
import streamlit.logger
from datetime import datetime

logger = streamlit.logger.get_logger(__name__)

async def process_user_question(question, cat_id_map, selected_cat, analysis, request_counter, last_reset, rate_limit_until):
    # 提取分析參數
    category_ids = analysis.get("category_ids", [cat_id_map[selected_cat]])
    data_type = analysis.get("data_type", "both")
    post_limit = min(analysis.get("post_limit", 5), 20)
    reply_limit = min(analysis.get("reply_limit", 100), 200)
    filters = analysis.get("filters", {})
    
    min_replies = filters.get("min_replies", 0)
    min_likes = filters.get("min_likes", 0)
    recent_only = filters.get("recent_only", True)
    
    # 預設分類
    cat_id = cat_id_map[selected_cat]
    for cat_name, cat_id_val in cat_id_map.items():
        if cat_name in question:
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
    
    for cat_id in category_ids[:2]:  # 限制最多 2 個分類
        result = await get_lihkg_topic_list(
            cat_id=cat_id,
            sub_cat_id=0,
            start_page=1,
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
        filtered_items = [
            item for item in items
            if item.get("no_of_reply", 0) >= min_replies and
               int(item.get("like_count", 0)) >= min_likes and
               (not recent_only or int(item.get("last_reply_time", 0)) >= today_timestamp)
        ]
        logger.info(f"篩選帖子: cat_id={cat_id}, 總數={len(items)}, 符合條件={len(filtered_items)}")
        
        thread_ids = [item["thread_id"] for item in filtered_items][:post_limit]
        for thread_id in thread_ids:
            item = next((x for x in filtered_items if x["thread_id"] == thread_id), None)
            if not item:
                continue
            
            logger.info(f"開始抓取帖子內容: thread_id={thread_id}, cat_id={cat_id}")
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit if data_type in ["replies", "both"] else 0
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
                "no_of_reply": item["no_of_reply"],
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
