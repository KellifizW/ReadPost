import aiohttp
import random
import asyncio
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from utils import clean_html
import streamlit.logger

logger = streamlit.logger.get_logger(__name__)

async def process_user_question(question, cat_id_map, selected_cat, request_counter, last_reset, rate_limit_until):
    # 預設使用 UI 選擇的分類
    cat_id = cat_id_map[selected_cat]
    max_pages = 1
    
    # 如果問題中包含其他分類名稱，則覆蓋 UI 選擇的分類
    for cat_name, cat_id_val in cat_id_map.items():
        if cat_name in question:
            selected_cat = cat_name
            cat_id = cat_id_val
            break
    
    # 抓取帖子列表
    result = await get_lihkg_topic_list(
        cat_id=cat_id,
        sub_cat_id=0,
        start_page=1,
        max_pages=max_pages,
        request_counter=request_counter,
        last_reset=last_reset,
        rate_limit_until=rate_limit_until
    )
    
    # 更新速率限制狀態
    request_counter = result.get("request_counter", request_counter)
    last_reset = result.get("last_reset", last_reset)
    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
    
    # 篩選回覆數 ≥ 125 的帖子
    items = result.get("items", [])
    filtered_items = [item for item in items if item.get("no_of_reply", 0) >= 125]
    logger.info(f"篩選帖子: cat_id={cat_id}, 總數={len(items)}, 符合條件={len(filtered_items)}")
    
    max_replies_per_thread = 3  # 每帖最多抓取 3 條回覆
    thread_data = []
    batch_size = 3  # 每次處理 3 個帖子
    
    # 分批抓取帖子內容
    thread_ids = [item["thread_id"] for item in filtered_items]
    for i in range(0, len(thread_ids), batch_size):
        batch_thread_ids = thread_ids[i:i + batch_size]
        for thread_id in batch_thread_ids:
            item = next((x for x in filtered_items if x["thread_id"] == thread_id), None)
            if not item:
                continue
            
            logger.info(f"開始抓取帖子內容: thread_id={thread_id}, cat_id={cat_id}")
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until
            )
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= 125:
                logger.warning(f"帖子無效或無權訪問: thread_id={thread_id}")
                continue
            
            # 按正評排序，取前 max_replies_per_thread 條
            sorted_replies = sorted(
                replies,
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:max_replies_per_thread]
            
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
                        "dislike_count": reply.get("dislike_count", 0)
                    }
                    for reply in sorted_replies
                ]
            })
            
            # 更新速率限制狀態
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            
            # 每次內容請求後延遲 5 秒
            await asyncio.sleep(5)
        
        # 每批次間延遲 5 秒
        if i + batch_size < len(thread_ids):
            await asyncio.sleep(5)
    
    result.update({
        "selected_cat": selected_cat,
        "max_pages": max_pages,
        "thread_data": thread_data,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    })
    return result
