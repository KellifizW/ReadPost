from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
import re
from utils import clean_html

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
    
    # 抓取回覆數據（僅限回覆數 ≥ 125 的帖子）
    items = result.get("items", [])
    filtered_items = [item for item in items if item.get("no_of_reply", 0) >= 125]
    max_replies_per_thread = 3  # 每帖最多抓取 3 條回覆
    thread_data = []
    
    for item in filtered_items:
        thread_id = item["thread_id"]
        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id,
            cat_id=cat_id,
            request_counter=result["request_counter"],
            last_reset=result["last_reset"],
            rate_limit_until=result["rate_limit_until"]
        )
        replies = thread_result.get("replies", [])
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
        result["request_counter"] = thread_result["request_counter"]
        result["last_reset"] = thread_result["last_reset"]
        result["rate_limit_until"] = thread_result["rate_limit_until"]
    
    result["selected_cat"] = selected_cat
    result["max_pages"] = max_pages
    result["thread_data"] = thread_data
    return result
