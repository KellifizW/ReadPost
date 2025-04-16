from lihkg_api import get_lihkg_topic_list
import re

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
    
    # 抓取帖子
    result = await get_lihkg_topic_list(
        cat_id=cat_id,
        sub_cat_id=0,  # 無需子分類
        start_page=1,
        max_pages=max_pages,
        request_counter=request_counter,
        last_reset=last_reset,
        rate_limit_until=rate_limit_until
    )
    
    result["selected_cat"] = selected_cat
    result["max_pages"] = max_pages
    return result
