import aiohttp
import random
import asyncio
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from utils import clean_html
import streamlit.logger

logger = streamlit.logger.get_logger(__name__)

async def batch_verify_threads(thread_ids, cat_id):
    """使用搜尋 API 批量驗證帖子是否可公開訪問"""
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
        ]),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"https://lihkg.com/category/{cat_id}",
    }
    
    valid_thread_ids = []
    for thread_id in thread_ids:
        url = f"https://lihkg.com/api_v2/thread/search?q={thread_id}&page=1&count=1&sort=score&type=thread"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"搜尋帖子失敗: thread_id={thread_id}, 狀態碼={response.status}")
                        continue
                    data = await response.json()
                    if not data.get("success"):
                        logger.warning(f"搜尋帖子 API 返回失敗: thread_id={thread_id}, 錯誤={data.get('error_message', '未知錯誤')}")
                        continue
                    items = data["response"].get("items", [])
                    if items and items[0]["thread_id"] == thread_id:
                        valid_thread_ids.append(thread_id)
                    else:
                        logger.warning(f"帖子無效: thread_id={thread_id}")
            except Exception as e:
                logger.error(f"搜尋帖子錯誤: thread_id={thread_id}, 錯誤={str(e)}")
                continue
        await asyncio.sleep(2)  # 每次搜尋請求間隔 2 秒
    logger.info(f"驗證帖子: 總數={len(thread_ids)}, 有效={len(valid_thread_ids)}")
    return valid_thread_ids

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
    
    # 批量驗證帖子可訪問性
    thread_ids = [item["thread_id"] for item in filtered_items]
    valid_thread_ids = await batch_verify_threads(thread_ids, cat_id)
    
    max_replies_per_thread = 3  # 每帖最多抓取 3 條回覆
    thread_data = []
    batch_size = 3  # 每次處理 3 個帖子
    
    # 分批抓取帖子內容
    for i in range(0, len(valid_thread_ids), batch_size):
        batch_thread_ids = valid_thread_ids[i:i + batch_size]
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
            
            # 每次內容請求後延遲 10 秒
            await asyncio.sleep(10)
        
        # 每批次間延遲 10 秒
        if i + batch_size < len(valid_thread_ids):
            await asyncio.sleep(10)
    
    result.update({
        "selected_cat": selected_cat,
        "max_pages": max_pages,
        "thread_data": thread_data,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    })
    return result
