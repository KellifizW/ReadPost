import aiohttp
import asyncio
import hashlib
import time
from datetime import datetime
import pytz
import streamlit.logger

logger = streamlit.logger.get_logger(__name__)

LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

async def async_request(method, url, headers=None, json=None, retries=3):
    connector = aiohttp.TCPConnector(limit=10)
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                if method == "get":
                    async with session.get(url, headers=headers, timeout=60) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method == "post":
                    async with session.post(url, headers=headers, json=json, timeout=60) as response:
                        response.raise_for_status()
                        return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                logger.warning(f"API 請求失敗，第 {attempt+1} 次重試: {url}, 錯誤: {str(e)}")
                await asyncio.sleep(2 ** attempt)
                continue
            logger.error(f"API 請求失敗: {url}, 錯誤: {str(e)}")
            raise e

async def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=5):
    all_items = []
    tasks = []
    
    endpoint = "thread/hot" if cat_id == 2 else "thread/category"
    sub_cat_ids = [0] if cat_id == 2 else ([0, 1, 2] if cat_id == 29 else [sub_cat_id])
    max_pages = 1 if cat_id == 2 else max_pages
    
    for sub_id in sub_cat_ids:
        for p in range(start_page, start_page + max_pages):
            if cat_id == 2:
                url = f"{LIHKG_BASE_URL}{endpoint}?cat_id={cat_id}&page={p}&count=60&type=now"
            else:
                url = f"{LIHKG_BASE_URL}{endpoint}?cat_id={cat_id}&sub_cat_id={sub_id}&page={p}&count=60&type=now"
            timestamp = int(time.time())
            digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
            
            headers = {
                "X-LI-DEVICE": LIHKG_DEVICE_ID,
                "X-LI-DEVICE-TYPE": "android",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "X-LI-REQUEST-TIME": str(timestamp),
                "X-LI-DIGEST": digest,
                "Cookie": LIHKG_COOKIE,
                "orginal": "https://lihkg.com",
                "referer": f"https://lihkg.com/category/{cat_id}",
                "accept": "application/json",
            }
            
            tasks.append((p, async_request("get", url, headers=headers)))
        
        for page, task in tasks:
            try:
                response = await task
                logger.info(f"LIHKG API 請求: cat_id={cat_id}, page={page}")
                data = response
                if data.get("success") == 0:
                    logger.info(f"LIHKG API 無帖子: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}")
                    break
                items = data.get("response", {}).get("items", [])
                filtered_items = [
                    {
                        "thread_id": item["thread_id"],
                        "title": item["title"],
                        "no_of_reply": int(item.get("no_of_reply", "0")) if item.get("no_of_reply") else 0,
                        "last_reply_time": item.get("last_reply_time", ""),
                        "like_count": int(item.get("like_count", "0")) if item.get("like_count") else 0,
                        "dislike_count": int(item.get("dislike_count", "0")) if item.get("dislike_count") else 0,
                    }
                    for item in items if item.get("title")
                ]
                logger.info(f"LIHKG 抓取: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 帖子數={len(filtered_items)}")
                all_items.extend(filtered_items)
                if not items:
                    logger.info(f"LIHKG 無更多帖子: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}")
                    break
            except Exception as e:
                logger.warning(f"LIHKG API 錯誤: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 錯誤: {str(e)}")
                break
        tasks = []
    
    if not all_items:
        logger.warning(f"無有效帖子: cat_id={cat_id}")
        st.warning(f"分類 {cat_id} 無帖子，請稍後重試")
    
    logger.info(f"元數據總計: cat_id={cat_id}, 帖子數={len(all_items)}")
    return all_items

async def get_lihkg_thread_content(thread_id, cat_id=None, max_replies=175):
    replies = []
    page = 1
    per_page = 50
    
    while len(replies) < max_replies:
        url = f"{LIHKG_BASE_URL}thread/{thread_id}/page/{page}?order=reply_time"
        timestamp = int(time.time())
        digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
        
        headers = {
            "X-LI-DEVICE": LIHKG_DEVICE_ID,
            "X-LI-DEVICE-TYPE": "android",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "Cookie": LIHKG_COOKIE,
            "orginal": "https://lihkg.com",
            "referer": f"https://lihkg.com/thread/{thread_id}",
            "accept": "application/json",
        }
        
        try:
            response = await async_request("get", url, headers=headers)
            logger.info(f"LIHKG 帖子內容: thread_id={thread_id}, page={page}")
            data = response
            page_replies = data.get("response", {}).get("item_data", [])
            for reply in page_replies:
                reply["like_count"] = int(reply.get("like_count", "0")) if reply.get("like_count") else 0
                reply["dislike_count"] = int(reply.get("dislike_count", "0")) if reply.get("dislike_count") else 0
            replies.extend(page_replies)
            page += 1
            if not page_replies:
                logger.info(f"LIHKG 帖子無更多回覆: thread_id={thread_id}, page={page}")
                break
        except Exception as e:
            logger.warning(f"LIHKG 帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤: {str(e)}")
            break
    
    return replies[:max_replies]
