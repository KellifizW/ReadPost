import aiohttp
import asyncio
import logging
import time
import random
import streamlit as st
from logging_config import configure_logger

# 配置日誌記錄器
logger = configure_logger(__name__, "lihkg_api.log")

# LIHKG API 配置
LIHKG_API_BASE = "https://lihkg.com/api_v2"
THREAD_LIST_ENDPOINT = f"{LIHKG_API_BASE}/thread/category"
THREAD_CONTENT_ENDPOINT = f"{LIHKG_API_BASE}/thread/{{thread_id}}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
API_TIMEOUT = 30  # 秒
MAX_RETRIES = 3
BASE_DELAY = 1.0

async def get_lihkg_topic_list(cat_id, page=1, count=30, order="popular"):
    """
    從 LIHKG API 獲取指定討論區的帖子列表。
    """
    params = {
        "cat_id": cat_id,
        "page": page,
        "count": min(count, 100),  # API 限制每頁最多100
        "order": "reply_time" if order == "recent" else "score"
    }
    headers = {"User-Agent": USER_AGENT}
    
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(THREAD_LIST_ENDPOINT, params=params, headers=headers, timeout=API_TIMEOUT) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limit hit, retry after {retry_after} seconds")
                        return {"error": "rate_limit", "retry_after": retry_after}
                    if response.status != 200:
                        logger.warning(f"Failed to fetch topic list: status={response.status}")
                        raise Exception(f"HTTP {response.status}")
                    data = await response.json()
                    if not data.get("success"):
                        logger.warning(f"API returned failure: {data.get('error_message', 'Unknown error')}")
                        raise Exception(data.get("error_message", "Unknown error"))
                    items = data.get("response", {}).get("items", [])
                    logger.info(f"Fetched {len(items)} threads from cat_id={cat_id}, page={page}")
                    return {
                        "items": [
                            {
                                "thread_id": item.get("thread_id"),
                                "title": item.get("title"),
                                "no_of_reply": item.get("total_replies", 0),
                                "like_count": item.get("like_count", 0),
                                "dislike_count": item.get("dislike_count", 0),
                                "last_reply_time": item.get("last_reply_time", 0)
                            }
                            for item in items
                        ],
                        "total_page": data.get("response", {}).get("total_page", 1)
                    }
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for topic list: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
                continue
            logger.error(f"Failed to fetch topic list after {MAX_RETRIES} attempts: {str(e)}")
            return {"error": str(e), "items": []}

async def get_lihkg_thread_content(thread_id, cat_id, max_replies=100, specific_pages=None, start_page=1):
    """
    從 LIHKG API 獲取指定帖子的內容，支援動態頁數選擇。
    """
    headers = {"User-Agent": USER_AGENT}
    total_replies = []
    fetched_pages = []
    total_pages = 1
    total_reply_count = 0
    page = start_page
    
    try:
        # 首先獲取帖子元數據
        async with aiohttp.ClientSession() as session:
            async with session.get(THREAD_CONTENT_ENDPOINT.format(thread_id=thread_id), headers=headers, timeout=API_TIMEOUT) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limit hit for thread {thread_id}, retry after {retry_after} seconds")
                    return {"error": "rate_limit", "retry_after": retry_after}
                if response.status != 200:
                    logger.warning(f"Failed to fetch thread {thread_id}: status={response.status}")
                    raise Exception(f"HTTP {response.status}")
                data = await response.json()
                if not data.get("success"):
                    logger.warning(f"API returned failure for thread {thread_id}: {data.get('error_message', 'Unknown error')}")
                    raise Exception(data.get("error_message", "Unknown error"))
                
                item = data.get("response", {}).get("item_data", {})
                total_pages = data.get("response", {}).get("total_page", 1)
                total_reply_count = item.get("total_replies", 0)
                logger.info(f"Thread {thread_id} has {total_reply_count} replies, {total_pages} pages")
                
                thread_info = {
                    "thread_id": item.get("thread_id"),
                    "title": item.get("title"),
                    "total_replies": total_reply_count,
                    "like_count": item.get("like_count", 0),
                    "dislike_count": item.get("dislike_count", 0),
                    "last_reply_time": item.get("last_reply_time", 0),
                    "total_pages": total_pages
                }
        
        # 動態選擇頁數
        if specific_pages:
            pages_to_fetch = specific_pages
        else:
            replies_per_page = 25  # LIHKG 每頁約25條回覆
            pages_needed = max(1, min((max_replies + replies_per_page - 1) // replies_per_page, total_pages))
            pages_to_fetch = list(range(start_page, min(start_page + pages_needed, total_pages + 1)))
            if max_replies > 0 and len(pages_to_fetch) < 3 and total_pages > len(pages_to_fetch):
                pages_to_fetch.extend(range(total_pages - 2, total_pages + 1))
            pages_to_fetch = sorted(list(set(pages_to_fetch)))
        
        # 並行抓取頁數
        tasks = []
        for page in pages_to_fetch:
            if page <= total_pages:
                tasks.append(fetch_thread_page(thread_id, page, headers))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for page, result in zip(pages_to_fetch, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch page {page} for thread {thread_id}: {str(result)}")
                continue
            total_replies.extend(result.get("replies", []))
            fetched_pages.append(page)
        
        # 清理並限制回覆數量
        total_replies = total_replies[:max_replies]
        logger.info(f"Fetched {len(total_replies)} replies for thread {thread_id}, pages: {fetched_pages}")
        
        return {
            **thread_info,
            "replies": total_replies,
            "fetched_pages": sorted(fetched_pages),
            "total_pages": total_pages
        }
    
    except Exception as e:
        logger.error(f"Failed to fetch thread content for {thread_id}: {str(e)}")
        return {"error": str(e), "replies": [], "fetched_pages": [], "total_pages": 1}

async def fetch_thread_page(thread_id, page, headers):
    """
    抓取單個帖子頁面的回覆內容。
    """
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    THREAD_CONTENT_ENDPOINT.format(thread_id=thread_id),
                    params={"page": page},
                    headers=headers,
                    timeout=API_TIMEOUT
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limit hit for thread {thread_id}, page {page}, retry after {retry_after} seconds")
                        raise Exception(f"Rate limit, retry after {retry_after} seconds")
                    if response.status != 200:
                        logger.warning(f"Failed to fetch thread {thread_id}, page {page}: status={response.status}")
                        raise Exception(f"HTTP {response.status}")
                    data = await response.json()
                    if not data.get("success"):
                        logger.warning(f"API returned failure for thread {thread_id}, page {page}: {data.get('error_message', 'Unknown error')}")
                        raise Exception(data.get("error_message", "Unknown error"))
                    
                    replies = [
                        {
                            "reply_id": reply.get("post_id"),
                            "msg": reply.get("msg"),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": reply.get("reply_time", "0")
                        }
                        for reply in data.get("response", {}).get("replies", [])
                    ]
                    logger.info(f"Fetched {len(replies)} replies for thread {thread_id}, page {page}")
                    return {"replies": replies}
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for thread {thread_id}, page {page}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
                continue
            raise
