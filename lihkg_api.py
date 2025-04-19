"""
LIHKG API 模組，提供帖子列表和內容抓取功能。
主要函數：
- get_category_name：獲取分類名稱。
- get_lihkg_topic_list：抓取帖子列表。
- get_lihkg_thread_content：抓取帖子內容。
硬編碼參數：
- BASE_URL
- HEADERS（部分）
- count=60
- max_replies=1000
- sleep=0.5/1
- timeout=10
- retries=3
"""

import aiohttp
import asyncio
import time
import logging
import hashlib
import json

logger = logging.getLogger(__name__)
BASE_URL = "https://lihkg.com/api_v2"
DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
SESSION_ID = "ckdp63v3gapcpo8jfngun6t3av"
CFRUID = "019429f"

def generate_digest(url, timestamp):
    secret = "lihkg"
    return hashlib.sha1(f"{url}{timestamp}{secret}".encode()).hexdigest()

async def get_category_name(cat_id, request_counter=0, last_reset=0, rate_limit_until=0):
    url = f"{BASE_URL}/category?cat_id={cat_id}"
    timestamp = str(int(time.time()))
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "X-LI-DEVICE": DEVICE_ID,
        "X-LI-REQUEST-TIME": timestamp,
        "Cookie": f"PHPSESSID={SESSION_ID}; __cfruid={CFRUID}",
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"https://lihkg.com/category/{cat_id}",
        "X-LI-DIGEST": generate_digest(url, timestamp)
    }
    
    if time.time() < rate_limit_until:
        logger.warning(f"Rate limit active until {rate_limit_until} for cat_id={cat_id}")
        return {
            "name": None,
            "rate_limit_info": [{"cat_id": cat_id, "timestamp": time.time(), "status": "rate_limited"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    for attempt in range(3):
        try:
            logger.info(f"LIHKG API request: cat_id={cat_id}, url={url}, headers={headers}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    data = await response.json()
                    logger.info(f"LIHKG API response: cat_id={cat_id}, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                    if response.status == 200 and data.get("success"):
                        return {
                            "name": data["response"]["name"],
                            "rate_limit_info": [{"cat_id": cat_id, "timestamp": time.time(), "status": "success"}],
                            "request_counter": request_counter + 1,
                            "last_reset": last_reset,
                            "rate_limit_until": rate_limit_until
                        }
                    elif response.status == 429:
                        rate_limit_until = time.time() + int(response.headers.get("X-RateLimit-Reset", 60))
                        logger.warning(f"Rate limit hit for cat_id={cat_id}, retry after {rate_limit_until}")
                        return {
                            "name": None,
                            "rate_limit_info": [{"cat_id": cat_id, "timestamp": time.time(), "status": "rate_limited"}],
                            "request_counter": request_counter,
                            "last_reset": last_reset,
                            "rate_limit_until": rate_limit_until
                        }
                    else:
                        logger.error(f"Fetch error: cat_id={cat_id}, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"Fetch error: cat_id={cat_id}, attempt={attempt+1}, error={str(e)}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "name": None,
                "rate_limit_info": [{"cat_id": cat_id, "timestamp": time.time(), "status": "error", "error": str(e)}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until
            }
    return {
        "name": None,
        "rate_limit_info": [{"cat_id": cat_id, "timestamp": time.time(), "status": "max_retries_exceeded"}],
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def get_lihkg_topic_list(cat_id, start_page=1, max_pages=1, request_counter=0, last_reset=0, rate_limit_until=0):
    if time.time() > last_reset + 3600:
        request_counter = 0
        last_reset = time.time()
    
    items = []
    rate_limit_info = []
    user_agents = [
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ]
    
    for page in range(start_page, start_page + max_pages):
        if time.time() < rate_limit_until:
            logger.warning(f"Rate limit active until {rate_limit_until} for cat_id={cat_id}, page={page}")
            rate_limit_info.append({"cat_id": cat_id, "page": page, "timestamp": time.time(), "status": "rate_limited"})
            return {
                "items": items,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until
            }
        
        url = f"{BASE_URL}/thread/latest?cat_id={cat_id}&page={page}&count=60&type=now&order=now"
        timestamp = str(int(time.time()))
        headers = {
            "User-Agent": user_agents[page % len(user_agents)],
            "X-LI-DEVICE": DEVICE_ID,
            "X-LI-REQUEST-TIME": timestamp,
            "Cookie": f"PHPSESSID={SESSION_ID}; __cfruid={CFRUID}",
            "Accept": "application/json",
            "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Referer": f"https://lihkg.com/category/{cat_id}",
            "X-LI-DIGEST": generate_digest(url, timestamp)
        }
        
        for attempt in range(3):
            try:
                logger.info(f"LIHKG API request: cat_id={cat_id}, page={page}, attempt={attempt+1}, url={url}, headers={headers}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=10) as response:
                        data = await response.json()
                        logger.info(f"LIHKG API response: cat_id={cat_id}, page={page}, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                        if response.status == 200 and data.get("success"):
                            threads = data["response"].get("items", [])
                            # 過濾僅包含指定 cat_id 的帖子
                            items.extend([
                                {
                                    "thread_id": thread["thread_id"],
                                    "title": thread["title"],
                                    "no_of_reply": thread.get("total_reply", 0),
                                    "last_reply_time": thread.get("last_time", 0)
                                } for thread in threads if str(thread["cat_id"]) == str(cat_id)
                            ])
                            rate_limit_info.append({"cat_id": cat_id, "page": page, "timestamp": time.time(), "status": "success"})
                            request_counter += 1
                            break
                        elif response.status == 429:
                            rate_limit_until = time.time() + int(response.headers.get("X-RateLimit-Reset", 60))
                            logger.warning(f"Rate limit hit for cat_id={cat_id}, page={page}, retry after {rate_limit_until}")
                            rate_limit_info.append({"cat_id": cat_id, "page": page, "timestamp": time.time(), "status": "rate_limited"})
                            return {
                                "items": items,
                                "rate_limit_info": rate_limit_info,
                                "request_counter": request_counter,
                                "last_reset": last_reset,
                                "rate_limit_until": rate_limit_until
                            }
                        else:
                            logger.error(f"Fetch error: cat_id={cat_id}, page={page}, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                            rate_limit_info.append({"cat_id": cat_id, "page": page, "timestamp": time.time(), "status": "error", "error": f"Status {response.status}"})
            except Exception as e:
                logger.error(f"Fetch error: cat_id={cat_id}, page={page}, attempt={attempt+1}, error={str(e)}")
                rate_limit_info.append({"cat_id": cat_id, "page": page, "timestamp": time.time(), "status": "error", "error": str(e)})
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "items": items,
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": rate_limit_until
                }
        await asyncio.sleep(0.5)
    
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def get_lihkg_thread_content(thread_id, cat_id, request_counter=0, last_reset=0, rate_limit_until=0, max_replies=1000, fetch_last_pages=0):
    if time.time() > last_reset + 3600:
        request_counter = 0
        last_reset = time.time()
    
    if time.time() < rate_limit_until:
        logger.warning(f"Rate limit active until {rate_limit_until} for thread_id={thread_id}")
        return {
            "title": None,
            "replies": [],
            "total_replies": 0,
            "last_reply_time": 0,
            "fetched_pages": [],
            "rate_limit_info": [{"thread_id": thread_id, "timestamp": time.time(), "status": "rate_limited"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    url = f"{BASE_URL}/thread/{thread_id}/page/1?order=reply_time"
    timestamp = str(int(time.time()))
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "X-LI-DEVICE": DEVICE_ID,
        "X-LI-REQUEST-TIME": timestamp,
        "Cookie": f"PHPSESSID={SESSION_ID}; __cfruid={CFRUID}",
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"https://lihkg.com/thread/{thread_id}",
        "X-LI-DIGEST": generate_digest(url, timestamp)
    }
    
    rate_limit_info = []
    replies = []
    fetched_pages = []
    total_replies = 0
    title = None
    last_reply_time = 0
    
    for attempt in range(3):
        try:
            logger.info(f"LIHKG API request: thread_id={thread_id}, page=1, attempt={attempt+1}, url={url}, headers={headers}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    data = await response.json()
                    logger.info(f"LIHKG API response: thread_id={thread_id}, page=1, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                    if response.status == 200 and data.get("success"):
                        thread = data["response"]
                        title = thread.get("title")
                        total_replies = thread.get("total_reply", 0)
                        last_reply_time = thread.get("last_reply_time", 0)
                        replies.extend([
                            {
                                "msg": reply["msg"],
                                "reply_time": reply.get("reply_time", 0)
                            } for reply in thread.get("item_data", [])
                        ])
                        fetched_pages.append(1)
                        request_counter += 1
                        break
                    elif response.status == 429:
                        rate_limit_until = time.time() + int(response.headers.get("X-RateLimit-Reset", 60))
                        logger.warning(f"Rate limit hit for thread_id={thread_id}, retry after {rate_limit_until}")
                        rate_limit_info.append({"thread_id": thread_id, "timestamp": time.time(), "status": "rate_limited"})
                        return {
                            "title": None,
                            "replies": [],
                            "total_replies": 0,
                            "last_reply_time": 0,
                            "fetched_pages": [],
                            "rate_limit_info": rate_limit_info,
                            "request_counter": request_counter,
                            "last_reset": last_reset,
                            "rate_limit_until": rate_limit_until
                        }
                    else:
                        logger.error(f"Fetch error: thread_id={thread_id}, page=1, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                        rate_limit_info.append({"thread_id": thread_id, "timestamp": time.time(), "status": "error", "error": f"Status {response.status}"})
        except Exception as e:
            logger.error(f"Fetch error: thread_id={thread_id}, page=1, attempt={attempt+1}, error={str(e)}")
            rate_limit_info.append({"thread_id": thread_id, "timestamp": time.time(), "status": "error", "error": str(e)})
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "title": None,
                "replies": [],
                "total_replies": 0,
                "last_reply_time": 0,
                "fetched_pages": [],
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until
            }
    
    total_pages = (total_replies + 24) // 25
    if fetch_last_pages > 0 and total_pages > 1:
        start_page = max(2, total_pages - fetch_last_pages + 1)
        for page in range(start_page, total_pages + 1):
            if len(replies) >= max_replies:
                break
            if time.time() < rate_limit_until:
                logger.warning(f"Rate limit active until {rate_limit_until} for thread_id={thread_id}, page={page}")
                rate_limit_info.append({"thread_id": thread_id, "page": page, "timestamp": time.time(), "status": "rate_limited"})
                break
            
            url = f"{BASE_URL}/thread/{thread_id}/page/{page}?order=reply_time"
            timestamp = str(int(time.time()))
            headers["X-LI-REQUEST-TIME"] = timestamp
            headers["X-LI-DIGEST"] = generate_digest(url, timestamp)
            headers["Referer"] = f"https://lihkg.com/thread/{thread_id}/page/{page}"
            
            for attempt in range(3):
                try:
                    logger.info(f"LIHKG API request: thread_id={thread_id}, page={page}, attempt={attempt+1}, url={url}, headers={headers}")
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers, timeout=10) as response:
                            data = await response.json()
                            logger.info(f"LIHKG API response: thread_id={thread_id}, page={page}, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                            if response.status == 200 and data.get("success"):
                                replies.extend([
                                    {
                                        "msg": reply["msg"],
                                        "reply_time": reply.get("reply_time", 0)
                                    } for reply in data["response"].get("item_data", [])
                                ])
                                fetched_pages.append(page)
                                request_counter += 1
                                break
                            elif response.status == 429:
                                rate_limit_until = time.time() + int(response.headers.get("X-RateLimit-Reset", 60))
                                logger.warning(f"Rate limit hit for thread_id={thread_id}, page={page}, retry after {rate_limit_until}")
                                rate_limit_info.append({"thread_id": thread_id, "page": page, "timestamp": time.time(), "status": "rate_limited"})
                                return {
                                    "title": title,
                                    "replies": replies,
                                    "total_replies": total_replies,
                                    "last_reply_time": last_reply_time,
                                    "fetched_pages": fetched_pages,
                                    "rate_limit_info": rate_limit_info,
                                    "request_counter": request_counter,
                                    "last_reset": last_reset,
                                    "rate_limit_until": rate_limit_until
                                }
                            else:
                                logger.error(f"Fetch error: thread_id={thread_id}, page={page}, status={response.status}, response={json.dumps(data, ensure_ascii=False)}")
                                rate_limit_info.append({"thread_id": thread_id, "page": page, "timestamp": time.time(), "status": "error", "error": f"Status {response.status}"})
                except Exception as e:
                    logger.error(f"Fetch error: thread_id={thread_id}, page={page}, attempt={attempt+1}, error={str(e)}")
                    rate_limit_info.append({"thread_id": thread_id, "page": page, "timestamp": time.time(), "status": "error", "error": str(e)})
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {
                        "title": title,
                        "replies": replies,
                        "total_replies": total_replies,
                        "last_reply_time": last_reply_time,
                        "fetched_pages": fetched_pages,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until
                    }
            await asyncio.sleep(1)
    
    return {
        "title": title,
        "replies": replies[:max_replies],
        "total_replies": total_replies,
        "last_reply_time": last_reply_time,
        "fetched_pages": fetched_pages,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }