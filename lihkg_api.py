"""
LIHKG API 模組，負責從 LIHKG 論壇抓取帖子標題和回覆內容。
提供速率限制管理、錯誤處理和日誌記錄功能，支援動態延遲。
主要函數：
- get_lihkg_topic_list：抓取指定分類的帖子標題。
- get_lihkg_thread_content：抓取指定帖子的回覆內容。
- get_category_name：返回分類名稱。
"""

import aiohttp
import asyncio
import time
from datetime import datetime
import random
import hashlib
import logging
import json

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 設置為 DEBUG 以捕獲詳細信息
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 檔案處理器：寫入 app.log
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台處理器：輸出到 stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# LIHKG API 基礎配置
LIHKG_BASE_URL = "https://lihkg.com"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"

# 用戶代理列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]

class RateLimiter:
    """
    速率限制器，控制 API 請求頻率。
    """
    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.requests = []

    async def acquire(self):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.period]
        if len(self.requests) >= self.max_requests:
            wait_time = self.period - (now - self.requests[0])
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.requests = self.requests[1:]
        self.requests.append(now)

class ApiClient:
    """
    LIHKG API 客戶端，處理共用請求邏輯，支援動態延遲。
    """
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.base_url = LIHKG_BASE_URL
        self.device_id = LIHKG_DEVICE_ID
        self.cookie = LIHKG_COOKIE
        self.avg_response_time = 1.0
        self.response_times = []

    def generate_headers(self, url: str, timestamp: int):
        # 改進 X-LI-DIGEST 生成邏輯，確保正確編碼
        encoded_url = url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c').replace(' ', '%20')
        digest_input = f"jeams$get${encoded_url}${timestamp}"
        digest = hashlib.sha1(digest_input.encode('utf-8')).hexdigest()
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "X-LI-DEVICE": self.device_id,
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "Cookie": self.cookie,
            "Accept": "application/json",
            "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Referer": url.rsplit('/', 1)[0]
        }

    async def get(self, url: str, function_name: str, params=None, max_retries=5, timeout=10):
        timestamp = int(time.time())
        headers = self.generate_headers(url, timestamp)
        rate_limit_info = []
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params, timeout=timeout) as response:
                        response_time = time.time() - start_time
                        self.response_times.append(response_time)
                        if len(self.response_times) > 10:
                            self.response_times.pop(0)
                        self.avg_response_time = sum(self.response_times) / len(self.response_times)
                        delay = max(0.5, min(2.0, self.avg_response_time * 0.8))
                        await asyncio.sleep(delay)
                        status = "success" if response.status == 200 else f"failed_status_{response.status}"
                        response_text = await response.text()
                        logger.info(
                            json.dumps({
                                "event": "lihkg_api_request",
                                "function": function_name,
                                "url": url,
                                "status": status,
                                "response_sample": response_text[:200]
                            }, ensure_ascii=False)
                        )
                        if response.status == 429:
                            wait_time = int(response.headers.get("Retry-After", "5"))
                            rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit hit, waiting {wait_time:.2f} seconds")
                            logger.warning(f"Rate limit hit for {function_name}, url={url}")
                            await asyncio.sleep(wait_time)
                            continue
                        if response.status != 200:
                            logger.error(f"Fetch failed: {function_name}, url={url}, status={response.status}, response={response_text[:500]}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            return None, rate_limit_info
                        data = await response.json()
                        if not data.get("success"):
                            error_message = data.get('error_message', 'Unknown')
                            logger.error(f"API error: {function_name}, url={url}, message={error_message}, response={response_text[:500]}")
                            if error_message == "Error (2)" and attempt < max_retries - 1:
                                await asyncio.sleep(5 + attempt * 5)
                                timestamp = int(time.time())
                                headers = self.generate_headers(url, timestamp)
                                continue
                            return None, rate_limit_info
                        return data, rate_limit_info
            except Exception as e:
                logger.error(
                    json.dumps({
                        "event": "lihkg_api_request",
                        "function": function_name,
                        "url": url,
                        "status": "failed",
                        "error": str(e)
                    }, ensure_ascii=False)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 + attempt * 2)
                    continue
                return None, rate_limit_info

# 初始化速率限制器和客戶端
rate_limiter = RateLimiter(max_requests=50, period=60)
api_client = ApiClient(rate_limiter)

def get_category_name(cat_id):
    """
    根據分類 ID 返回分類名稱。
    """
    categories = {
        "1": "吹水台", "2": "熱門台", "5": "時事台", "14": "上班台",
        "15": "財經台", "29": "成人台", "31": "創意台"
    }
    return categories.get(str(cat_id), "未知分類")

async def get_lihkg_topic_list(cat_id, start_page=1, max_pages=3, request_counter=0, last_reset=0, rate_limit_until=0):
    """
    抓取指定分類的帖子標題列表。
    """
    from grok_processing import safe_int  # Local import
    items = []
    rate_limit_info = []
    current_time = time.time()
    
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        logger.warning(f"Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        return {
            "items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter,
            "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    if current_time - last_reset >= 60:
        request_counter = 0
        last_reset = current_time
    
    for page in range(start_page, start_page + max_pages):
        url = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id={cat_id}&page={page}&count=60&type=now&order=now"
        data, page_rate_limit_info = await api_client.get(url, "get_lihkg_topic_list")
        request_counter += 1
        rate_limit_info.extend(page_rate_limit_info)
        
        if data and data.get("response", {}).get("items"):
            filtered_items = [
                {
                    **item,
                    "no_of_reply": safe_int(item.get("no_of_reply", 0), 0),
                    "like_count": safe_int(item.get("like_count", 0), 0),
                    "dislike_count": safe_int(item.get("dislike_count", 0), 0),
                    "total_pages": safe_int(item.get("total_pages", 1), 1)
                }
                for item in data["response"]["items"]
                if item.get("title") and safe_int(item.get("no_of_reply", 0), 0) > 0
            ]
            logger.debug(f"Fetched cat_id={cat_id}, page={page}, items={len(filtered_items)}, sample_data={[{'thread_id': item['thread_id'], 'no_of_reply': item['no_of_reply'], 'like_count': item['like_count'], 'total_pages': item['total_pages']} for item in filtered_items[:3]]}")
            items.extend(filtered_items)
        else:
            logger.error(f"No data fetched for cat_id={cat_id}, page={page}")
        
        await asyncio.sleep(1)
    
    return {
        "items": items[:90],
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def get_lihkg_thread_content(thread_id, cat_id, request_counter=0, last_reset=0, rate_limit_until=0, max_replies=100, fetch_last_pages=0, specific_pages=None, start_page=1):
    """
    抓取指定帖子的內容，包括回覆。
    """
    from grok_processing import clean_html, safe_int  # Local import
    replies = []
    rate_limit_info = []
    fetched_pages = []
    current_time = time.time()
    
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        logger.warning(f"Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        return {
            "thread_id": thread_id, "title": "", "replies": replies, "fetched_pages": fetched_pages,
            "total_replies": 0, "rate_limit_info": rate_limit_info, "request_counter": request_counter,
            "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    if current_time - last_reset >= 60:
        request_counter = 0
        last_reset = current_time
    
    total_pages = 1
    pages_to_fetch = []
    
    if specific_pages:
        pages_to_fetch = specific_pages
    elif fetch_last_pages:
        url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}?page=1"
        data, page_rate_limit_info = await api_client.get(url, f"get_lihkg_thread_content_page_1")
        request_counter += 1
        rate_limit_info.extend(page_rate_limit_info)
        if data and data.get("response", {}).get("total_page"):
            total_pages = safe_int(data["response"]["total_page"], 1)
            pages_to_fetch = list(range(max(1, total_pages - fetch_last_pages + 1), total_pages + 1))
        else:
            logger.error(f"Failed to fetch thread_id={thread_id}, page=1")
            return {
                "thread_id": thread_id, "title": "", "replies": [], "fetched_pages": [],
                "total_replies": 0, "rate_limit_info": rate_limit_info, "request_counter": request_counter,
                "last_reset": last_reset, "rate_limit_until": rate_limit_until
            }
    else:
        pages_to_fetch = list(range(start_page, min(start_page + 3, total_pages + 1)))
    
    title = ""
    for page in pages_to_fetch:
        url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}?page={page}"
        data, page_rate_limit_info = await api_client.get(url, f"get_lihkg_thread_content_page_{page}")
        request_counter += 1
        rate_limit_info.extend(page_rate_limit_info)
        
        if data and data.get("response", {}).get("items"):
            if not title and data["response"].get("title"):
                title = clean_html(data["response"]["title"])
            page_replies = data["response"]["items"]
            logger.debug(f"Thread_id={thread_id}, page={page}, raw_replies={len(page_replies)}, sample_data={[{'post_id': r.get('post_id'), 'like_count': r.get('like_count'), 'dislike_count': r.get('dislike_count')} for r in page_replies[:3]]}")
            for reply in page_replies:
                msg = clean_html(reply.get("msg", ""))
                if not msg:
                    continue
                normalized_reply = {
                    "post_id": reply.get("post_id"),
                    "msg": msg,
                    "like_count": safe_int(reply.get("like_count", 0), 0),
                    "dislike_count": safe_int(reply.get("dislike_count", 0), 0),
                    "reply_time": reply.get("reply_time", "0")
                }
                replies.append(normalized_reply)
            fetched_pages.append(page)
            if len(replies) >= max_replies:
                replies = replies[:max_replies]
                break
        else:
            logger.warning(f"No data fetched for thread_id={thread_id}, page={page}")
        
        await asyncio.sleep(1)
    
    total_replies = safe_int(data["response"].get("total_reply", 0), len(replies)) if data and data.get("response") else len(replies)
    logger.info(f"Fetched thread_id={thread_id}, total_replies={total_replies}, fetched_pages={fetched_pages}")
    
    return {
        "thread_id": thread_id,
        "title": title,
        "replies": replies,
        "fetched_pages": fetched_pages,
        "total_replies": total_replies,
        "total_pages": total_pages,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }
