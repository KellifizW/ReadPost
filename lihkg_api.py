"""
LIHKG API 模組，負責抓取帖子標題和回覆內容，內嵌日誌配置。
"""

import aiohttp
import asyncio
import time
from datetime import datetime
import random
import hashlib
import logging
import json
import pytz

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 日誌配置
class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S,%f")[:-3] + " HKT"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler("lihkg_api.log")
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# LIHKG API 配置
LIHKG_BASE_URL = "https://lihkg.com"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]

class RateLimiter:
    """速率限制器"""
    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.requests = []
        self.request_counter = 0
        self.last_reset = time.time()
        self.rate_limit_until = 0
        self.last_response_time = 1.0

    async def acquire(self):
        now = time.time()
        if now - self.last_reset >= self.period:
            self.request_counter = 0
            self.last_reset = now
        if now < self.rate_limit_until:
            wait_time = self.rate_limit_until - now
            logger.warning(f"Rate limit active, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            return False, {
                "request_counter": self.request_counter,
                "last_reset": self.last_reset,
                "rate_limit_until": self.rate_limit_until
            }
        self.requests = [t for t in self.requests if now - t < self.period]
        if len(self.requests) >= self.max_requests:
            wait_time = self.period - (now - self.requests[0])
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.requests = self.requests[1:]
        self.requests.append(now)
        self.request_counter += 1
        await asyncio.sleep(max(0.5, min(self.last_response_time, 2.0)))
        return True, {
            "request_counter": self.request_counter,
            "last_reset": self.last_reset,
            "rate_limit_until": self.rate_limit_until
        }

    def update_rate_limit(self, retry_after):
        self.rate_limit_until = time.time() + int(retry_after)

class ApiClient:
    """LIHKG API 客戶端"""
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.base_url = LIHKG_BASE_URL
        self.device_id = LIHKG_DEVICE_ID
        self.cookie = LIHKG_COOKIE

    def generate_headers(self, url: str, timestamp: int):
        digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
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

    async def get(self, url: str, function_name: str, params=None, max_retries=3, timeout=10):
        rate_limit_info = []
        for attempt in range(max_retries):
            success, rate_limit_data = await self.rate_limiter.acquire()
            if not success:
                rate_limit_info.append(f"{datetime.now(tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} HKT - Rate limit active until {datetime.fromtimestamp(rate_limit_data['rate_limit_until'], tz=HONG_KONG_TZ)}")
                return None, rate_limit_info, rate_limit_data
            try:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.generate_headers(url, int(time.time())), params=params, timeout=timeout) as response:
                        response_time = time.time() - start_time
                        self.rate_limiter.last_response_time = response_time
                        logger.info(f"API response time: {response_time:.2f} seconds for {function_name}")
                        status = "success" if response.status == 200 else f"failed_status_{response.status}"
                        logger.info(
                            json.dumps({
                                "event": "lihkg_api_request",
                                "function": function_name,
                                "url": url,
                                "status": status
                            }, ensure_ascii=False)
                        )
                        if response.status == 429:
                            wait_time = response.headers.get("Retry-After", "5")
                            self.rate_limiter.update_rate_limit(wait_time)
                            rate_limit_info.append(f"{datetime.now(tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} HKT - Rate limit hit, waiting {wait_time:.2f} seconds")
                            continue
                        if response.status != 200:
                            logger.error(f"Fetch failed: {function_name}, url={url}, status={response.status}")
                            break
                        data = await response.json()
                        if not data.get("success"):
                            logger.error(f"API error: {function_name}, url={url}, message={data.get('error_message', 'Unknown')}")
                            break
                        return data, rate_limit_info, rate_limit_data
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
                    await asyncio.sleep(2)
                    continue
                break
        return None, rate_limit_info, rate_limit_data

# 初始化
rate_limiter = RateLimiter(max_requests=50, period=60)
api_client = ApiClient(rate_limiter)

def get_category_name(cat_id):
    """返回分類名稱"""
    categories = {
        "1": "吹水台", "2": "熱門台", "5": "時事台", "14": "上班台",
        "15": "財經台", "29": "成人台", "31": "創意台"
    }
    return categories.get(str(cat_id), "未知分類")

async def get_lihkg_topic_list(cat_id, start_page=1, max_pages=3):
    """抓取帖子標題列表"""
    items = []
    rate_limit_info = []
    
    for page in range(start_page, start_page + max_pages):
        if cat_id == "2":
            url = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id=1&page={page}&count=60&type=now&order=hot"
        else:
            url = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id={cat_id}&page={page}&count=60&type=now"
        data, page_rate_limit_info, rate_limit_data = await api_client.get(url, "get_lihkg_topic_list")
        rate_limit_info.extend(page_rate_limit_info)
        
        if data and data.get("response", {}).get("items"):
            filtered_items = [item for item in data["response"]["items"] if item.get("title") and item.get("no_of_reply", 0) > 0]
            items.extend(filtered_items)
            logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(filtered_items)}")
        else:
            logger.error(f"No data fetched for cat_id={cat_id}, page={page}")
        
        await asyncio.sleep(1)
    
    return {
        "items": items[:90],
        "rate_limit_info": rate_limit_info,
        "request_counter": rate_limit_data["request_counter"],
        "last_reset": rate_limit_data["last_reset"],
        "rate_limit_until": rate_limit_data["rate_limit_until"]
    }

async def get_lihkg_thread_content(thread_id, cat_id=None, max_replies=250, fetch_last_pages=0, specific_pages=None, start_page=1):
    """抓取帖子回覆內容"""
    replies = []
    fetched_pages = []
    thread_title = None
    total_replies = None
    total_pages = None
    rate_limit_info = []
    max_replies = max(max_replies, 250)
    
    url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/page/1?order=reply_time"
    data, page_rate_limit_info, rate_limit_data = await api_client.get(url, "get_lihkg_thread_content")
    rate_limit_info.extend(page_rate_limit_info)
    
    if data and data.get("response"):
        response_data = data.get("response", {})
        thread_title = response_data.get("title") or response_data.get("thread", {}).get("title")
        total_replies = response_data.get("total_replies", 0)
        total_pages = response_data.get("total_page", 1)
        
        page_replies = response_data.get("item_data", [])
        for reply in page_replies:
            reply["like_count"] = int(reply.get("like_count", "0"))
            reply["dislike_count"] = int(reply.get("dislike_count", "0"))
            reply["reply_time"] = reply.get("reply_time", "0")
        replies.extend(page_replies[:max_replies])
        fetched_pages.append(1)
        logger.info(f"Fetched thread_id={thread_id}, page=1, replies={len(page_replies)}, total_stored={len(replies)}")
    else:
        logger.error(f"No data fetched for thread_id={thread_id}, page=1")
        return {
            "replies": [], "title": None, "total_replies": 0, "total_pages": 0,
            "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
            "request_counter": rate_limit_data["request_counter"],
            "last_reset": rate_limit_data["last_reset"],
            "rate_limit_until": rate_limit_data["rate_limit_until"]
        }
    
    pages_to_fetch = []
    if specific_pages:
        pages_to_fetch = [p for p in specific_pages if 1 <= p <= total_pages and p not in fetched_pages]
    elif fetch_last_pages > 0:
        start = max(start_page, 2)
        end = total_pages + 1
        if end - start < fetch_last_pages:
            start = max(2, end - fetch_last_pages)
        pages_to_fetch = list(range(start, end))[:fetch_last_pages]
        pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages and 1 <= p <= total_pages]
    elif start_page > 1:
        pages_to_fetch = list(range(start_page, total_pages + 1))
        pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages]
    
    pages_to_fetch = sorted(set(pages_to_fetch))
    
    for page in pages_to_fetch:
        if len(replies) >= max_replies:
            logger.info(f"Stopped fetching: thread_id={thread_id}, replies={len(replies)} reached max {max_replies}")
            break
        
        url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/page/{page}?order=reply_time"
        data, page_rate_limit_info, rate_limit_data = await api_client.get(url, "fetch_thread_page")
        rate_limit_info.extend(page_rate_limit_info)
        
        if data and data.get("response"):
            page_replies = data["response"].get("item_data", [])
            for reply in page_replies:
                reply["like_count"] = int(reply.get("like_count", "0"))
                reply["dislike_count"] = int(reply.get("dislike_count", "0"))
                reply["reply_time"] = reply.get("reply_time", "0")
            remaining_slots = max_replies - len(replies)
            page_replies = page_replies[:remaining_slots]
            replies.extend(page_replies)
            fetched_pages.append(page)
            logger.info(f"Fetched thread_id={thread_id}, page={page}, replies={len(page_replies)}")
        await asyncio.sleep(1)
    
    return {
        "replies": replies, "title": thread_title, "total_replies": total_replies,
        "total_pages": total_pages, "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
        "request_counter": rate_limit_data["request_counter"],
        "last_reset": rate_limit_data["last_reset"],
        "rate_limit_until": rate_limit_data["rate_limit_until"]
    }
