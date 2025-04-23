"""
LIHKG API 模組，負責從 LIHKG 論壇抓取帖子標題和回覆內容。
提供速率限制管理、錯誤處理和日誌記錄功能。
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
import pytz

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 自定義日誌格式器，將時間戳設為香港時區
class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

# 防止重複添加處理器
if not logger.handlers:
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
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
    LIHKG API 客戶端，處理共用請求邏輯。
    """
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
        timestamp = int(time.time())
        headers = self.generate_headers(url, timestamp)
        rate_limit_info = []
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params, timeout=timeout) as response:
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
                            wait_time = int(response.headers.get("Retry-After", "5"))
                            rate_limit_info.append(f"{datetime.now(tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} - Rate limit hit, waiting {wait_time:.2f} seconds")
                            logger.warning(f"Rate limit hit for {function_name}, url={url}")
                            await asyncio.sleep(wait_time)
                            continue
                        if response.status != 200:
                            logger.error(f"Fetch failed: {function_name}, url={url}, status={response.status}")
                            break
                        data = await response.json(content_type=None)
                        if not data.get("success"):
                            logger.error(f"API error: {function_name}, url={url}, message={data.get('error_message', 'Unknown')}")
                            break
                        return data, rate_limit_info
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Request error in {function_name}: {str(e)}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 + attempt * 2)
                    continue
                return {"success": False, "error_message": str(e)}, rate_limit_info
        return {"success": False, "error_message": "Max retries exceeded"}, rate_limit_info

# 全局速率限制器
rate_limiter = RateLimiter(max_requests=20, period=60)
api_client = ApiClient(rate_limiter)

async def get_lihkg_topic_list(cat_id: str, order: str, start_page: int, max_pages: int, request_counter: int, last_reset: float, rate_limit_until: float):
    """
    抓取 LIHKG 指定分類的帖子標題列表。
    """
    logger.info(f"Fetching topic list: cat_id={cat_id}, order={order}, start_page={start_page}, max_pages={max_pages}")
    
    now = time.time()
    if now - last_reset > 3600:
        request_counter = 0
        last_reset = now
    if now < rate_limit_until:
        logger.warning(f"Rate limit active until {datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
        return {
            "success": False,
            "items": [],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "rate_limit_info": []
        }

    items = []
    rate_limit_info = []
    pages_fetched = 0
    max_items = 135

    for page in range(start_page, start_page + max_pages):
        if len(items) >= max_items:
            break
        url = f"{LIHKG_BASE_URL}/api/v2/thread/category"
        params = {
            "cat_id": cat_id,
            "page": page,
            "order": order,
            "count": 45
        }
        data, rate_info = await api_client.get(url, "get_lihkg_topic_list", params=params)
        rate_limit_info.extend(rate_info)
        
        if not data.get("success"):
            logger.error(f"Failed to fetch topic list: page={page}, error={data.get('error_message', 'Unknown')}")
            break
        
        response = data.get("response", {})
        threads = response.get("items", [])
        if not threads:
            break
        
        for thread in threads:
            items.append({
                "thread_id": str(thread.get("thread_id", "")),
                "title": thread.get("title", ""),
                "no_of_reply": thread.get("total_replies", 0),
                "like_count": thread.get("like_count", 0),
                "dislike_count": thread.get("dislike_count", 0),
                "last_reply_time": str(thread.get("last_post_time", "0"))
            })
        
        pages_fetched += 1
        request_counter += 1
        if request_counter >= 100:
            rate_limit_until = now + 300
            logger.warning(f"Request counter reached limit, setting rate limit until {datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
            break
    
    logger.info(f"Fetched {len(items)} topics across {pages_fetched} pages")
    return {
        "success": True,
        "items": items,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until,
        "rate_limit_info": rate_limit_info
    }

async def get_lihkg_thread_content(thread_id: str, cat_id: str, request_counter: int, last_reset: float, rate_limit_until: float, max_replies: int = 50):
    """
    抓取指定帖子的回覆內容。
    """
    logger.info(f"Fetching thread content: thread_id={thread_id}, cat_id={cat_id}, max_replies={max_replies}")
    
    now = time.time()
    if now - last_reset > 3600:
        request_counter = 0
        last_reset = now
    if now < rate_limit_until:
        logger.warning(f"Rate limit active until {datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
        return {
            "success": False,
            "thread_id": thread_id,
            "title": "",
            "replies": [],
            "total_replies": 0,
            "fetched_pages": [],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "rate_limit_info": []
        }

    replies = []
    rate_limit_info = []
    page = 1
    fetched_pages = []
    max_pages = (max_replies // 25) + 1 if max_replies > 0 else 2

    while page <= max_pages:
        url = f"{LIHKG_BASE_URL}/api/v2/thread/{thread_id}/post"
        params = {
            "post_type": "post",
            "page": page,
            "order": "reply_time"
        }
        data, rate_info = await api_client.get(url, "get_lihkg_thread_content", params=params)
        rate_limit_info.extend(rate_info)
        
        if not data.get("success"):
            logger.error(f"Failed to fetch thread content: thread_id={thread_id}, page={page}, error={data.get('error_message', 'Unknown')}")
            break
        
        response = data.get("response", {})
        thread = response.get("thread", {})
        posts = response.get("items", [])
        
        if not posts:
            break
        
        for post in posts:
            if len(replies) >= max_replies:
                break
            replies.append({
                "post_id": str(post.get("post_id", "")),
                "msg": post.get("msg", ""),
                "like_count": post.get("like_count", 0),
                "dislike_count": post.get("dislike_count", 0),
                "reply_time": str(post.get("reply_time", "0"))
            })
        
        fetched_pages.append(page)
        page += 1
        request_counter += 1
        if request_counter >= 100:
            rate_limit_until = now + 300
            logger.warning(f"Request counter reached limit, setting rate limit until {datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
            break
    
    logger.info(f"Fetched {len(replies)} replies for thread_id={thread_id} across {len(fetched_pages)} pages")
    return {
        "success": True,
        "thread_id": thread_id,
        "title": thread.get("title", ""),
        "replies": replies,
        "total_replies": thread.get("total_replies", 0),
        "fetched_pages": fetched_pages,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until,
        "rate_limit_info": rate_limit_info
    }

def get_category_name(cat_id: str) -> str:
    """
    根據 cat_id 返回分類名稱。
    """
    category_map = {
        "1": "吹水台",
        "5": "時事台",
        "14": "上班台",
        "15": "財經台",
        "29": "成人台",
        "31": "創意台"
    }
    return category_map.get(cat_id, "未知分類")
