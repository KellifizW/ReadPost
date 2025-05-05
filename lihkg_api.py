"""
LIHKG API 模組，負責從 LIHKG 論壇抓取帖子標題和回覆內容。
提供速率限制管理、錯誤處理和日誌記錄功能。
主要函數：
- get_lihkg_topic_list：抓取指定分類的帖子標題列表。
- get_lihkg_thread_content：抓取指定帖子的回覆內容，支援多頁迭代。
- get_category_name：返回分類名稱。
- get_lihkg_thread_content_batch：批量抓取多個帖子的回覆內容。
- verify_lihkg_thread_category：驗證指定帖子的分類並記錄詳細日誌。
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
from logging_config import configure_logger

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = configure_logger(__name__, "lihkg_api.log")

# LIHKG API 基礎配置
LIHKG_BASE_URL = "https://lihkg.com"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=33mda4rq2c68ka0q69j3qli704; __cfruid=6613e31ceda225cd19b78296f41fcdd792896165-1746435095; cf_clearance=lKvL5WE.wiQ3OPQ1vBByvADiC7YQyySKHwbpY.OIMDM-1746437382-1.2.1.1-yvhsMxUOu2oIxxQWYDQ8RLr6DSql1PiOJoewKLjnP4cUk1kpehYNcPdBMd5db78MFblUmscp9x3IZcZF6GLrq6iQogh7XT6_KrQv4mv_yRfDvUzHv8VTTtDB_3NJjSn6hkj5rZ68etejlTWzje51WQRHzaNNzUEzwSgyq5B1Orhc75wvzbibCJYtQm7kZwDbOyvT3ZI8EpVTckCdutk_F8tQ7iEIuXaclhuWrMa1h0wgJjXEeZGxvVHOfSAHiC837fxwz2m.uQ5vAe8nsaU8dY5W_8P1OCDpcHChWhWoK5JeYaXHq7Zsb8yfv9TceR7PaCU0XDP2KU_m_44b4uX.vGqZWa1qumQttbxUBciyQu4"

# 用戶代理列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]

class RateLimiter:
    """
    速率限制器，控制 API 請求頻率，動態調整延遲。
    """
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
        # 重置計數器
        if now - self.last_reset >= self.period:
            self.request_counter = 0
            self.last_reset = now
        # 檢查速率限制
        if now < self.rate_limit_until:
            wait_time = self.rate_limit_until - now
            logger.warning(f"Rate limit active, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            return False, {
                "request_counter": self.request_counter,
                "last_reset": self.last_reset,
                "rate_limit_until": self.rate_limit_until
            }
        # 清理過期請求
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
                        logger.debug(f"API response time: {response_time:.2f} seconds for {function_name}")
                        status = "success" if response.status == 200 else f"failed_status_{response.status}"
                        logger.debug(
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
                            logger.warning(f"Rate limit hit for {function_name}, url={url}")
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

async def get_lihkg_topic_list(cat_id, start_page=1, max_pages=3, target_items=20):
    """
    抓取指定分類的帖子標題列表，動態調整頁數以達到目標貼文數量。
    Args:
        cat_id: 分類 ID
        start_page: 起始頁數
        max_pages: 最大頁數（可動態增加）
        target_items: 目標貼文數量
    Returns:
        Dict: 包含貼文列表、速率限制信息等
    """
    items = []
    rate_limit_info = []
    current_page = start_page
    max_pages = max(max_pages, 10)  # 最多抓取 10 頁，避免過多請求
    attempts = 0

    while len(items) < target_items and current_page < start_page + max_pages and attempts < 2:
        if cat_id == "2":  # 熱門台使用專用端點
            url = f"{LIHKG_BASE_URL}/api_v2/thread/hot?cat_id={cat_id}&page={current_page}&count=60&type=now"
        else:
            url = f"{LIHKG_BASE_URL}/api_v2/thread/category?cat_id={cat_id}&page={current_page}&count=60&type=now"
        data, page_rate_limit_info, rate_limit_data = await api_client.get(url, "get_lihkg_topic_list")
        rate_limit_info.extend(page_rate_limit_info)
        
        raw_items = []
        if data and data.get("response", {}).get("items"):
            raw_items = [
                item for item in data["response"]["items"]
                if item.get("title") and item.get("no_of_reply", 0) > 0
            ]
            # 驗證貼文分類
            filtered_items = []
            for item in raw_items:
                thread_id = str(item.get("thread_id"))
                verification = await verify_lihkg_thread_category(thread_id, selected_cat_id=cat_id)
                item_cat_id = verification.get("cat_id") if "cat_id" in verification else None
                if item_cat_id == cat_id or (cat_id in ["1", "2"] and item_cat_id):
                    filtered_items.append(item)
                # 若需為熱門台收緊過濾，使用以下條件（取消註釋即可）：
                # if cat_id == "2" and item_cat_id == cat_id:
                #     filtered_items.append(item)
                # elif cat_id != "2" and (item_cat_id == cat_id or (cat_id == "1" and item_cat_id)):
                #     filtered_items.append(item)
            
            items.extend(filtered_items[:target_items - len(items)])
            logger.info(
                json.dumps({
                    "event": "lihkg_topic_fetch",
                    "cat_id": cat_id,
                    "page": current_page,
                    "raw_items": len(raw_items),
                    "filtered_items": len(filtered_items),
                    "item_cat_ids": [item.get("cat_id") for item in raw_items],
                    "verified_cat_ids": [verification.get("cat_id") for item in filtered_items]
                }, ensure_ascii=False)
            )
            attempts = 0  # 重置重試計數
        else:
            logger.error(
                json.dumps({
                    "event": "lihkg_topic_fetch",
                    "cat_id": cat_id,
                    "page": current_page,
                    "error": "No data fetched",
                    "api_response": data.get("error_message", "Unknown") if data else "No response"
                }, ensure_ascii=False)
            )
            attempts += 1  # 記錄連續失敗次數
        
        current_page += 1
        await asyncio.sleep(1)
    
    logger.info(
        json.dumps({
            "event": "lihkg_topic_fetch_summary",
            "cat_id": cat_id,
            "total_items": len(items),
            "pages_fetched": current_page - start_page,
            "target_items": target_items
        }, ensure_ascii=False)
    )
    
    return {
        "items": items[:target_items],
        "rate_limit_info": rate_limit_info,
        "request_counter": rate_limit_data["request_counter"],
        "last_reset": rate_limit_data["last_reset"],
        "rate_limit_until": rate_limit_data["rate_limit_until"]
    }

async def get_lihkg_thread_content(thread_id, cat_id=None, max_replies=250, fetch_last_pages=0, specific_pages=None, start_page=1):
    """
    抓取指定帖子的回覆內容，支援多頁迭代直到達到 max_replies 或無更多回覆。
    Args:
        thread_id: 帖子 ID
        cat_id: 分類 ID
        max_replies: 最大回覆數
        fetch_last_pages: 抓取最後幾頁（優先於 specific_pages）
        specific_pages: 指定頁數列表
        start_page: 起始頁數
    Returns:
        Dict: 包含回覆數據、帖子標題、總回覆數、總頁數等
    """
    replies = []
    fetched_pages = []
    thread_title = None
    total_replies = None
    total_pages = None
    rate_limit_info = []
    max_replies = max(max_replies, 25)  # 確保至少抓取 25 條
    
    # 抓取第一頁以獲取帖子元數據
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
        logger.debug(f"Fetched thread_id={thread_id}, page=1, replies={len(page_replies)}, total_stored={len(replies)}, expected_replies={max_replies}")
    else:
        logger.error(f"No data fetched for thread_id={thread_id}, page=1")
        return {
            "replies": [], "title": None, "total_replies": 0, "total_pages": 0,
            "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
            "request_counter": rate_limit_data["request_counter"],
            "last_reset": rate_limit_data["last_reset"],
            "rate_limit_until": rate_limit_data["rate_limit_until"]
        }
    
    # 確定後續頁面
    pages_to_fetch = []
    if specific_pages:
        pages_to_fetch = [p for p in specific_pages if 1 <= p <= total_pages and p not in fetched_pages]
    elif fetch_last_pages > 0:
        start = max(start_page, 2)
        end = min(total_pages + 1, start + fetch_last_pages)
        pages_to_fetch = list(range(start, end))
        pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages]
    else:
        max_pages = (max_replies // 25) + 1  # 每頁最多 25 條
        pages_to_fetch = list(range(2, min(max_pages + 1, total_pages + 1)))
        pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages]
    
    pages_to_fetch = sorted(set(pages_to_fetch))
    
    # 抓取後續頁面
    for page in pages_to_fetch:
        if len(replies) >= max_replies:
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
            logger.debug(f"Fetched thread_id={thread_id}, page={page}, replies={len(page_replies)}, total_stored={len(replies)}, expected_replies={max_replies}")
        else:
            logger.warning(f"No data fetched for thread_id={thread_id}, page={page}")
        
        await asyncio.sleep(0.5)  # 避免過快請求
    
    # 統一記錄抓取總結日誌
    status_message = f"Completed fetching thread_id={thread_id}, pages={fetched_pages}, total_replies={len(replies)}"
    if len(replies) >= max_replies:
        status  status_message += f", reached max {max_replies}"
    elif len(replies) < max_replies:
        status_message += f", fewer than expected {max_replies}"
    logger.info(status_message)
    
    return {
        "replies": replies,
        "title": thread_title,
        "total_replies": total_replies,
        "total_pages": total_pages,
        "fetched_pages": fetched_pages,
        "rate_limit_info": rate_limit_info,
        "request_counter": rate_limit_data["request_counter"],
        "last_reset": rate_limit_data["last_reset"],
        "rate_limit_until": rate_limit_data["rate_limit_until"]
    }

async def get_lihkg_thread_content_batch(thread_ids, cat_id=None, max_replies=250, fetch_last_pages=0, specific_pages=None, start_page=1):
    """
    批量抓取多個帖子的回覆內容，減少 API 請求次數。
    Args:
        thread_ids: List[str]，帖子 ID 列表
        cat_id: str，分類 ID
        max_replies: int，每個帖子的最大回覆數
        fetch_last_pages: int，抓取最後幾頁
        specific_pages: List[int]，指定頁數
        start_page: int，起始頁數
    Returns:
        Dict: 包含所有帖子的回覆數據
    """
    results = []
    rate_limit_info = []
    max_replies = max(max_replies, 25)  # 確保至少抓取 25 條
    
    tasks = []
    for thread_id in thread_ids:
        tasks.append(get_lihkg_thread_content(
            thread_id=thread_id,
            cat_id=cat_id,
            max_replies=max_replies,
            fetch_last_pages=fetch_last_pages,
            specific_pages=specific_pages,
            start_page=start_page
        ))
    
    content_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for idx, result in enumerate(content_results):
        thread_id = thread_ids[idx]
        if isinstance(result, Exception):
            logger.warning(f"Failed to fetch content for thread {thread_id}: {str(result)}")
            results.append({
                "thread_id": thread_id,
                "replies": [],
                "title": None,
                "total_replies": 0,
                "total_pages": 0,
                "fetched_pages": [],
                "rate_limit_info": [{"message": f"Fetch error: {str(result)}"}],
                "request_counter": 0,
                "last_reset": time.time(),
                "rate_limit_until": 0
            })
            continue
        
        result["thread_id"] = thread_id
        rate_limit_info.extend(result.get("rate_limit_info", []))
        results.append(result)
    
    aggregated_rate_limit_data = {
        "request_counter": max([r.get("request_counter", 0) for r in results]),
        "last_reset": min([r.get("last_reset", time.time()) for r in results]),
        "rate_limit_until": max([r.get("rate_limit_until", 0) for r in results])
    }
    
    return {
        "results": results,
        "rate_limit_info": rate_limit_info,
        "request_counter": aggregated_rate_limit_data["request_counter"],
        "last_reset": aggregated_rate_limit_data["last_reset"],
        "rate_limit_until": aggregated_rate_limit_data["rate_limit_until"]
    }

async def verify_lihkg_thread_category(thread_id, selected_cat_id=None):
    """
    驗證指定帖子的分類，記錄詳細日誌以便後台分析。
    Args:
        thread_id: 帖子 ID
        selected_cat_id: 用戶選擇的分類 ID（可選）
    Returns:
        Dict: 包含 thread_id 和找到的分類信息
    """
    # 初始化本地緩存
    local_cache = getattr(verify_lihkg_thread_category, "local_cache", {})
    verify_lihkg_thread_category.local_cache = local_cache

    # 檢查緩存
    cache_key = f"thread_{thread_id}_category"
    try:
        import streamlit as st
        cache = st.session_state.thread_cache if hasattr(st, "session_state") else local_cache
    except ImportError:
        cache = local_cache
        logger.debug("Streamlit not available, using local cache for verification")

    if cache_key in cache:
        cached_data = cache[cache_key]
        if time.time() - cached_data["timestamp"] < 300:  # 緩存 5 分鐘
            logger.info(
                json.dumps({
                    "event": "lihkg_category_verification",
                    "thread_id": thread_id,
                    "step": "cache_hit",
                    "cache_key": cache_key,
                    "result": cached_data["data"]
                }, ensure_ascii=False)
            )
            return cached_data["data"]

    logger.info(
        json.dumps({
            "event": "lihkg_category_verification",
            "thread_id": thread_id,
            "step": "start",
            "selected_cat_id": selected_cat_id
        }, ensure_ascii=False)
    )

    # 步驟 1：查詢帖子元數據
    url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/page/1?order=reply_time"
    data, rate_limit_info, rate_limit_data = await api_client.get(url, "verify_thread_category")
    
    if data and data.get("response"):
        response_data = data.get("response", {})
        cat_id = response_data.get("cat_id") or response_data.get("thread", {}).get("cat_id")
        if cat_id:
            category_name = get_category_name(cat_id)
            result = {"thread_id": thread_id, "cat_id": cat_id, "category_name": category_name}
            cache[cache_key] = {"data": result, "timestamp": time.time()}
            logger.info(
                json.dumps({
                    "event": "lihkg_category_verification",
                    "thread_id": thread_id,
                    "step": "metadata_check",
                    "url": url,
                    "cat_id": cat_id,
                    "category_name": category_name,
                    "result": result
                }, ensure_ascii=False)
            )
            return result
        else:
            logger.warning(
                json.dumps({
                    "event": "lihkg_category_verification",
                    "thread_id": thread_id,
                    "step": "metadata_check",
                    "url": url,
                    "warning": "No cat_id found in metadata"
                }, ensure_ascii=False)
            )
    else:
        logger.error(
            json.dumps({
                "event": "lihkg_category_verification",
                "thread_id": thread_id,
                "step": "metadata_check",
                "url": url,
                "error": "No data fetched"
            }, ensure_ascii=False)
        )

    # 步驟 2：檢查用戶選擇的分類（如果提供）
    if selected_cat_id:
        result = await get_lihkg_topic_list(cat_id=selected_cat_id, start_page=1, max_pages=1)
        if any(str(item.get("thread_id")) == str(thread_id) for item in result.get("items", [])):
            category_name = get_category_name(selected_cat_id)
            result = {"thread_id": thread_id, "cat_id": selected_cat_id, "category_name": category_name}
            cache[cache_key] = {"data": result, "timestamp": time.time()}
            logger.info(
                json.dumps({
                    "event": "lihkg_category_verification",
                    "thread_id": thread_id,
                    "step": "selected_category_check",
                    "cat_id": selected_cat_id,
                    "category_name": category_name,
                    "result": result
                }, ensure_ascii=False)
            )
            return result
        else:
            logger.warning(
                json.dumps({
                    "event": "lihkg_category_verification",
                    "thread_id": thread_id,
                    "step": "selected_category_check",
                    "cat_id": selected_cat_id,
                    "warning": f"Thread {thread_id} not found in selected category"
                }, ensure_ascii=False)
            )

    # 步驟 3：掃描其他分類
    categories = {
        "5": "時事台", "14": "上班台", "15": "財經台", "29": "成人台", "31": "創意台"
    }
    found_categories = []
    for cat_id, cat_name in categories.items():
        result = await get_lihkg_topic_list(cat_id=cat_id, start_page=1, max_pages=1)
        if any(str(item.get("thread_id")) == str(thread_id) for item in result.get("items", [])):
            found_categories.append({"cat_id": cat_id, "category_name": cat_name})
            logger.info(
                json.dumps({
                    "event": "lihkg_category_verification",
                    "thread_id": thread_id,
                    "step": "category_scan",
                    "cat_id": cat_id,
                    "category_name": cat_name,
                    "found": True
                }, ensure_ascii=False)
            )
        else:
            logger.debug(
                json.dumps({
                    "event": "lihkg_category_verification",
                    "thread_id": thread_id,
                    "step": "category_scan",
                    "cat_id": cat_id,
                    "category_name": cat_name,
                    "found": False
                }, ensure_ascii=False)
            )

    # 步驟 4：確定最終結果
    if found_categories:
        result = {"thread_id": thread_id, "categories": found_categories}
    else:
        result = {"thread_id": thread_id, "cat_id": "1", "category_name": "吹水台"}  # 默認為吹水台

    cache[cache_key] = {"data": result, "timestamp": time.time()}
    logger.info(
        json.dumps({
            "event": "lihkg_category_verification",
            "thread_id": thread_id,
            "step": "final_result",
            "result": result
        }, ensure_ascii=False)
    )
    return result
