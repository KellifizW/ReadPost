import aiohttp
import asyncio
import time
from datetime import datetime
import random
import hashlib
import streamlit.logger

logger = streamlit.logger.get_logger(__name__)

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
    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.requests = []

    async def acquire(self, context: dict = None):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.period]
        if len(self.requests) >= self.max_requests:
            wait_time = self.period - (now - self.requests[0])
            context_info = f", 上下文={context}" if context else ""
            logger.warning(f"達到內部速率限制，當前請求數={len(self.requests)}/{self.max_requests}，等待 {wait_time:.2f} 秒{context_info}")
            await asyncio.sleep(wait_time)
            self.requests = self.requests[1:]
        self.requests.append(now)

rate_limiter = RateLimiter(max_requests=20, period=60)

def get_category_name(cat_id):
    """根據 cat_id 返回分類名稱"""
    categories = {
        "1": "吹水台",
        "2": "熱門台",
        "5": "時事台",
        "14": "上班台",
        "15": "財經台",
        "29": "成人台",
        "31": "創意台"
    }
    return categories.get(str(cat_id), "未知分類")

async def get_lihkg_topic_list(cat_id, sub_cat_id, start_page=1, max_pages=3, request_counter=0, last_reset=0, rate_limit_until=0):
    """抓取指定分類的多頁帖子標題（30-90 個）"""
    timestamp = int(time.time())
    url_template = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id={cat_id}&page={{page}}&count=60&type=now&order=now"
    
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-REQUEST-TIME": str(timestamp),
        "Cookie": LIHKG_COOKIE,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"{LIHKG_BASE_URL}/category/{cat_id}",
        "DNT": "1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    
    items = []
    rate_limit_info = []
    max_retries = 3
    
    current_time = time.time()
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 速率限制中，請在 {datetime.fromtimestamp(rate_limit_until)} 後重試")
        logger.warning(f"API 速率限制中，需等待至 {datetime.fromtimestamp(rate_limit_until)}")
        return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
    
    async with aiohttp.ClientSession() as session:
        for page in range(start_page, start_page + max_pages):
            if current_time - last_reset >= 60:
                request_counter = 0
                last_reset = current_time
            
            url = url_template.format(page=page)
            digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
            headers["X-LI-DIGEST"] = digest
            headers["X-LI-REQUEST-TIME"] = str(timestamp)
            
            fetch_conditions = {
                "cat_id": cat_id,
                "sub_cat_id": sub_cat_id,
                "page": page,
                "max_pages": max_pages,
                "user_agent": headers["User-Agent"],
                "device_id": LIHKG_DEVICE_ID,
                "request_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "request_counter": request_counter,
                "last_reset": datetime.fromtimestamp(last_reset).strftime("%Y-%m-%d %H:%M:%S"),
                "rate_limit_until": datetime.fromtimestamp(rate_limit_until).strftime("%Y-%m-%d %H:%M:%S") if rate_limit_until > time.time() else "無"
            }
            
            for attempt in range(max_retries):
                try:
                    await rate_limiter.acquire(context=fetch_conditions)
                    request_counter += 1
                    async with session.get(url, headers=headers, timeout=10) as response:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After", "5")
                            headers_info = dict(response.headers)
                            wait_time = int(retry_after) if retry_after.isdigit() else 5
                            wait_time = min(wait_time * (2 ** attempt), 60) + random.uniform(0, 0.1)
                            rate_limit_until = time.time() + wait_time
                            rate_limit_info.append(
                                f"{current_time} - 伺服器速率限制: cat_id={cat_id}, page={page}, "
                                f"狀態碼=429, 第 {attempt+1} 次重試，等待 {wait_time:.2f} 秒"
                            )
                            logger.warning(
                                f"伺服器速率限制: cat_id={cat_id}, page={page}, 狀態碼=429, "
                                f"等待 {wait_time:.2f} 秒, Retry-After={retry_after}, 標頭={headers_info}, "
                                f"條件={fetch_conditions}"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        
                        if response.status != 200:
                            headers_info = dict(response.headers)
                            rate_limit_info.append(
                                f"{current_time} - 抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}"
                            )
                            logger.error(
                                f"抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}, "
                                f"標頭={headers_info}, 條件={fetch_conditions}"
                            )
                            await asyncio.sleep(1)
                            break
                        
                        data = await response.json()
                        logger.debug(f"帖子列表 API 回應: cat_id={cat_id}, page={page}, 數據={data}")
                        if not data.get("success"):
                            error_message = data.get("error_message", "未知錯誤")
                            rate_limit_info.append(
                                f"{current_time} - API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={error_message}"
                            )
                            logger.error(
                                f"API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={error_message}, "
                                f"條件={fetch_conditions}"
                            )
                            await asyncio.sleep(1)
                            break
                        
                        new_items = data["response"]["items"]
                        filtered_items = [
                            item for item in new_items
                            if item.get("title") and item.get("no_of_reply", 0) > 0
                        ]
                        if not filtered_items:
                            break
                        
                        logger.info(
                            f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(filtered_items)}"
                        )
                        
                        if filtered_items:
                            first_item = filtered_items[0]
                            last_item = filtered_items[-1]
                            logger.info(
                                f"排序檢查: cat_id={cat_id}, page={page}, "
                                f"首帖 thread_id={first_item['thread_id']}, last_reply_time={first_item.get('last_reply_time', '未知')}, "
                                f"末帖 thread_id={last_item['thread_id']}, last_reply_time={last_item.get('last_reply_time', '未知')}"
                            )
                        
                        items.extend(filtered_items)
                        break
                    
                except Exception as e:
                    rate_limit_info.append(
                        f"{current_time} - 抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}"
                    )
                    logger.error(
                        f"抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}, 條件={fetch_conditions}"
                    )
                    await asyncio.sleep(1)
                    break
            
            await asyncio.sleep(5)
            current_time = time.time()
    
    return {
        "items": items[:90],  # 限制最大 90 個標題
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def get_lihkg_thread_content(thread_id, cat_id=None, request_counter=0, last_reset=0, rate_limit_until=0, max_replies=150, fetch_last_pages=0, specific_pages=None, start_page=1):
    """抓取帖子回覆，支持首頁、末頁、指定頁面或從指定起始頁開始"""
    timestamp = int(time.time())
    url_template = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/page/{{page}}?order=reply_time"
    
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-REQUEST-TIME": str(timestamp),
        "Cookie": LIHKG_COOKIE,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"{LIHKG_BASE_URL}/thread/{thread_id}",
        "DNT": "1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    
    replies = []
    fetched_pages = []
    thread_title = None
    total_replies = None
    total_pages = None
    rate_limit_info = []
    max_retries = 3
    per_page = 25  # 每頁最大回覆數
    
    current_time = time.time()
    if current_time < rate_limit_until:
        rate_limit_info.append(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 速率限制中，"
            f"請在 {datetime.fromtimestamp(rate_limit_until)} 後重試"
        )
        logger.warning(f"API 速率限制中，需等待至 {datetime.fromtimestamp(rate_limit_until)}")
        return {
            "replies": replies,
            "title": thread_title,
            "total_replies": total_replies,
            "total_pages": total_pages,
            "fetched_pages": fetched_pages,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    async with aiohttp.ClientSession() as session:
        # 先抓取第一頁以獲取總頁數
        url = url_template.format(page=1)
        digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
        headers["X-LI-DIGEST"] = digest
        headers["X-LI-REQUEST-TIME"] = str(timestamp)
        
        fetch_conditions = {
            "thread_id": thread_id,
            "cat_id": cat_id if cat_id else "無",
            "page": 1,
            "user_agent": headers["User-Agent"],
            "device_id": LIHKG_DEVICE_ID,
            "request_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "request_counter": request_counter,
            "last_reset": datetime.fromtimestamp(last_reset).strftime("%Y-%m-%d %H:%M:%S"),
            "rate_limit_until": datetime.fromtimestamp(rate_limit_until).strftime("%Y-%m-%d %H:%M:%S") if rate_limit_until > time.time() else "無"
        }
        
        try:
            await rate_limiter.acquire(context=fetch_conditions)
            request_counter += 1
            async with session.get(url, headers=headers, timeout=10) as response:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After", "5")
                    wait_time = int(retry_after) if retry_after.isdigit() else 5
                    rate_limit_until = time.time() + wait_time
                    rate_limit_info.append(
                        f"{current_time} - 伺服器速率限制: thread_id={thread_id}, page=1, 狀態碼=429, 等待 {wait_time:.2f} 秒"
                    )
                    logger.warning(f"伺服器速率限制: thread_id={thread_id}, page=1, 狀態碼=429, 等待 {wait_time:.2f} 秒")
                    return {
                        "replies": [],
                        "title": None,
                        "total_replies": 0,
                        "total_pages": 0,
                        "fetched_pages": fetched_pages,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until
                    }
                
                if response.status != 200:
                    rate_limit_info.append(
                        f"{current_time} - 抓取帖子內容失敗: thread_id={thread_id}, page=1, 狀態碼={response.status}"
                    )
                    logger.error(f"抓取帖子內容失敗: thread_id={thread_id}, page=1, 狀態碼={response.status}")
                    return {
                        "replies": [],
                        "title": None,
                        "total_replies": 0,
                        "total_pages": 0,
                        "fetched_pages": fetched_pages,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until
                    }
                
                data = await response.json()
                if not data.get("success"):
                    error_message = data.get("error_message", "未知錯誤")
                    rate_limit_info.append(
                        f"{current_time} - API 返回失敗: thread_id={thread_id}, page=1, 錯誤={error_message}"
                    )
                    logger.warning(f"API 返回失敗: thread_id={thread_id}, page=1, 錯誤={error_message}")
                    return {
                        "replies": [],
                        "title": None,
                        "total_replies": 0,
                        "total_pages": 0,
                        "fetched_pages": fetched_pages,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until
                    }
                
                response_data = data.get("response", {})
                thread_title = response_data.get("title") or response_data.get("thread", {}).get("title")
                total_replies = response_data.get("total_replies") or response_data.get("total_reply", 0)
                total_pages = response_data.get("total_page", 1)
                
                page_replies = response_data.get("item_data", [])
                for reply in page_replies:
                    reply["like_count"] = int(reply.get("like_count", "0")) if reply.get("like_count") else 0
                    reply["dislike_count"] = int(reply.get("dislike_count", "0")) if reply.get("dislike_count") else 0
                    reply["reply_time"] = reply.get("reply_time", "0")
                replies.extend(page_replies[:max_replies - len(replies)])
                fetched_pages.append(1)
                logger.info(f"成功抓取帖子回覆: thread_id={thread_id}, page=1, 回覆數={len(page_replies)}")
        
        except Exception as e:
            rate_limit_info.append(f"{current_time} - 抓取帖子內容錯誤: thread_id={thread_id}, page=1, 錯誤={str(e)}")
            logger.error(f"抓取帖子內容錯誤: thread_id={thread_id}, page=1, 錯誤={str(e)}")
            return {
                "replies": [],
                "title": None,
                "total_replies": 0,
                "total_pages": 0,
                "fetched_pages": fetched_pages,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until
            }
        
        # 確定抓取頁數
        pages_to_fetch = []
        if specific_pages:
            # 優先使用 specific_pages
            pages_to_fetch = [p for p in specific_pages if 1 <= p <= total_pages and p not in fetched_pages]
        elif fetch_last_pages > 0:
            # 抓取末尾頁，從 start_page 開始
            start = max(start_page, 2)  # 確保不重複抓取第一頁，除非 start_page=1
            end = total_pages + 1
            if end - start < fetch_last_pages:
                start = max(2, end - fetch_last_pages)
            pages_to_fetch = list(range(start, end))[:fetch_last_pages]
            pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages and 1 <= p <= total_pages]
        elif start_page > 1:
            # 從 start_page 開始，抓取直到 max_replies 或結束
            pages_to_fetch = list(range(start_page, total_pages + 1))
            pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages]
        else:
            # 僅抓取首頁（默認行為）
            pages_to_fetch = []
        
        # 去重並排序
        pages_to_fetch = sorted(set(pages_to_fetch))
        
        # 抓取剩餘頁數
        for page in pages_to_fetch:
            if len(replies) >= max_replies:
                logger.info(f"終止抓取: thread_id={thread_id}, 回覆數={len(replies)} 已達最大值 {max_replies}")
                break
            
            url = url_template.format(page=page)
            digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
            headers["X-LI-DIGEST"] = digest
            headers["X-LI-REQUEST-TIME"] = str(timestamp)
            
            fetch_conditions["page"] = page
            
            for attempt in range(max_retries):
                try:
                    await rate_limiter.acquire(context=fetch_conditions)
                    request_counter += 1
                    async with session.get(url, headers=headers, timeout=10) as response:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After", "5")
                            headers_info = dict(response.headers)
                            wait_time = int(retry_after) if retry_after.isdigit() else 5
                            wait_time = min(wait_time * (2 ** attempt), 60) + random.uniform(0, 0.1)
                            rate_limit_until = time.time() + wait_time
                            rate_limit_info.append(
                                f"{current_time} - 伺服器速率限制: thread_id={thread_id}, page={page}, "
                                f"狀態碼=429, 第 {attempt+1} 次重試，等待 {wait_time:.2f} 秒"
                            )
                            logger.warning(
                                f"伺服器速率限制: thread_id={thread_id}, page={page}, 狀態碼=429, "
                                f"等待 {wait_time:.2f} 秒, Retry-After={retry_after}, 標頭={headers_info}, "
                                f"條件={fetch_conditions}"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        
                        if response.status != 200:
                            headers_info = dict(response.headers)
                            rate_limit_info.append(
                                f"{current_time} - 抓取帖子內容失敗: thread_id={thread_id}, page={page}, 狀態碼={response.status}"
                            )
                            logger.error(
                                f"抓取帖子內容失敗: thread_id={thread_id}, page={page}, 狀態碼={response.status}, "
                                f"標頭={headers_info}, 條件={fetch_conditions}"
                            )
                            await asyncio.sleep(1)
                            break
                        
                        data = await response.json()
                        logger.debug(f"帖子內容 API 回應: thread_id={thread_id}, page={page}, 數據={data}")
                        if not data.get("success"):
                            error_message = data.get("error_message", "未知錯誤")
                            rate_limit_info.append(
                                f"{current_time} - API 返回失敗: thread_id={thread_id}, page={page}, 錯誤={error_message}"
                            )
                            logger.warning(
                                f"API 返回失敗: thread_id={thread_id}, page={page}, 錯誤={error_message}, "
                                f"條件={fetch_conditions}"
                            )
                            break
                        
                        page_replies = data["response"].get("item_data", [])
                        if not page_replies:
                            logger.info(
                                f"成功抓取帖子回覆: thread_id={thread_id}, page={page}, 回覆數=0"
                            )
                            break
                        
                        for reply in page_replies:
                            reply["like_count"] = int(reply.get("like_count", "0")) if reply.get("like_count") else 0
                            reply["dislike_count"] = int(reply.get("dislike_count", "0")) if reply.get("dislike_count") else 0
                            reply["reply_time"] = reply.get("reply_time", "0")
                        
                        # 限制回覆數
                        remaining_slots = max_replies - len(replies)
                        page_replies = page_replies[:remaining_slots]
                        replies.extend(page_replies)
                        fetched_pages.append(page)
                        
                        logger.info(
                            f"成功抓取帖子回覆: thread_id={thread_id}, page={page}, 回覆數={len(page_replies)}"
                        )
                        
                        if len(page_replies) < per_page:
                            logger.info(
                                f"終止抓取: thread_id={thread_id}, page={page}, "
                                f"回覆數={len(page_replies)} 小於每頁最大值 {per_page}"
                            )
                            break
                        
                        break
                    
                except Exception as e:
                    rate_limit_info.append(
                        f"{current_time} - 抓取帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤={str(e)}"
                    )
                    logger.error(
                        f"抓取帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤={str(e)}, "
                        f"條件={fetch_conditions}"
                    )
                    await asyncio.sleep(1)
                    break
            
            await asyncio.sleep(5)
            current_time = time.time()
    
    return {
        "replies": replies,
        "title": thread_title,
        "total_replies": total_replies,
        "total_pages": total_pages,
        "fetched_pages": fetched_pages,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }
