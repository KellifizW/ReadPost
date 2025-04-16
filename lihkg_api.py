import aiohttp
import asyncio
import time
import random
from datetime import datetime
import logging.handlers

# 設置日誌
try:
    import streamlit.logger
    logger = streamlit.logger.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    handler = logging.handlers.RotatingFileHandler("lihkg_api.log", maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

LIHKG_BASE_URL = "https://lihkg.com"

# 隨機 User-Agent 列表
USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

# 代理列表（需自行替換為實際代理地址）
PROXY_LIST = [
    # 示例代理，需替換為實際代理服務地址
    # "http://proxy1.example.com:8080",
    # "http://proxy2.example.com:8080",
]

async def get_lihkg_topic_list(cat_id, sub_cat_id, start_page, max_pages, request_counter, last_reset, rate_limit_until):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
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
    proxy = random.choice(PROXY_LIST) if PROXY_LIST else None
    
    # 檢查速率限制
    current_time = time.time()
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 速率限制中，請在 {datetime.fromtimestamp(rate_limit_until)} 後重試")
        logger.warning(f"API 速率限制中，需等待至 {datetime.fromtimestamp(rate_limit_until)}")
        return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
    
    async with aiohttp.ClientSession() as session:
        for page in range(start_page, start_page + max_pages):
            # 重置請求計數
            if current_time - last_reset >= 60:
                request_counter = 0
                last_reset = current_time
            
            request_counter += 1
            if request_counter > 90:  # 每分鐘 90 次上限
                wait_time = 60 - (current_time - last_reset)
                rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 即將達到每分鐘請求限制，暫停 {wait_time:.2f} 秒")
                logger.warning(f"即將達到每分鐘請求限制，暫停 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                last_reset = time.time()
            
            url = f"{LIHKG_BASE_URL}/api_v2/thread/category?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={page}&count=60"
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                try:
                    async with session.get(url, headers=headers, proxy=proxy) as response:
                        if response.status == 429:
                            attempt += 1
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            retry_after = response.headers.get("Retry-After")
                            wait_time = int(retry_after) if retry_after else 5 * attempt
                            rate_limit_until = time.time() + wait_time
                            rate_limit_info.append(
                                f"{current_time} - 速率限制: cat_id={cat_id}, page={page}, 嘗試={attempt}, "
                                f"狀態碼=429, 等待 {wait_time} 秒, 請求計數={request_counter}, "
                                f"最後重置={datetime.fromtimestamp(last_reset)}, 代理={proxy}"
                            )
                            logger.warning(
                                f"速率限制: cat_id={cat_id}, page={page}, 嘗試={attempt}, 狀態碼=429, "
                                f"等待 {wait_time} 秒, 代理={proxy}"
                            )
                            await asyncio.sleep(wait_time)
                            proxy = random.choice(PROXY_LIST) if PROXY_LIST else None  # 更換代理
                            continue
                        
                        if response.status != 200:
                            rate_limit_info.append(
                                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取失敗: "
                                f"cat_id={cat_id}, page={page}, 狀態碼={response.status}"
                            )
                            logger.error(f"抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}")
                            return {
                                "items": items,
                                "rate_limit_info": rate_limit_info,
                                "request_counter": request_counter,
                                "last_reset": last_reset,
                                "rate_limit_until": rate_limit_until
                            }
                        
                        data = await response.json()
                        logger.debug(f"帖子列表 API 回應: cat_id={cat_id}, page={page}, 數據={data}")
                        if not data.get("success"):
                            rate_limit_info.append(
                                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 返回失敗: "
                                f"cat_id={cat_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}"
                            )
                            logger.error(
                                f"API 返回失敗: cat_id={cat_id}, page={page}, "
                                f"錯誤={data.get('error_message', '未知錯誤')}"
                            )
                            return {
                                "items": items,
                                "rate_limit_info": rate_limit_info,
                                "request_counter": request_counter,
                                "last_reset": last_reset,
                                "rate_limit_until": rate_limit_until
                            }
                        
                        new_items = data["response"]["items"]
                        if not new_items:
                            break
                        
                        items.extend(new_items)
                        logger.info(f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(new_items)}, 代理={proxy}")
                        await asyncio.sleep(random.uniform(2, 10))  # 隨機等待 2-10 秒
                        break
                
                except Exception as e:
                    rate_limit_info.append(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取錯誤: "
                        f"cat_id={cat_id}, page={page}, 錯誤={str(e)}, 代理={proxy}"
                    )
                    logger.error(f"抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}, 代理={proxy}")
                    return {
                        "items": items,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until
                    }
            
            if attempt >= max_attempts:
                rate_limit_info.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 重試失敗: "
                    f"cat_id={cat_id}, page={page}, 已達最大重試次數 {max_attempts}, 代理={proxy}"
                )
                logger.error(
                    f"重試失敗: cat_id={cat_id}, page={page}, 已達最大重試次數 {max_attempts}, 代理={proxy}"
                )
                return {
                    "items": items,
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": rate_limit_until
                }
    
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def get_lihkg_thread_content(thread_id, cat_id=None, request_counter=0, last_reset=0, rate_limit_until=0):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
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
    page = 1
    thread_title = None
    total_replies = None
    rate_limit_info = []
    proxy = random.choice(PROXY_LIST) if PROXY_LIST else None
    
    # 檢查速率限制
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
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    async with aiohttp.ClientSession() as session:
        while True:
            # 重置請求計數
            if current_time - last_reset >= 60:
                request_counter = 0
                last_reset = current_time
            
            request_counter += 1
            if request_counter > 90:
                wait_time = 60 - (current_time - last_reset)
                rate_limit_info.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 即將達到每分鐘請求限制，"
                    f"暫停 {wait_time:.2f} 秒"
                )
                logger.warning(f"即將達到每分鐘請求限制，暫停 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                last_reset = time.time()
            
            url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/message?page={page}&count=100"
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                try:
                    async with session.get(url, headers=headers, proxy=proxy) as response:
                        if response.status == 429:
                            attempt += 1
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            retry_after = response.headers.get("Retry-After")
                            wait_time = int(retry_after) if retry_after else 5 * attempt
                            rate_limit_until = time.time() + wait_time
                            rate_limit_info.append(
                                f"{current_time} - 速率限制: thread_id={thread_id}, page={page}, 嘗試={attempt}, "
                                f"狀態碼=429, 等待 {wait_time} 秒, 請求計數={request_counter}, "
                                f"最後重置={datetime.fromtimestamp(last_reset)}, 代理={proxy}"
                            )
                            logger.warning(
                                f"速率限制: thread_id={thread_id}, page={page}, 嘗試={attempt}, 狀態碼=429, "
                                f"等待 {wait_time} 秒, 代理={proxy}"
                            )
                            await asyncio.sleep(wait_time)
                            proxy = random.choice(PROXY_LIST) if PROXY_LIST else None  # 更換代理
                            continue
                        
                        if response.status != 200:
                            rate_limit_info.append(
                                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取帖子內容失敗: "
                                f"thread_id={thread_id}, page={page}, 狀態碼={response.status}"
                            )
                            logger.error(
                                f"抓取帖子內容失敗: thread_id={thread_id}, page={page}, 狀態碼={response.status}"
                            )
                            return {
                                "replies": replies,
                                "title": thread_title,
                                "total_replies": total_replies,
                                "rate_limit_info": rate_limit_info,
                                "request_counter": request_counter,
                                "last_reset": last_reset,
                                "rate_limit_until": rate_limit_until
                            }
                        
                        data = await response.json()
                        logger.debug(f"帖子內容 API 回應: thread_id={thread_id}, page={page}, 數據={data}")
                        if not data.get("success"):
                            rate_limit_info.append(
                                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 返回失敗: "
                                f"thread_id={thread_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}"
                            )
                            logger.error(
                                f"API 返回失敗: thread_id={thread_id}, page={page}, "
                                f"錯誤={data.get('error_message', '未知錯誤')}"
                            )
                            return {
                                "replies": replies,
                                "title": thread_title,
                                "total_replies": total_replies,
                                "rate_limit_info": rate_limit_info,
                                "request_counter": request_counter,
                                "last_reset": last_reset,
                                "rate_limit_until": rate_limit_until
                            }
                        
                        # 提取標題
                        if page == 1:
                            response_data = data.get("response", {})
                            thread_title = response_data.get("title") or response_data.get("thread", {}).get("title")
                            total_replies = response_data.get("total_replies") or response_data.get("total_reply", 0)
                        
                        # 提取回覆
                        new_replies = data["response"].get("items", [])
                        if not new_replies:
                            break
                        
                        replies.extend(new_replies)
                        logger.info(
                            f"成功抓取帖子回覆: thread_id={thread_id}, page={page}, "
                            f"回覆數={len(new_replies)}, 代理={proxy}"
                        )
                        page += 1
                        await asyncio.sleep(random.uniform(2, 10))  # 隨機等待 2-10 秒
                        break
                
                except Exception as e:
                    rate_limit_info.append(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取帖子內容錯誤: "
                        f"thread_id={thread_id}, page={page}, 錯誤={str(e)}, 代理={proxy}"
                    )
                    logger.error(
                        f"抓取帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤={str(e)}, 代理={proxy}"
                    )
                    return {
                        "replies": replies,
                        "title": thread_title,
                        "total_replies": total_replies,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until
                    }
            
            if attempt >= max_attempts:
                rate_limit_info.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 重試失敗: "
                    f"thread_id={thread_id}, page={page}, 已達最大重試次數 {max_attempts}, 代理={proxy}"
                )
                logger.error(
                    f"重試失敗: thread_id={thread_id}, page={page}, "
                    f"已達最大重試次數 {max_attempts}, 代理={proxy}"
                )
                return {
                    "replies": replies,
                    "title": thread_title,
                    "total_replies": total_replies,
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": rate_limit_until
                }
            
            if total_replies and len(replies) >= total_replies:
                break
    
    return {
        "replies": replies,
        "title": thread_title,
        "total_replies": total_replies,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }
