import aiohttp
import asyncio
import time
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

# 固定 User-Agent
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

async def get_lihkg_topic_list(cat_id, sub_cat_id, start_page, max_pages, request_counter, last_reset, rate_limit_until):
    headers = {
        "User-Agent": USER_AGENT,
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
            if request_counter > 30:  # 每分鐘 30 次上限
                wait_time = 60 - (current_time - last_reset)
                rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 即將達到每分鐘請求限制，暫停 {wait_time:.2f} 秒")
                logger.warning(f"即將達到每分鐘請求限制，暫停 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                last_reset = time.time()
            
            url = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id={cat_id}&page={page}&count=60&type=now&order=now"
            
            # 記錄抓取條件
            fetch_conditions = {
                "cat_id": cat_id,
                "sub_cat_id": sub_cat_id,
                "page": page,
                "max_pages": max_pages,
                "user_agent": headers["User-Agent"],
                "request_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "request_counter": request_counter,
                "last_reset": datetime.fromtimestamp(last_reset).strftime("%Y-%m-%d %H:%M:%S"),
                "rate_limit_until": datetime.fromtimestamp(rate_limit_until).strftime("%Y-%m-%d %H:%M:%S") if rate_limit_until > time.time() else "無"
            }
            
            try:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 429:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        retry_after = response.headers.get("Retry-After")
                        wait_time = int(retry_after) if retry_after else 5
                        rate_limit_until = time.time() + wait_time
                        rate_limit_info.append(
                            f"{current_time} - 速率限制: cat_id={cat_id}, page={page}, "
                            f"狀態碼=429, 等待 {wait_time} 秒, 請求計數={request_counter}, "
                            f"最後重置={datetime.fromtimestamp(last_reset)}"
                        )
                        logger.warning(
                            f"速率限制: cat_id={cat_id}, page={page}, 狀態碼=429, "
                            f"等待 {wait_time} 秒, 條件={fetch_conditions}"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status != 200:
                        rate_limit_info.append(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取失敗: "
                            f"cat_id={cat_id}, page={page}, 狀態碼={response.status}"
                        )
                        logger.error(
                            f"抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}, 條件={fetch_conditions}"
                        )
                        await asyncio.sleep(1)
                        continue
                    
                    data = await response.json()
                    logger.debug(f"帖子列表 API 回應: cat_id={cat_id}, page={page}, 數據={data}")
                    if not data.get("success"):
                        error_message = data.get("error_message", "未知錯誤")
                        rate_limit_info.append(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 返回失敗: "
                            f"cat_id={cat_id}, page={page}, 錯誤={error_message}"
                        )
                        logger.error(
                            f"API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={error_message}, 條件={fetch_conditions}"
                        )
                        await asyncio.sleep(1)
                        continue
                    
                    new_items = data["response"]["items"]
                    if not new_items:
                        break
                    
                    logger.info(
                        f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(new_items)}, "
                        f"條件={fetch_conditions}"
                    )
                    
                    if new_items:
                        first_item = new_items[0]
                        last_item = new_items[-1]
                        logger.info(
                            f"排序檢查: cat_id={cat_id}, page={page}, "
                            f"首帖 thread_id={first_item['thread_id']}, last_reply_time={first_item.get('last_reply_time', '未知')}, "
                            f"末帖 thread_id={last_item['thread_id']}, last_reply_time={last_item.get('last_reply_time', '未知')}"
                        )
                    
                    items.extend(new_items)
                    await asyncio.sleep(1)  # 固定 1 秒延遲
                    break
                
            except Exception as e:
                rate_limit_info.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取錯誤: "
                    f"cat_id={cat_id}, page={page}, 錯誤={str(e)}"
                )
                logger.error(
                    f"抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}, 條件={fetch_conditions}"
                )
                await asyncio.sleep(1)
                continue
    
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def get_lihkg_thread_content(thread_id, cat_id=None, request_counter=0, last_reset=0, rate_limit_until=0):
    headers = {
        "User-Agent": USER_AGENT,
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
            if current_time - last_reset >= 60:
                request_counter = 0
                last_reset = current_time
            
            request_counter += 1
            if request_counter > 30:
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
            
            fetch_conditions = {
                "thread_id": thread_id,
                "cat_id": cat_id if cat_id else "無",
                "page": page,
                "user_agent": headers["User-Agent"],
                "request_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "request_counter": request_counter,
                "last_reset": datetime.fromtimestamp(last_reset).strftime("%Y-%m-%d %H:%M:%S"),
                "rate_limit_until": datetime.fromtimestamp(rate_limit_until).strftime("%Y-%m-%d %H:%M:%S") if rate_limit_until > time.time() else "無"
            }
            
            try:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 429:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        retry_after = response.headers.get("Retry-After")
                        wait_time = int(retry_after) if retry_after else 5
                        rate_limit_until = time.time() + wait_time
                        rate_limit_info.append(
                            f"{current_time} - 速率限制: thread_id={thread_id}, page={page}, "
                            f"狀態碼=429, 等待 {wait_time} 秒, 請求計數={request_counter}, "
                            f"最後重置={datetime.fromtimestamp(last_reset)}"
                        )
                        logger.warning(
                            f"速率限制: thread_id={thread_id}, page={page}, 狀態碼=429, "
                            f"等待 {wait_time} 秒, 條件={fetch_conditions}"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status != 200:
                        rate_limit_info.append(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取帖子內容失敗: "
                            f"thread_id={thread_id}, page={page}, 狀態碼={response.status}"
                        )
                        logger.error(
                            f"抓取帖子內容失敗: thread_id={thread_id}, page={page}, 狀態碼={response.status}, "
                            f"條件={fetch_conditions}"
                        )
                        await asyncio.sleep(1)
                        continue
                    
                    data = await response.json()
                    logger.debug(f"帖子內容 API 回應: thread_id={thread_id}, page={page}, 數據={data}")
                    if not data.get("success"):
                        error_message = data.get("error_message", "未知錯誤")
                        if "998" in error_message:
                            rate_limit_info.append(
                                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 帖子無效或無權訪問: "
                                f"thread_id={thread_id}, page={page}, 錯誤={error_message}"
                            )
                            logger.warning(
                                f"帖子無效或無權訪問: thread_id={thread_id}, page={page}, 錯誤={error_message}, "
                                f"條件={fetch_conditions}"
                            )
                            return {
                                "replies": [],
                                "title": None,
                                "total_replies": 0,
                                "rate_limit_info": rate_limit_info,
                                "request_counter": request_counter,
                                "last_reset": last_reset,
                                "rate_limit_until": rate_limit_until
                            }
                        rate_limit_info.append(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 返回失敗: "
                            f"thread_id={thread_id}, page={page}, 錯誤={error_message}"
                        )
                        logger.error(
                            f"API 返回失敗: thread_id={thread_id}, page={page}, 錯誤={error_message}, 條件={fetch_conditions}"
                        )
                        await asyncio.sleep(1)
                        continue
                    
                    if page == 1:
                        response_data = data.get("response", {})
                        thread_title = response_data.get("title") or response_data.get("thread", {}).get("title")
                        total_replies = response_data.get("total_replies") or response_data.get("total_reply", 0)
                    
                    new_replies = data["response"].get("items", [])
                    if not new_replies:
                        break
                    
                    logger.info(
                        f"成功抓取帖子回覆: thread_id={thread_id}, page={page}, "
                        f"回覆數={len(new_replies)}, 條件={fetch_conditions}"
                    )
                    
                    replies.extend(new_replies)
                    page += 1
                    await asyncio.sleep(1)  # 固定 1 秒延遲
                    break
                
            except Exception as e:
                rate_limit_info.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取帖子內容錯誤: "
                    f"thread_id={thread_id}, page={page}, 錯誤={str(e)}"
                )
                logger.error(
                    f"抓取帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤={str(e)}, "
                    f"條件={fetch_conditions}"
                )
                await asyncio.sleep(1)
                continue
            
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
