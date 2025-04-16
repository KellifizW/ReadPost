import aiohttp
import streamlit.logger
import asyncio
import time
from datetime import datetime

logger = streamlit.logger.get_logger(__name__)

LIHKG_BASE_URL = "https://lihkg.com"

async def get_lihkg_topic_list(cat_id, sub_cat_id, start_page, max_pages):
    headers = {
        "User-Agent": "LIHKG/7.6.2 (iPhone; iOS 17.4.1; Scale/3.00)",
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9",
        "Connection": "keep-alive",
    }
    
    items = []
    rate_limit_info = []  # 用於記錄 429 錯誤的調試信息
    
    async with aiohttp.ClientSession() as session:
        for page in range(start_page, start_page + max_pages):
            url = f"{LIHKG_BASE_URL}/api_v2/thread/category?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={page}&count=60"
            headers["Referer"] = f"{LIHKG_BASE_URL}/category/{cat_id}"
            
            first_429_time = None
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 429:
                            attempt += 1
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            if first_429_time is None:
                                first_429_time = time.time()
                            
                            # 檢查 Retry-After 頭
                            retry_after = response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait_time = int(retry_after)
                                except ValueError:
                                    wait_time = 5 * attempt  # 如果 Retry-After 不是數字，使用指數退避
                            else:
                                wait_time = 5 * attempt  # 指數退避：第 1 次 5 秒，第 2 次 10 秒，第 3 次 15 秒
                            
                            rate_limit_info.append(f"{current_time} - 速率限制: cat_id={cat_id}, page={page}, 嘗試={attempt}, 狀態碼=429, 等待 {wait_time} 秒")
                            logger.warning(f"速率限制: cat_id={cat_id}, page={page}, 嘗試={attempt}, 狀態碼=429, 等待 {wait_time} 秒")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        if response.status != 200:
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}")
                            logger.error(f"抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}")
                            return {"items": items, "rate_limit_info": rate_limit_info}
                        
                        data = await response.json()
                        logger.debug(f"帖子列表 API 回應: cat_id={cat_id}, page={page}, 數據={data}")
                        if not data.get("success"):
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                            logger.error(f"API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                            return {"items": items, "rate_limit_info": rate_limit_info}
                        
                        new_items = data["response"]["items"]
                        if not new_items:
                            break
                        
                        if first_429_time:
                            total_wait_time = time.time() - first_429_time
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 重試成功: cat_id={cat_id}, page={page}, 總等待時間={total_wait_time:.2f} 秒")
                            logger.info(f"重試成功: cat_id={cat_id}, page={page}, 總等待時間={total_wait_time:.2f} 秒")
                        
                        items.extend(new_items)
                        logger.info(f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(new_items)}")
                        await asyncio.sleep(1)  # 每次請求後等待 1 秒
                        break
                
                except Exception as e:
                    rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}")
                    logger.error(f"抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}")
                    return {"items": items, "rate_limit_info": rate_limit_info}
            
            if attempt >= max_attempts:
                rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 重試失敗: cat_id={cat_id}, page={page}, 已達最大重試次數 {max_attempts}")
                logger.error(f"重試失敗: cat_id={cat_id}, page={page}, 已達最大重試次數 {max_attempts}")
                return {"items": items, "rate_limit_info": rate_limit_info}
    
    return {"items": items, "rate_limit_info": rate_limit_info}

async def get_lihkg_thread_content(thread_id, cat_id=None):
    headers = {
        "User-Agent": "LIHKG/7.6.2 (iPhone; iOS 17.4.1; Scale/3.00)",
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9",
        "Connection": "keep-alive",
    }
    
    replies = []
    page = 1
    thread_title = None
    total_replies = None
    rate_limit_info = []
    
    async with aiohttp.ClientSession() as session:
        while True:
            url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/message?page={page}&count=100"
            headers["Referer"] = f"{LIHKG_BASE_URL}/thread/{thread_id}"
            
            first_429_time = None
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 429:
                            attempt += 1
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            if first_429_time is None:
                                first_429_time = time.time()
                            
                            retry_after = response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait_time = int(retry_after)
                                except ValueError:
                                    wait_time = 5 * attempt
                            else:
                                wait_time = 5 * attempt
                            
                            rate_limit_info.append(f"{current_time} - 速率限制: thread_id={thread_id}, page={page}, 嘗試={attempt}, 狀態碼=429, 等待 {wait_time} 秒")
                            logger.warning(f"速率限制: thread_id={thread_id}, page={page}, 嘗試={attempt}, 狀態碼=429, 等待 {wait_time} 秒")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        if response.status != 200:
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取帖子內容失敗: thread_id={thread_id}, page={page}, 狀態碼={response.status}")
                            logger.error(f"抓取帖子內容失敗: thread_id={thread_id}, page={page}, 狀態碼={response.status}")
                            return {"replies": replies, "title": thread_title, "total_replies": total_replies, "rate_limit_info": rate_limit_info}
                        
                        data = await response.json()
                        logger.debug(f"帖子內容 API 回應: thread_id={thread_id}, page={page}, 數據={data}")
                        if not data.get("success"):
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 返回失敗: thread_id={thread_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                            logger.error(f"API 返回失敗: thread_id={thread_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                            return {"replies": replies, "title": thread_title, "total_replies": total_replies, "rate_limit_info": rate_limit_info}
                        
                        if first_429_time:
                            total_wait_time = time.time() - first_429_time
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 重試成功: thread_id={thread_id}, page={page}, 總等待時間={total_wait_time:.2f} 秒")
                            logger.info(f"重試成功: thread_id={thread_id}, page={page}, 總等待時間={total_wait_time:.2f} 秒")
                        
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
                        logger.info(f"成功抓取帖子回覆: thread_id={thread_id}, page={page}, 回覆數={len(new_replies)}")
                        page += 1
                        await asyncio.sleep(1)  # 每次請求後等待 1 秒
                        break
                
                except Exception as e:
                    rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤={str(e)}")
                    logger.error(f"抓取帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤={str(e)}")
                    return {"replies": replies, "title": thread_title, "total_replies": total_replies, "rate_limit_info": rate_limit_info}
            
            if attempt >= max_attempts:
                rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 重試失敗: thread_id={thread_id}, page={page}, 已達最大重試次數 {max_attempts}")
                logger.error(f"重試失敗: thread_id={thread_id}, page={page}, 已達最大重試次數 {max_attempts}")
                return {"replies": replies, "title": thread_title, "total_replies": total_replies, "rate_limit_info": rate_limit_info}
            
            if total_replies and len(replies) >= total_replies:
                break
    
    return {"replies": replies, "title": thread_title, "total_replies": total_replies, "rate_limit_info": rate_limit_info}
