import aiohttp
import asyncio
import time
import random
from datetime import datetime

USER_AGENTS = [
    # 原有 User-Agent 列表
]

PROXY_LIST = [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080",
    # 從代理服務獲取
]

async def get_lihkg_topic_list(cat_id, sub_cat_id, start_page, max_pages, request_counter, last_reset, rate_limit_until):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"{LIHKG_BASE_URL}/category/{cat_id}",
    }
    
    items = []
    rate_limit_info = []
    proxy = random.choice(PROXY_LIST) if PROXY_LIST else None
    
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
            
            request_counter += 1
            if request_counter > 90:
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
                            retry_after = response.headers.get("Retry-After")
                            wait_time = int(retry_after) if retry_after else 5 * attempt
                            rate_limit_until = time.time() + wait_time
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 速率限制: cat_id={cat_id}, page={page}, 嘗試={attempt}, 狀態碼=429, 等待 {wait_time} 秒")
                            logger.warning(f"速率限制: cat_id={cat_id}, page={page}, 嘗試={attempt}, 狀態碼=429, 等待 {wait_time} 秒")
                            await asyncio.sleep(wait_time)
                            proxy = random.choice(PROXY_LIST) if PROXY_LIST else None  # 更換代理
                            continue
                        
                        if response.status != 200:
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}")
                            logger.error(f"抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}")
                            return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
                        
                        data = await response.json()
                        if not data.get("success"):
                            rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                            logger.error(f"API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                            return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
                        
                        new_items = data["response"]["items"]
                        if not new_items:
                            break
                        
                        items.extend(new_items)
                        logger.info(f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(new_items)}")
                        await asyncio.sleep(random.uniform(2, 10))  # 隨機等待
                        break
                
                except Exception as e:
                    rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}")
                    logger.error(f"抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}")
                    return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
            
            if attempt >= max_attempts:
                rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - 重試失敗: cat_id={cat_id}, page={page}, 已達最大重試次數 {max_attempts}")
                logger.error(f"重試失敗: cat_id={cat_id}, page={page}, 已達最大重試次數 {max_attempts}")
                return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
    
    return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
