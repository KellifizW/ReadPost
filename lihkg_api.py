import aiohttp
import streamlit.logger
import asyncio

logger = streamlit.logger.get_logger(__name__)

LIHKG_BASE_URL = "https://lihkg.com"
LIHKG_DEVICE_ID = "device_id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
LIHKG_COOKIE = "LIHKG_NEXT_ACCESS=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx; LIHKG_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

async def get_lihkg_topic_list(cat_id, sub_cat_id, start_page, max_pages):
    headers = {
        "User-Agent": "LIHKG/7.6.2 (iPhone; iOS 17.4.1; Scale/3.00)",
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9",
        "Connection": "keep-alive",
        "Cookie": LIHKG_COOKIE,
    }
    
    items = []
    async with aiohttp.ClientSession() as session:
        for page in range(start_page, start_page + max_pages):
            url = f"{LIHKG_BASE_URL}/api_v2/thread/category?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={page}&count=60"
            headers["Referer"] = f"{LIHKG_BASE_URL}/category/{cat_id}"
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"抓取失敗: cat_id={cat_id}, page={page}, 狀態碼={response.status}")
                        break
                    
                    data = await response.json()
                    logger.debug(f"帖子列表 API 回應: cat_id={cat_id}, page={page}, 數據={data}")
                    if not data.get("success"):
                        logger.error(f"API 返回失敗: cat_id={cat_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                        break
                    
                    new_items = data["response"]["items"]
                    if not new_items:
                        break
                    
                    items.extend(new_items)
                    logger.info(f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(new_items)}")
            
            except Exception as e:
                logger.error(f"抓取錯誤: cat_id={cat_id}, page={page}, 錯誤={str(e)}")
                break
    
    return items

async def get_lihkg_thread_content(thread_id, cat_id=None):
    headers = {
        "User-Agent": "LIHKG/7.6.2 (iPhone; iOS 17.4.1; Scale/3.00)",
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9",
        "Connection": "keep-alive",
        "Cookie": LIHKG_COOKIE,
    }
    
    replies = []
    page = 1
    thread_title = None
    total_replies = None
    
    async with aiohttp.ClientSession() as session:
        while True:
            url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/message?page={page}&count=100"
            headers["Referer"] = f"{LIHKG_BASE_URL}/thread/{thread_id}"
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"抓取帖子內容失敗: thread_id={thread_id}, page={page}, 狀態碼={response.status}")
                        break
                    
                    data = await response.json()
                    logger.debug(f"帖子內容 API 回應: thread_id={thread_id}, page={page}, 數據={data}")
                    if not data.get("success"):
                        logger.error(f"API 返回失敗: thread_id={thread_id}, page={page}, 錯誤={data.get('error_message', '未知錯誤')}")
                        break
                    
                    # 提取標題（檢查多個可能的字段）
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
                    
                    # 檢查是否還有更多回覆
                    if total_replies and len(replies) >= total_replies:
                        break
            
            except Exception as e:
                logger.error(f"抓取帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤={str(e)}")
                break
    
    return {"replies": replies, "title": thread_title, "total_replies": total_replies}
