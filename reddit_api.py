import asyncpraw
import logging
import asyncio
from datetime import datetime
import streamlit as st
import time
import pytz
from logging_config import configure_logger

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = configure_logger(__name__, "reddit_api.log")

# 記錄當前請求次數和速率限制狀態
request_counter = 0
last_reset = time.time()
RATE_LIMIT_REQUESTS_PER_MINUTE = 600  # Reddit API 速率限制：每分鐘 600 個請求（認證用戶）
client_initialized = False  # 全局標誌，避免重複日誌

# 簡單緩存：存儲子版抓取結果和貼文內容
topic_cache = {}
thread_cache = {}
CACHE_DURATION = 300  # 緩存 5 分鐘
MAX_CACHE_SIZE = 100  # 最大緩存條目數

# 初始化 Reddit 客戶端
async def init_reddit_client():
    global request_counter, last_reset, client_initialized
    try:
        current_time = time.time()
        if current_time - last_reset >= 60:
            request_counter = 0
            last_reset = current_time
            logger.info("速率限制計數器重置")

        reddit = asyncpraw.Reddit(
            client_id=st.secrets["reddit"]["client_id"],
            client_secret=st.secrets["reddit"]["client_secret"],
            username=st.secrets["reddit"]["username"],
            password=st.secrets["reddit"]["password"],
            user_agent=f"LIHKGChatBot/v1.0 by u/{st.secrets['reddit']['username']}"
        )
        user = await reddit.user.me()
        if user and not client_initialized:
            logger.info(f"Reddit 客戶端初始化成功，已認證用戶：{user.name}")
            client_initialized = True
        elif user:
            logger.debug(f"Reddit 客戶端已初始化，重用認證用戶：{user.name}")
        else:
            logger.warning("Reddit 客戶端初始化成功，但未成功認證用戶，可能未獲得更高配額")
        return reddit
    except KeyError as e:
        logger.error(f"缺少 Secrets 配置：{str(e)}")
        raise
    except Exception as e:
        logger.error(f"Reddit 客戶端初始化失敗：{str(e)}")
        raise

async def get_subreddit_name(subreddit, reddit=None):
    if reddit is None:
        reddit = await init_reddit_client()
    try:
        subreddit_obj = await reddit.subreddit(subreddit)
        display_name = subreddit_obj.display_name
        return display_name
    except Exception as e:
        logger.error(f"獲取子版名稱失敗：{str(e)}")
        return "未知子版"
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()

def clean_cache(cache, cache_type="topic"):
    current_time = time.time()
    expired_keys = [key for key, value in cache.items() if current_time - value["timestamp"] > CACHE_DURATION]
    for key in expired_keys:
        del cache[key]
    # 限制緩存大小
    if len(cache) > MAX_CACHE_SIZE:
        sorted_keys = sorted(cache.items(), key=lambda x: x[1]["timestamp"])
        for key, _ in sorted_keys[:len(cache) - MAX_CACHE_SIZE]:
            del cache[key]
        logger.info(f"清理 {cache_type} 緩存，移除 {len(sorted_keys[:len(cache) - MAX_CACHE_SIZE])} 個過舊條目，當前緩存大小：{len(cache)}")

async def get_reddit_topic_list(subreddit, start_page=1, max_pages=1, reddit=None):
    global request_counter, last_reset, topic_cache
    cache_key = f"{subreddit}_{start_page}_{max_pages}"
    
    clean_cache(topic_cache, "topic")
    if cache_key in topic_cache:
        cached_data = topic_cache[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_DURATION:
            logger.info(f"使用緩存數據，子版：{subreddit}, 鍵：{cache_key}")
            return cached_data["data"]
    
    if reddit is None:
        reddit = await init_reddit_client()
    items = []
    rate_limit_info = []
    local_last_reset = last_reset  # 保存全局 last_reset 的當前值
    
    try:
        total_limit = 100
        subreddit_obj = await reddit.subreddit(subreddit)
        request_counter += 1
        if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
            wait_time = 60 - (time.time() - local_last_reset)
            if wait_time > 0:
                logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                local_last_reset = time.time()
                last_reset = local_last_reset  # 更新全局 last_reset
                logger.info("速率限制計數器重置")
        
        logger.info(f"開始抓取子版 {subreddit}，當前請求次數 {request_counter}")
        
        async for submission in subreddit_obj.new(limit=total_limit):
            if submission.created_utc:
                hk_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ)
                readable_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
                item = {
                    "thread_id": submission.id,
                    "title": submission.title,
                    "no_of_reply": submission.num_comments,
                    "last_reply_time": readable_time,
                    "like_count": submission.score
                }
                items.append(item)
        logger.info(f"抓取子版 {subreddit} 成功，總項目數 {len(items)}")
        
        topic_cache[cache_key] = {
            "timestamp": time.time(),
            "data": {
                "items": items,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": local_last_reset,
                "rate_limit_until": 0
            }
        }
    except Exception as e:
        logger.error(f"抓取貼文列表失敗：{str(e)}")
        rate_limit_info.append({"message": f"抓取子版 {subreddit} 失敗：{str(e)}"})
        return {
            "items": items,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": 0
        }
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()
    
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": local_last_reset,
        "rate_limit_until": 0
    }

async def get_reddit_thread_content(post_id, subreddit, max_comments=50, reddit=None):
    global request_counter, last_reset, thread_cache
    cache_key = f"{post_id}_subreddit_{subreddit}"
    
    clean_cache(thread_cache, "thread")
    if cache_key in thread_cache:
        cached_data = thread_cache[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_DURATION:
            logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
            return cached_data["data"]
    
    if reddit is None:
        reddit = await init_reddit_client()
        logger.debug(f"重新初始化 Reddit 客戶端，post_id: {post_id}, reddit type: {type(reddit)}")
    else:
        logger.debug(f"使用傳入 Reddit 客戶端，post_id: {post_id}, reddit type: {type(reddit)}")
    
    replies = []
    rate_limit_info = []
    local_last_reset = last_reset  # 保存全局 last_reset 的當前值
    
    try:
        request_counter += 1
        if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
            wait_time = 60 - (time.time() - local_last_reset)
            if wait_time > 0:
                logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                local_last_reset = time.time()
                last_reset = local_last_reset  # 更新全局 last_reset
                logger.info("速率限制計數器重置")
        
        logger.info(f"開始抓取貼文 {post_id}，當前請求次數 {request_counter}")
        
        await asyncio.sleep(1.0)  # 固定延遲

        submission = await reddit.submission(id=post_id)
        if not submission:
            logger.error(f"無法獲取貼文 {post_id}，submission 為 None")
            raise ValueError(f"貼文 {post_id} 獲取失敗")
        
        await submission.comments.replace_more(limit=max_comments)
        hk_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ)
        readable_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        
        comments_list = submission.comments.list()
        logger.debug(f"貼文 {post_id} 的評論數量：{len(comments_list)}")
        
        for comment in comments_list:
            if comment.body:
                hk_comment_time = datetime.fromtimestamp(comment.created_utc, tz=HONG_KONG_TZ)
                readable_comment_time = hk_comment_time.strftime("%Y-%m-%d %H:%M:%S")
                reply = {
                    "reply_id": comment.id,
                    "msg": comment.body,
                    "like_count": comment.score,
                    "reply_time": readable_comment_time
                }
                replies.append(reply)
        
        result = {
            "thread_id": submission.id,
            "title": submission.title,
            "no_of_reply": submission.num_comments,
            "last_reply_time": readable_time,
            "like_count": submission.score,
            "replies": replies[:max_comments],
            "total_replies": submission.num_comments,
            "fetched_pages": [1],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": 0
        }
        logger.info(f"抓取貼文 {post_id} 成功，總回覆數 {len(replies)}")
        
        thread_cache[cache_key] = {
            "timestamp": time.time(),
            "data": result
        }
        return result
    except Exception as e:
        logger.error(f"抓取貼文 {post_id} 失敗：{str(e)}")
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
        if "429" in str(e):
            wait_time = 120
            logger.warning(f"觸發速率限制，貼文 {post_id}，等待 {wait_time} 秒後重試")
            await asyncio.sleep(wait_time)
            return await get_reddit_thread_content(post_id, subreddit, max_comments, reddit)
        return {
            "replies": [],
            "title": None,
            "total_replies": 0,
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": last_reset + 120 if "429" in str(e) else 0
        }
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()

async def get_reddit_thread_content_batch(post_ids, subreddit, max_comments=50):
    global request_counter, last_reset
    # 移除重複貼文 ID
    unique_post_ids = list(dict.fromkeys(post_ids))
    if len(unique_post_ids) < len(post_ids):
        logger.info(f"移除重複貼文 ID，原始數量：{len(post_ids)}，唯一數量：{len(unique_post_ids)}")
    
    reddit = await init_reddit_client()
    reddit.is_shared = True
    results = []
    rate_limit_info = []
    local_last_reset = last_reset  # 保存全局 last_reset 的當前值
    
    try:
        batch_size = 1  # 單個請求，避免過多並行
        for i in range(0, len(unique_post_ids), batch_size):
            batch = unique_post_ids[i:i + batch_size]
            logger.info(f"開始抓取批次 {i//batch_size + 1}，貼文數量：{len(batch)}")
            
            tasks = [get_reddit_thread_content(post_id, subreddit, max_comments, reddit) for post_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for post_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"批量抓取貼文 {post_id} 失敗：{str(result)}")
                    # 檢查緩存作為回退
                    cache_key = f"{post_id}_subreddit_{subreddit}"
                    clean_cache(thread_cache, "thread")
                    if cache_key in thread_cache:
                        logger.info(f"抓取失敗，回退到緩存數據，貼文：{post_id}")
                        results.append(thread_cache[cache_key]["data"])
                    else:
                        results.append({
                            "thread_id": post_id,
                            "replies": [],
                            "title": None,
                            "total_replies": 0,
                            "fetched_pages": [],
                            "rate_limit_info": [{"message": f"抓取錯誤：{str(result)}"}],
                            "request_counter": request_counter,
                            "last_reset": local_last_reset,
                            "rate_limit_until": local_last_reset + 120 if "429" in str(result) else 0
                        })
                    rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(result)}"})
                    continue
                result["thread_id"] = post_id
                rate_limit_info.extend(result.get("rate_limit_info", []))
                results.append(result)
            
            # 批次間延遲
            delay = 10.0
            if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE - 10:
                delay = 60 - (time.time() - local_last_reset)
                if delay <= 0:
                    delay = 60
                request_counter = 0
                local_last_reset = time.time()
                last_reset = local_last_reset  # 更新全局 last_reset
                logger.info("速率限制計數器重置")
            logger.info(f"批次間延遲 {delay} 秒，當前請求次數 {request_counter}")
            if i + batch_size < len(unique_post_ids):
                await asyncio.sleep(delay)
        
        aggregated_rate_limit_data = {
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": max(r.get("rate_limit_until", 0) for r in results)
        }
        
        return {
            "results": results,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": aggregated_rate_limit_data["rate_limit_until"]
        }
    except Exception as e:
        logger.error(f"批量抓取失敗：{str(e)}")
        rate_limit_info.append({"message": f"批量抓取失敗：{str(e)}"})
        return {
            "results": results,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": local_last_reset + 120 if "429" in str(e) else 0
        }
    finally:
        await reddit.close()
