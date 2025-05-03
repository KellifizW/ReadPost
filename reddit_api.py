import asyncpraw
import logging
import asyncio
from datetime import datetime
import streamlit as st
import time
import pytz
from logging_config import configure_logger

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_KONG")

# 配置日誌記錄器
logger = configure_logger(__name__, "reddit_api.log")

# 記錄當前請求次數和速率限制狀態
request_counter = 0
last_reset = time.time()
RATE_LIMIT_REQUESTS_PER_MINUTE = 60  # Reddit API 速率限制：每分鐘 60 個請求
request_lock = asyncio.Lock()  # 用於同步請求計數器

# 簡單緩存：存儲子版抓取結果和貼文內容
topic_cache = {}
thread_cache = {}
CACHE_DURATION = 300  # 緩存 5 分鐘

# 初始化 Reddit 客戶端（保持不變）
async def init_reddit_client():
    global request_counter, last_reset
    async with request_lock:
        current_time = time.time()
        if current_time - last_reset >= 60:
            request_counter = 0
            last_reset = current_time
            logger.info("速率限制計數器重置")
    
    try:
        reddit = asyncpraw.Reddit(
            client_id=st.secrets["reddit"]["client_id"],
            client_secret=st.secrets["reddit"]["client_secret"],
            username=st.secrets["reddit"]["username"],
            password=st.secrets["reddit"]["password"],
            user_agent=f"LIHKGChatBot/v1.0 by u/{st.secrets['reddit']['username']}"
        )
        user = await reddit.user.me()
        if user:
            logger.info(f"Reddit 客戶端初始化成功，已認證用戶：{user.name}")
        else:
            logger.warning("Reddit 客戶端初始化成功，但未成功認證用戶，可能未獲得更高配額")
        return reddit
    except KeyError as e:
        logger.error(f"缺少 Secrets 配置：{str(e)}")
        raise
    except Exception as e:
        logger.error(f"Reddit 客戶端初始化失敗：{str(e)}")
        raise

# get_subreddit_name（保持不變）
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

# get_reddit_topic_list（保持不變）
async def get_reddit_topic_list(subreddit, start_page=1, max_pages=1, reddit=None):
    global request_counter, topic_cache
    cache_key = f"{subreddit}"
    
    if cache_key in topic_cache:
        cached_data = topic_cache[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_DURATION:
            logger.info(f"使用緩存數據，子版：{subreddit}")
            return cached_data["data"]
    
    if reddit is None:
        reddit = await init_reddit_client()
    items = []
    rate_limit_info = []
    
    try:
        async with request_lock:
            current_time = time.time()
            elapsed = current_time - last_reset
            if elapsed < 60 and request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
                wait_time = 60 - elapsed
                logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                last_reset = current_time
            
            request_counter += 1
            logger.info(f"開始抓取子版 {subreddit}，當前請求次數 {request_counter}")
        
        total_limit = 100
        subreddit_obj = await reddit.subreddit(subreddit)
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
                "last_reset": last_reset,
                "rate_limit_until": 0
            }
        }
    except Exception as e:
        logger.error(f"抓取貼文列表失敗：{str(e)}")
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()
    
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": 0
    }

async def get_reddit_thread_content(post_id, subreddit, max_comments=50, reddit=None):
    global request_counter, thread_cache
    cache_key = f"{post_id}_subreddit_{subreddit}"
    
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
    
    try:
        async with request_lock:
            current_time = time.time()
            elapsed = current_time - last_reset
            if elapsed < 60 and request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
                wait_time = 60 - elapsed
                logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                last_reset = current_time
            
            request_counter += 1
            logger.info(f"開始抓取貼文 {post_id}，當前請求次數 {request_counter}")
        
        # 根據剩餘配額動態延遲
        if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE * 0.8:  # 達到 80% 配額
            wait_time = max(1.0, (60 - elapsed) / (RATE_LIMIT_REQUESTS_PER_MINUTE - request_counter + 1))
            logger.info(f"接近速率限制，剩餘請求數：{RATE_LIMIT_REQUESTS_PER_MINUTE - request_counter}，延遲 {wait_time:.2f} 秒")
            await asyncio.sleep(wait_time)
        
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
            "last_reset": last_reset,
            "rate_limit_until": 0
        }
        logger.info(f"抓取貼文 {post_id} 成功，總回覆數 {len(replies)}")
        
        thread_cache[cache_key] = {
            "timestamp": time.time(),
            "data": result
        }
        return result
    except asyncpraw.exceptions.RedditAPIException as e:
        if "RATELIMIT" in str(e) or "429" in str(e):
            wait_time = 120  # 默認等待時間
            try:
                # 檢查響應頭中的速率限制信息
                if hasattr(e, 'response') and e.response:
                    headers = e.response.headers
                    remaining = int(headers.get('x-ratelimit-remaining', 0))
                    reset_time = int(headers.get('x-ratelimit-reset', 0))
                    if remaining == 0 and reset_time > 0:
                        wait_time = max(wait_time, reset_time - time.time() + 1)
                        logger.info(f"從頭信息獲取重置時間，等待 {wait_time:.2f} 秒")
                # 嘗試從異常中提取速率限制信息
                for item in e.items:
                    if item.error_type == "RATELIMIT":
                        message = item.message.lower()
                        if "seconds" in message:
                            seconds = int(''.join(filter(str.isdigit, message)))
                            wait_time = max(wait_time, seconds)
                        elif "minutes" in message:
                            minutes = int(''.join(filter(str.isdigit, message)))
                            wait_time = max(wait_time, minutes * 60)
            except Exception as parse_error:
                logger.warning(f"無法解析速率限制信息：{parse_error}")
            
            logger.warning(f"觸發速率限制，貼文 {post_id}，等待 {wait_time} 秒後重試")
            async with request_lock:
                rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：RATELIMIT", "wait_time": wait_time})
                current_time = time.time()
                rate_limit_until = current_time + wait_time
            await asyncio.sleep(wait_time)
            async with request_lock:
                request_counter = 0
                last_reset = time.time()
            return await get_reddit_thread_content(post_id, subreddit, max_comments, reddit)
        else:
            logger.error(f"抓取貼文 {post_id} 失敗：{str(e)}")
            rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
            return {
                "replies": [],
                "title": None,
                "total_replies": 0,
                "fetched_pages": [],
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": 0
            }
    except Exception as e:
        logger.error(f"抓取貼文 {post_id} 失敗：{str(e)}")
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
        return {
            "replies": [],
            "title": None,
            "total_replies": 0,
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": 0
        }
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()

async def get_reddit_thread_content_batch(post_ids, subreddit, max_comments=50):
    global request_counter
    reddit = await init_reddit_client()
    reddit.is_shared = True
    results = []
    rate_limit_info = []
    
    try:
        batch_size = 2  # 減少批次大小，降低請求壓力
        post_ids = list(dict.fromkeys(post_ids))  # 保留順序去重
        
        # 檢查緩存，過濾已緩存的貼文
        cached_results = []
        uncached_post_ids = []
        for post_id in post_ids:
            cache_key = f"{post_id}_subreddit_{subreddit}"
            if cache_key in thread_cache:
                cached_data = thread_cache[cache_key]
                if time.time() - cached_data["timestamp"] < CACHE_DURATION:
                    logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
                    cached_results.append(cached_data["data"])
                    continue
            uncached_post_ids.append(post_id)
        
        for i in range(0, len(uncached_post_ids), batch_size):
            batch = uncached_post_ids[i:i + batch_size]
            logger.info(f"開始抓取批次 {i//batch_size + 1}，貼文數量：{len(batch)}")
            
            async with request_lock:
                current_time = time.time()
                elapsed = current_time - last_reset
                requests_remaining = RATE_LIMIT_REQUESTS_PER_MINUTE - request_counter
                if elapsed < 60 and requests_remaining < len(batch):
                    wait_time = 60 - elapsed
                    logger.warning(f"批次請求數 {len(batch)} 超過剩餘配額 {requests_remaining}，等待 {wait_time:.2f} 秒")
                    await asyncio.sleep(wait_time)
                    request_counter = 0
                    last_reset = current_time
            
            tasks = [get_reddit_thread_content(post_id, subreddit, max_comments, reddit) for post_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for post_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"批量抓取貼文 {post_id} 失敗：{str(result)}")
                    results.append({
                        "thread_id": post_id,
                        "replies": [],
                        "title": None,
                        "total_replies": 0,
                        "fetched_pages": [],
                        "rate_limit_info": [{"message": f"抓取錯誤：{str(result)}"}],
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": last_reset + 60 if "429" in str(result) else 0
                    })
                    rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(result)}"})
                    continue
                result["thread_id"] = post_id
                rate_limit_info.extend(result.get("rate_limit_info", []))
                results.append(result)
            
            # 動態批次間延遲
            async with request_lock:
                current_time = time.time()
                elapsed = current_time - last_reset
                requests_remaining = RATE_LIMIT_REQUESTS_PER_MINUTE - request_counter
                requests_per_second = RATE_LIMIT_REQUESTS_PER_MINUTE / 60.0
                delay = max(5.0, len(batch) / requests_per_second) if elapsed < 60 else 5.0
                logger.info(f"批次間延遲 {delay:.2f} 秒，當前請求次數 {request_counter}")
            
            if i + batch_size < len(uncached_post_ids):
                await asyncio.sleep(delay)
        
        # 合併緩存結果和抓取結果
        results.extend(cached_results)
        aggregated_rate_limit_data = {
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": max(r.get("rate_limit_until", 0) for r in results)
        }
        
        return {
            "results": results,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": aggregated_rate_limit_data["rate_limit_until"]
        }
    finally:
        await reddit.close()
