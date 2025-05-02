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
RATE_LIMIT_REQUESTS_PER_MINUTE = 60  # Reddit API 速率限制：每分鐘 60 個請求

# 初始化 Reddit 客戶端
async def init_reddit_client():
    global request_counter, last_reset
    try:
        # 重置速率限制計數器（每分鐘重置一次）
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
        # 驗證認證狀態
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

async def get_subreddit_name(subreddit):
    """
    返回指定子版的顯示名稱。
    """
    reddit = await init_reddit_client()
    try:
        subreddit_obj = await reddit.subreddit(subreddit)
        display_name = subreddit_obj.display_name
        return display_name
    except Exception as e:
        logger.error(f"獲取子版名稱失敗：{str(e)}")
        return "未知子版"
    finally:
        await reddit.close()

async def get_reddit_topic_list(subreddit="wallstreetbets", start_page=1, max_pages=3):
    """
    抓取指定子版的貼文元數據。
    """
    global request_counter
    reddit = await init_reddit_client()
    items = []
    rate_limit_info = []
    
    try:
        total_limit = 100 * max_pages  # 模擬多頁，總數量限制
        subreddit_obj = await reddit.subreddit(subreddit)
        request_counter += 1
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
            if len(items) >= 90:  # 限制最多 90 項，與原邏輯一致
                break
        logger.info(f"抓取子版 {subreddit} 成功，總項目數 {len(items)}")
    except Exception as e:
        logger.error(f"抓取貼文列表失敗：{str(e)}")
    finally:
        await reddit.close()
    
    return {
        "items": items[:90],
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": 0
    }

async def get_reddit_thread_content(post_id, subreddit="wallstreetbets", max_comments=250, reddit=None):
    """
    抓取指定貼文的詳細內容。
    """
    global request_counter
    replies = []
    rate_limit_info = []
    
    try:
        request_counter += 1
        logger.info(f"開始抓取貼文 {post_id}，當前請求次數 {request_counter}")
        
        submission = await reddit.submission(id=post_id)
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
        return result
    except Exception as e:
        logger.error(f"抓取貼文 {post_id} 失敗：{str(e)}")
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
        if "429" in str(e):
            logger.warning(f"觸發速率限制，貼文 {post_id}，當前請求次數 {request_counter}")
        return {
            "replies": [],
            "title": None,
            "total_replies": 0,
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": last_reset + 60 if "429" in str(e) else 0
        }

async def get_reddit_thread_content_batch(post_ids, subreddit="wallstreetbets", max_comments=250):
    """
    批量抓取多個貼文的詳細內容。
    """
    global request_counter
    reddit = await init_reddit_client()
    results = []
    rate_limit_info = []
    
    try:
        tasks = []
        for idx, post_id in enumerate(post_ids):
            # 在每個請求之間添加延遲以避免速率限制
            if idx > 0:
                delay = 1.0  # 每個請求間隔 1 秒
                logger.info(f"為避免速率限制，在抓取貼文 {post_id} 前延遲 {delay} 秒，當前請求次數 {request_counter}")
                await asyncio.sleep(delay)
            tasks.append(get_reddit_thread_content(post_id, subreddit, max_comments, reddit))
        
        content_results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(content_results):
            post_id = post_ids[idx]
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
        
        aggregated_rate_limit_data = {
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": max(r.get("rate_limit_until", 0) for r in results)
        }
        
        return {
            "results": results,
            "rate_limit_info": rate_limit_info,
            "request_counter": aggregated_rate_limit_data["request_counter"],
            "last_reset": aggregated_rate_limit_data["last_reset"],
            "rate_limit_until": aggregated_rate_limit_data["rate_limit_until"]
        }
    finally:
        await reddit.close()
