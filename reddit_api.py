"""
Reddit API 模組，負責從 Reddit 抓取子版貼文和回覆內容。
提供速率限制管理、錯誤處理和日誌記錄功能。
主要函數：
- get_reddit_submission_list：抓取指定子版的貼文列表。
- get_reddit_submission_content：抓取指定貼文的回覆內容。
- get_reddit_submission_content_batch：批量抓取多個貼文的回覆內容。
"""

import asyncpraw
import asyncio
import time
from datetime import datetime
import logging
import json
import pytz
from logging_config import configure_logger

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = configure_logger(__name__, "reddit_api.log")

# Reddit API 配置（請替換為實際憑證）
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "SocialMediaBot/1.0 by dopeloner2025"
REDDIT_USERNAME = "dopeloner2025"
REDDIT_PASSWORD = "your_password"

class RateLimiter:
    """
    速率限制器，控制 Reddit API 請求頻率，動態調整延遲。
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

# 初始化速率限制器
rate_limiter = RateLimiter(max_requests=60, period=60)

# Reddit 客戶端（單例模式）
reddit_client = None

async def initialize_reddit_client():
    """
    初始化 Reddit 客戶端，確保單例模式。
    """
    global reddit_client
    if reddit_client is None:
        try:
            reddit_client = asyncpraw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
                username=REDDIT_USERNAME,
                password=REDDIT_PASSWORD
            )
            user = await reddit_client.user.me()
            logger.info(f"Reddit 客戶端初始化成功，已認證用戶：{user.name}")
        except Exception as e:
            logger.error(f"Reddit 客戶端初始化失敗：{str(e)}")
            raise
    return reddit_client

async def get_reddit_submission_list(subreddit, limit=100):
    """
    抓取指定子版的貼文列表。
    """
    reddit = await initialize_reddit_client()
    items = []
    rate_limit_info = []
    
    success, rate_limit_data = await rate_limiter.acquire()
    if not success:
        rate_limit_info.append(f"{datetime.now(tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} HKT - Rate limit active until {datetime.fromtimestamp(rate_limit_data['rate_limit_until'], tz=HONG_KONG_TZ)}")
        return {
            "items": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": rate_limit_data["request_counter"],
            "last_reset": rate_limit_data["last_reset"],
            "rate_limit_until": rate_limit_data["rate_limit_until"]
        }
    
    logger.info(f"開始抓取子版 {subreddit}，當前請求次數 {rate_limit_data['request_counter']}")
    try:
        start_time = time.time()
        subreddit_obj = await reddit.subreddit(subreddit)
        async for submission in subreddit_obj.hot(limit=limit):
            items.append({
                "thread_id": submission.id,
                "title": submission.title,
                "no_of_reply": submission.num_comments,
                "like_count": submission.score,
                "last_reply_time": str(int(submission.created_utc))
            })
        response_time = time.time() - start_time
        rate_limiter.last_response_time = response_time
        logger.info(f"抓取子版 {subreddit} 成功，總項目數 {len(items)}")
    except asyncpraw.exceptions.RedditAPIException as e:
        if "RATELIMIT" in str(e):
            wait_time = 60  # 默認等待 60 秒
            rate_limiter.update_rate_limit(wait_time)
            rate_limit_info.append(f"{datetime.now(tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} HKT - Rate limit hit, waiting {wait_time:.2f} seconds")
            logger.warning(f"觸發速率限制，子版 {subreddit}")
        logger.error(f"抓取子版 {subreddit} 失敗：{str(e)}")
        items = []
    except Exception as e:
        logger.error(f"抓取子版 {subreddit} 失敗：{str(e)}")
        items = []
    
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": rate_limit_data["request_counter"],
        "last_reset": rate_limit_data["last_reset"],
        "rate_limit_until": rate_limit_data["rate_limit_until"]
    }

async def get_reddit_submission_content(submission_id, subreddit=None, max_replies=250):
    """
    抓取指定貼文的回覆內容。
    """
    reddit = await initialize_reddit_client()
    replies = []
    rate_limit_info = []
    
    success, rate_limit_data = await rate_limiter.acquire()
    if not success:
        rate_limit_info.append(f"{datetime.now(tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} HKT - Rate limit active until {datetime.fromtimestamp(rate_limit_data['rate_limit_until'], tz=HONG_KONG_TZ)}")
        return {
            "replies": [],
            "title": None,
            "total_replies": 0,
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": rate_limit_data["request_counter"],
            "last_reset": rate_limit_data["last_reset"],
            "rate_limit_until": rate_limit_data["rate_limit_until"]
        }
    
    logger.info(f"開始抓取貼文 {submission_id}，當前請求次數 {rate_limit_data['request_counter']}")
    try:
        start_time = time.time()
        submission = await reddit.submission(id=submission_id)
        await submission.load()
        submission.comment_sort = "top"
        await submission.comments.replace_more(limit=0)
        total_replies = len(submission.comments)
        replies_data = []
        for comment in submission.comments[:max_replies]:
            replies_data.append({
                "reply_id": comment.id,
                "msg": comment.body,
                "like_count": comment.score,
                "dislike_count": 0,
                "reply_time": str(int(comment.created_utc))
            })
        response_time = time.time() - start_time
        rate_limiter.last_response_time = response_time
        logger.info(f"抓取貼文 {submission_id} 成功，總回覆數 {len(replies_data)}")
        replies = replies_data
    except asyncpraw.exceptions.RedditAPIException as e:
        if "RATELIMIT" in str(e):
            wait_time = 60
            rate_limiter.update_rate_limit(wait_time)
            rate_limit_info.append(f"{datetime.now(tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} HKT - Rate limit hit, waiting {wait_time:.2f} seconds")
            logger.warning(f"觸發速率限制，貼文 {submission_id}")
        logger.error(f"抓取貼文 {submission_id} 失敗：{str(e)}")
        replies = []
    except Exception as e:
        logger.error(f"抓取貼文 {submission_id} 失敗：{str(e)}")
        replies = []
    
    return {
        "replies": replies,
        "title": submission.title if 'submission' in locals() else None,
        "total_replies": total_replies if 'total_replies' in locals() else 0,
        "fetched_pages": [1],
        "rate_limit_info": rate_limit_info,
        "request_counter": rate_limit_data["request_counter"],
        "last_reset": rate_limit_data["last_reset"],
        "rate_limit_until": rate_limit_data["rate_limit_until"]
    }

async def get_reddit_submission_content_batch(submission_ids, subreddit=None, max_replies=250):
    """
    批量抓取多個貼文的回覆內容。
    """
    results = []
    rate_limit_info = []
    
    for submission_id in submission_ids:
        result = await get_reddit_submission_content(
            submission_id=submission_id,
            subreddit=subreddit,
            max_replies=max_replies
        )
        result["thread_id"] = submission_id
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
