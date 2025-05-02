import praw
import time
import streamlit as st
import logging
from datetime import datetime
import pytz
from logging_config import configure_logger

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = configure_logger(__name__, "reddit_api.log")

# 初始化 Reddit 客戶端
def init_reddit_client():
    try:
        reddit = praw.Reddit(
            client_id=st.secrets["reddit"]["client_id"],
            client_secret=st.secrets["reddit"]["client_secret"],
            username=st.secrets["reddit"]["username"],
            password=st.secrets["reddit"]["password"],
            user_agent=f"LIHKGChatBot/v1.0 by u/{st.secrets['reddit']['username']}"
        )
        logger.info("Reddit 客戶端初始化成功")
        return reddit
    except KeyError as e:
        logger.error(f"缺少 Secrets 配置：{str(e)}")
        raise
    except Exception as e:
        logger.error(f"Reddit 客戶端初始化失敗：{str(e)}")
        raise

def get_subreddit_name(subreddit):
    """
    返回指定子版的顯示名稱。
    """
    reddit = init_reddit_client()
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        return subreddit_obj.display_name
    except Exception as e:
        logger.error(f"獲取子版名稱失敗：{str(e)}")
        return "未知子版"

async def get_reddit_topic_list(subreddit="wallstreetbets", start_page=1, max_pages=3):
    """
    抓取指定子版的貼文元數據。
    """
    reddit = init_reddit_client()
    items = []
    rate_limit_info = []
    
    try:
        for page in range(start_page, start_page + max_pages):
            submissions = reddit.subreddit(subreddit).new(limit=100)
            for submission in submissions:
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
            logger.info(f"抓取子版 {subreddit}，頁面 {page}，項目數 {len(items)}")
            if not list(submissions):
                break
    except Exception as e:
        logger.error(f"抓取貼文列表失敗：{str(e)}")
    
    return {
        "items": items[:90],
        "rate_limit_info": rate_limit_info,
        "request_counter": 0,
        "last_reset": time.time(),
        "rate_limit_until": 0
    }

async def get_reddit_thread_content(post_id, subreddit="wallstreetbets", max_comments=250):
    """
    抓取指定貼文的詳細內容。
    """
    reddit = init_reddit_client()
    replies = []
    rate_limit_info = []
    
    try:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=max_comments)
        hk_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ)
        readable_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        
        for comment in submission.comments.list():
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
            "request_counter": 0,
            "last_reset": time.time(),
            "rate_limit_until": 0
        }
        logger.info(f"抓取貼文 {post_id} 成功，總回覆數 {len(replies)}")
        return result
    except Exception as e:
        logger.error(f"抓取貼文 {post_id} 失敗：{str(e)}")
        return {
            "replies": [],
            "title": None,
            "total_replies": 0,
            "fetched_pages": [],
            "rate_limit_info": [{"message": f"抓取失敗：{str(e)}"}],
            "request_counter": 0,
            "last_reset": time.time(),
            "rate_limit_until": 0
        }

async def get_reddit_thread_content_batch(post_ids, subreddit="wallstreetbets", max_comments=250):
    """
    批量抓取多個貼文的詳細內容。
    """
    reddit = init_reddit_client()
    results = []
    rate_limit_info = []
    
    tasks = []
    for post_id in post_ids:
        tasks.append(get_reddit_thread_content(post_id, subreddit, max_comments))
    
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
                "request_counter": 0,
                "last_reset": time.time(),
                "rate_limit_until": 0
            })
            continue
        result["thread_id"] = post_id
        rate_limit_info.extend(result.get("rate_limit_info", []))
        results.append(result)
    
    aggregated_rate_limit_data = {
        "request_counter": 0,
        "last_reset": time.time(),
        "rate_limit_until": 0
    }
    
    return {
        "results": results,
        "rate_limit_info": rate_limit_info,
        "request_counter": aggregated_rate_limit_data["request_counter"],
        "last_reset": aggregated_rate_limit_data["last_reset"],
        "rate_limit_until": aggregated_rate_limit_data["rate_limit_until"]
    }
