import asyncpraw
import logging
import asyncio
from datetime import datetime
import streamlit as st
import time
import pytz
from contextlib import asynccontextmanager
from logging_config import configure_logger
import aiohttp

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "reddit_api.log")

class RedditClientManager:
    def __init__(self):
        try:
            credentials = st.secrets["reddit"]["credentials"]
            if not credentials or len(credentials) < 1:
                raise KeyError("Reddit credentials are missing or empty")
            self.clients = [None] * len(credentials)
            self.request_counters = [0] * len(self.clients)
            self.last_resets = [time.time()] * len(self.clients)
            self.rate_limit_requests_per_minute = 60
            self.current_client_index = 0
            self.topic_cache = {}
            self.thread_cache = {}
            self.topic_cache_duration = 1800
            self.thread_cache_duration = 300
            self.max_cache_size = 100
        except KeyError as e:
            logger.error(f"Secrets 配置錯誤：{str(e)}")
            raise

    @asynccontextmanager
    async def get_client(self):
        client_idx = self.current_client_index % len(self.clients)
        if self.clients[client_idx] is None or not await self._is_client_valid(self clients[client_idx]):
            self.clients[client_idx] = await self._init_reddit_client(client_idx)
        try:
            self.request_counters[client_idx], self.last_resets[client_idx] = await self._handle_rate_limit(client_idx)
            self.current_client_index += 1
            yield self.clients[client_idx]
        finally:
            pass

    async def _init_reddit_client(self, client_idx):
        try:
            cred = st.secrets["reddit"]["credentials"][client_idx]
            reddit = asyncpraw.Reddit(
                client_id=cred["client_id"],
                client_secret=cred["client_secret"],
                username=st.secrets["reddit"]["username"],
                password=st.secrets["reddit"]["password"],
                user_agent=f"LIHKGChatBot/v1.0 by u/{st.secrets['reddit']['username']}_{client_idx}",
                timeout=30
            )
            reddit.is_shared = True
            user = await reddit.user.me()
            logger.info(f"Reddit 客戶端 {client_idx} 初始化成功，已認證用戶：{user.name}")
            return reddit
        except KeyError as e:
            logger.error(f"缺少 Secrets 配置：{str(e)}")
            raise
        except Exception as e:
            logger.error(f"Reddit 客戶端 {client_idx} 初始化失敗：{str(e)}")
            raise

    async def _handle_rate_limit(self, client_idx):
        if self.request_counters[client_idx] >= self.rate_limit_requests_per_minute:
            wait_time = 60 - (time.time() - self.last_resets[client_idx])
            if wait_time > 0:
                logger.warning(f"客戶端 {client_idx} 達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                self.request_counters[client_idx] = 0
                self.last_resets[client_idx] = time.time()
                logger.info(f"客戶端 {client_idx} 速率限制計數器重置")
        self.request_counters[client_idx] += 1
        return self.request_counters[client_idx], self.last_resets[client_idx]

    async def _is_client_valid(self, reddit):
        try:
            await reddit.user.me()
            return True
        except Exception:
            logger.warning(f"客戶端無效，重新初始化")
            return False

    def clean_cache(self, cache, cache_type="topic"):
        duration = self.topic_cache_duration if cache_type == "topic" else self.thread_cache_duration
        current_time = time.time()
        expired_keys = [k for k, v in cache.items() if current_time - v["timestamp"] > duration]
        for k in expired_keys:
            del cache[k]
        if len(cache) > self.max_cache_size:
            sorted_keys = sorted(cache.items(), key=lambda x: x[1]["timestamp"])[:len(cache) - self.max_cache_size]
            for k, _ in sorted_keys:
                del cache[k]
            logger.info(f"清理 {cache_type} 緩存，移除 {len(sorted_keys)} 個條目，當前緩存大小：{len(cache)}")

client_manager = RedditClientManager()

async def get_subreddit_name(subreddit):
    async with client_manager.get_client() as reddit:
        try:
            subreddit_obj = await reddit.subreddit(subreddit)
            return subreddit_obj.display_name
        except Exception as e:
            logger.error(f"獲取子版名稱失敗：{str(e)}")
            return "未知子版"

async def format_submission(submission):
    hk_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ)
    return {
        "thread_id": submission.id,
        "title": submission.title,
        "no_of_reply": submission.num_comments,
        "last_reply_time": hk_time.strftime("%Y-%m-%d %H:%M:%S"),
        "like_count": submission.score
    }

async def get_reddit_topic_list(subreddit, start_page=1, max_pages=1, sort="best"):
    cache_key = f"{subreddit}_{start_page}_{max_pages}_{sort}"
    client_manager.clean_cache(client_manager.topic_cache, "topic")
    
    if cache_key in client_manager.topic_cache:
        cached_data = client_manager.topic_cache[cache_key]
        if time.time() - cached_data["timestamp"] < client_manager.topic_cache_duration:
            logger.info(f"使用緩存數據，子版：{subreddit}, 鍵：{cache_key}")
            return cached_data["data"]
    
    async with client_manager.get_client() as reddit:
        items = []
        rate_limit_info = []
        total_limit = 100
        
        try:
            subreddit_obj = await reddit.subreddit(subreddit)
            sort_methods = {"best": lambda: subreddit_obj.top(time_filter="day", limit=total_limit),
                          "new": lambda: subreddit_obj.new(limit=total_limit)}
            if sort not in sort_methods:
                raise ValueError(f"不支持的排序方式：{sort}")
            
            logger.info(f"開始抓取子版 {subreddit}，排序：{sort}，當前請求次數 {client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)]}")
            
            async for submission in sort_methods[sort]():
                if submission.created_utc:
                    items.append(await format_submission(submission))
            
            logger.info(f"抓取子版 {subreddit} 成功，總項目數 {len(items)}")
            
            client_manager.topic_cache[cache_key] = {
                "timestamp": time.time(),
                "data": {
                    "items": items,
                    "rate_limit_info": rate_limit_info,
                    "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                    "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                    "rate_limit_until": 0,
                    "total_posts": len(items)  # Track total posts
                }
            }
        except Exception as e:
            logger.error(f"抓取貼文列表失敗：{str(e)}")
            rate_limit_info.append({"message": f"抓取子版 {subreddit} 失敗：{str(e)}"})
        
        return {
            "items": items,
            "rate_limit_info": rate_limit_info,
            "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
            "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
            "rate_limit_until": 0,
            "total_posts": len(items)
        }

async def collect_more_comments(reddit, submission, max_comments, request_counter, last_reset):
    more_comments_ids = []
    comments = []
    
    try:
        async for comment in submission.comments:
            if isinstance(comment, asyncpraw.models.MoreComments):
                more_comments_ids.extend(comment.children)
            elif hasattr(comment, 'body') and comment.body:
                hk_time = datetime.fromtimestamp(comment.created_utc, tz=HONG_KONG_TZ)
                comments.append({
                    "reply_id": comment.id,
                    "msg": comment.body,
                    "like_count": comment.score,
                    "reply_time": hk_time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        BATCH_SIZE = 100
        for i in range(0, len(more_comments_ids), BATCH_SIZE):
            if len(comments) >= max_comments:
                break
            batch_ids = more_comments_ids[i:i + BATCH_SIZE]
            if not batch_ids:
                continue
            
            request_counter, last_reset = await client_manager._handle_rate_limit(client_manager.current_client_index % len(client_manager.clients))
            
            try:
                more_comments = await reddit.comment.more_children(
                    link_id=f"t3_{submission.id}",
                    children=",".join(batch_ids),
                    sort="best"
                )
                for comment in more_comments:
                    if hasattr(comment, 'body') and comment.body:
                        hk_time = datetime.fromtimestamp(comment.created_utc, tz=HONG_KONG_TZ)
                        comments.append({
                            "reply_id": comment.id,
                            "msg": comment.body,
                            "like_count": comment.score,
                            "reply_time": hk_time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        if len(comments) >= max_comments:
                            break
            except (aiohttp.ClientResponseError, asyncio.TimeoutError) as e:
                logger.warning(f"/api/morechildren 請求失敗：{str(e)}")
                continue
            
        return comments[:max_comments], request_counter, last_reset
    
    except Exception as e:
        logger.error(f"收集更多評論失敗：{str(e)}")
        return comments[:max_comments], request_counter, last_reset

async def fetch_single_thread(post_id, subreddit, max_comments, reddit, request_counter, last_reset):
    cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}"
    client_manager.clean_cache(client_manager.thread_cache, "thread")
    
    if cache_key in client_manager.thread_cache:
        cached_data = client_manager.thread_cache[cache_key]
        if time.time() - cached_data["timestamp"] < client_manager.thread_cache_duration:
            logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
            return cached_data["data"], request_counter, last_reset
    
    replies = []
    rate_limit_info = []
    
    try:
        async with asyncio.timeout(30):
            request_counter, last_reset = await client_manager._handle_rate_limit(client_manager.current_client_index % len(client_manager.clients))
            
            logger.info(f"開始抓取貼文：[{post_id}]，當前請求次數：{request_counter}")
            
            submission = await reddit.submission(id=post_id)
            if not submission:
                logger.error(f"無法獲取貼文：{post_id}")
                rate_limit_info.append({"message": f"無法獲取貼文 {post_id}"})
                return {
                    "title": "",
                    "total_replies": 0,
                    "last_reply_time": "1970-01-01 00:00:00",
                    "like_count": 0,
                    "replies": [],
                    "fetched_pages": [],
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": 0
                }, request_counter, last_reset
            
            title = submission.title
            total_replies = submission.num_comments
            like_count = submission.score
            last_reply_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:%M:%S")
            
            replies, request_counter, last_reset = await collect_more_comments(
                reddit, submission, max_comments, request_counter, last_reset
            )
            
            logger.info(f"抓取貼文完成：{{{post_id}: {len(replies)}}}，總回覆數：{len(replies)}")
            
            result = {
                "title": title,
                "total_replies": total_replies,
                "last_reply_time": last_reply_time,
                "like_count": like_count,
                "replies": replies,
                "fetched_pages": [1],
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": 0,
                "replies_count": len(replies)  # Track replies count
            }
            
            client_manager.thread_cache[cache_key] = {
                "timestamp": time.time(),
                "data": result
            }
            
            return result, request_counter, last_reset
        
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            logger.error(f"速率限制錯誤（429）抓取貼文 {post_id}：{str(e)}")
            rate_limit_info.append({"message": f"速率限制錯誤（429）抓取貼文 {post_id}：{str(e)}"})
        else:
            logger.error(f"抓取貼文內容失敗：{str(e)}")
            rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
        return {
            "title": "",
            "total_replies": 0,
            "last_reply_time": "1970-01-01 00:00:00",
            "like_count": 0,
            "replies": [],
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": 0
        }, request_counter, last_reset
    except asyncio.TimeoutError as e:
        logger.error(f"抓取貼文內容超時：{str(e)}")
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 超時：{str(e)}"})
        return {
            "title": "",
            "total_replies": 0,
            "last_reply_time": "1970-01-01 00:00:00",
            "like_count": 0,
            "replies": [],
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": 0
        }, request_counter, last_reset
    except Exception as e:
        logger.error(f"抓取貼文內容失敗：{str(e)}")
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
        return {
            "title": "",
            "total_replies": 0,
            "last_reply_time": "1970-01-01 00:00:00",
            "like_count": 0,
            "replies": [],
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": 0
        }, request_counter, last_reset

async def get_reddit_thread_content(post_id, subreddit, max_comments=100):
    async with client_manager.get_client() as reddit:
        result, request_counter, last_reset = await fetch_single_thread(
            post_id, subreddit, max_comments, reddit, 
            client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
            client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)]
        )
        return result

async def get_reddit_thread_content_batch(post_ids, subreddit, max_comments=100):
    results = []
    rate_limit_info = []
    fetch_status = {}
    batch_size = 5
    total_posts = 0
    total_replies = 0
    
    async with client_manager.get_client() as reddit:
        try:
            logger.info(f"開始批次抓取貼文：{post_ids}")
            
            ids = [f"t3_{pid}" for pid in post_ids]
            submissions = {s.id: s async for s in reddit.info(fullnames=ids)}
            
            for i in range(0, len(post_ids), batch_size):
                batch_ids = post_ids[i:i + batch_size]
                tasks = []
                for post_id in batch_ids:
                    cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}"
                    client_manager.clean_cache(client_manager.thread_cache, "thread")
                    if cache_key in client_manager.thread_cache:
                        cached_data = client_manager.thread_cache[cache_key]
                        if time.time() - cached_data["timestamp"] < client_manager.thread_cache_duration:
                            logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
                            results.append(cached_data["data"])
                            fetch_status[post_id] = {"status": "cached", "replies": len(cached_data["data"]["replies"])}
                            total_posts += 1
                            total_replies += len(cached_data["data"]["replies"])
                            continue
                    tasks.append(fetch_single_thread(
                        post_id, subreddit, max_comments, reddit,
                        client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                        client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)]
                    ))
                
                if tasks:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in batch_results:
                        if isinstance(result, tuple):
                            result, req_count, last_reset = result
                            results.append(result)
                            fetch_status[result.get("title", post_id)] = {
                                "status": "success",
                                "replies": len(result.get("replies", []))
                            }
                            rate_limit_info.extend(result.get("rate_limit_info", []))
                            client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)] = req_count
                            client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)] = last_reset
                            total_posts += 1
                            total_replies += len(result.get("replies", []))
                        else:
                            logger.error(f"批次抓取貼文失敗：{str(result)}")
                            rate_limit_info.append({"message": f"批次抓取貼文失敗：{str(result)}"})
                            results.append({
                                "title": "",
                                "total_replies": 0,
                                "last_reply_time": "1970-01-01 00:00:00",
                                "like_count": 0,
                                "replies": [],
                                "fetched_pages": [],
                                "rate_limit_info": rate_limit_info,
                                "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                                "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                                "rate_limit_until": 0
                            })
                            fetch_status[post_id] = {"status": "failed", "replies": 0}
            
            total_requests = sum(client_manager.request_counters)
            logger.info(f"Reddit API 調用統計：抓取帖子數={total_posts}, 回覆數={total_replies}, 總請求次數={total_requests}")
        
        except Exception as e:
            logger.error(f"批次抓取貼文內容失敗：{str(e)}")
            rate_limit_info.append({"message": f"批次抓取貼文失敗：{str(e)}"})
        
        return {
            "results": results,
            "rate_limit_info": rate_limit_info,
            "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
            "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
            "rate_limit_until": 0,
            "total_posts": total_posts,
            "total_replies": total_replies
        }
