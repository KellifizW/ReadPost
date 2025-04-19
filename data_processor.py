import asyncio
import logging
from typing import Dict, List, Any, Set, Tuple
from lihkg_api import get_lihkg_threads, get_lihkg_thread_content

logger = logging.getLogger(__name__)

async def process_user_question(
    user_question: str,
    cat_id: int,
    cat_name: str,
    analysis: Dict[str, Any],
    used_thread_ids: Set[str],
    request_counter: int,
    last_reset: float,
    rate_limit_until: float,
    fetch_last_pages: int = 1,
    max_replies: int = 60,
    fetch_remaining_pages: bool = False
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    post_limit = analysis.get("post_limit", 5)
    category_ids = analysis.get("category_ids", [cat_id])
    filters = analysis.get("filters", {
        "min_replies": 10,
        "min_likes": 5,
        "dislike_count_max": 50
    })
    
    # Stage 1: Fetch thread titles
    all_threads = []
    for cid in category_ids:
        for page in range(1, 4):  # Fetch 3 pages per category
            threads = await get_lihkg_threads(
                cat_id=cid,
                page=page,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until
            )
            logger.info(f"成功抓取: cat_id={cid}, page={page}, 帖子數={len(threads)}")
            all_threads.extend(threads)
    
    logger.info(f"初始抓取: cat_id={cat_id}, 帖子數={len(all_threads)}")
    
    # Stage 2: Filter candidate threads
    candidate_threads = [
        t for t in all_threads
        if t.get("no_of_reply", 0) >= filters["min_replies"]
        and t.get("like_count", 0) >= filters["min_likes"]
        and t.get("dislike_count", 0) <= filters["dislike_count_max"]
        and str(t["thread_id"]) not in used_thread_ids
    ]
    
    logger.info(f"篩選候選帖子: 總數={len(all_threads)}, 符合條件={len(candidate_threads)}")
    
    # Stage 3: Fetch candidate thread replies
    candidate_thread_ids = analysis.get("candidate_thread_ids", [])
    if candidate_thread_ids and all(str(tid) in [t["thread_id"] for t in candidate_threads] for tid in candidate_thread_ids):
        final_candidates = [t for t in candidate_threads if str(t["thread_id"]) in candidate_thread_ids]
    else:
        final_candidates = candidate_threads[:10]  # Fallback to top 10
    
    for thread in final_candidates:
        thread_result = await get_lihkg_thread_content(
            thread_id=thread["thread_id"],
            cat_id=cat_id,
            request_counter=request_counter,
            last_reset=last_reset,
            rate_limit_until=rate_limit_until,
            max_replies=25,
            fetch_last_pages=0
        )
        thread.update(thread_result)
        logger.info(f"抓取候選回覆: thread_id={thread['thread_id']}, 回覆數={len(thread_result.get('replies', []))}")
    
    # Stage 4: Select final threads and fetch replies
    top_thread_ids = analysis.get("top_thread_ids", [])
    if top_thread_ids and all(str(tid) in [t["thread_id"] for t in final_candidates] for tid in top_thread_ids):
        final_threads = [t for t in final_candidates if str(t["thread_id"]) in top_thread_ids][:post_limit]
    else:
        final_threads = final_candidates[:post_limit]
    
    logger.info(f"最終帖子抓取: post_limit={post_limit}, fetch_last_pages={fetch_last_pages}")
    
    for thread in final_threads:
        if fetch_remaining_pages:
            # Fetch remaining pages not fetched in initial analysis
            already_fetched_pages = thread.get("fetched_pages", [])
            total_pages = (thread.get("no_of_reply", 0) // 25) + 1
            remaining_pages = [p for p in range(1, total_pages + 1) if p not in already_fetched_pages]
            if remaining_pages:
                thread_result = await get_lihkg_thread_content(
                    thread_id=thread["thread_id"],
                    cat_id=cat_id,
                    request_counter=request_counter,
                    last_reset=last_reset,
                    rate_limit_until=rate_limit_until,
                    max_replies=100,
                    specific_pages=remaining_pages[:4]  # Limit to 4 pages
                )
                thread.update(thread_result)
                logger.info(f"最終帖子數據: thread_id={thread['thread_id']}, 回覆數={len(thread_result.get('replies', []))}")
        else:
            # Fetch first and last pages
            thread_result = await get_lihkg_thread_content(
                thread_id=thread["thread_id"],
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=max_replies,
                fetch_last_pages=fetch_last_pages
            )
            thread.update(thread_result)
            logger.info(f"最終帖子數據: thread_id={thread['thread_id']}, 回覆數={len(thread_result.get('replies', []))}")
    
    used_thread_ids = set(str(thread["thread_id"]) for thread in final_threads)
    return final_threads, used_thread_ids
