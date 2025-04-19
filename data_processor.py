import aiohttp
import random
import asyncio
import time
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from grok3_client import screen_thread_titles
from utils import clean_html
import streamlit.logger
from datetime import datetime

logger = streamlit.logger.get_logger(__name__)

async def process_user_question(user_question, cat_id_map, selected_cat, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None):
    category_ids = [cat_id_map[selected_cat]]
    data_type = analysis.get("data_type", "both")
    post_limit = min(analysis.get("post_limit", 2), 10)
    reply_limit = min(analysis.get("reply_limit", 75), 75)
    filters = analysis.get("filters", {})
    
    min_replies = filters.get("min_replies", 10)
    min_likes = 10 if cat_id_map[selected_cat] == 2 else filters.get("min_likes", 20)
    dislike_count_max = 20 if cat_id_map[selected_cat] == 2 else filters.get("dislike_count_max", 5)
    recent_only = False
    exclude_thread_ids = filters.get("exclude_thread_ids", [])
    
    candidate_thread_ids = analysis.get("candidate_thread_ids", [])
    top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else (previous_thread_ids or [])
    
    cat_id = cat_id_map[selected_cat]
    for cat_name, cat_id_val in cat_id_map.items():
        if cat_name in user_question:
            selected_cat = cat_name
            cat_id = cat_id_val
            break
    if cat_id not in category_ids:
        category_ids = [cat_id]
    
    thread_data = []
    rate_limit_info = []
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today.timestamp())
    
    if is_advanced and top_thread_ids:
        logger.info(f"進階分析：優先抓取第一次回應的帖子: top_thread_ids={top_thread_ids}")
        for thread_id in top_thread_ids:
            logger.info(f"開始抓取進階帖子回覆: thread_id={thread_id}")
            fetch_last_pages = 3 if is_advanced else 2
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit,
                fetch_last_pages=fetch_last_pages
            )
            
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"帖子無效或無權訪問: thread_id={thread_id}")
                continue
            
            sorted_replies = sorted(
                replies,
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:reply_limit]
            
            thread_data.append({
                "thread_id": str(thread_id),
                "title": thread_result.get("title", "未知標題"),
                "no_of_reply": thread_result.get("total_replies", 0),
                "last_reply_time": thread_result.get("last_reply_time", 0),
                "like_count": thread_result.get("like_count", 0),
                "dislike_count": thread_result.get("dislike_count", 0),
                "replies": [
                    {
                        "msg": clean_html(reply["msg"]),
                        "like_count": reply.get("like_count", 0),
                        "dislike_count": reply.get("dislike_count", 0),
                        "reply_time": reply.get("reply_time", 0)
                    }
                    for reply in sorted_replies
                ]
            })
            
            logger.info(f"進階帖子數據: thread_id={thread_id}, 回覆數={len(replies)}")
            await asyncio.sleep(5)
        
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    filtered_items = []
    total_fetched = 0
    for page in range(1, 4):
        result = await get_lihkg_topic_list(
            cat_id=cat_id,
            sub_cat_id=0,
            start_page=page,
            max_pages=1,
            request_counter=request_counter,
            last_reset=last_reset,
            rate_limit_until=rate_limit_until
        )
        
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(result.get("rate_limit_info", []))
        
        page_items = result.get("items", [])
        total_fetched += len(page_items)
        logger.info(f"成功抓取: cat_id={cat_id}, page={page}, 帖子數={len(page_items)}")
        
        if page_items:
            sample_item = page_items[0]
            logger.info(f"抓取數據樣本: thread_id={sample_item.get('thread_id')}, cat_id={sample_item.get('cat_id')}, title={sample_item.get('title')[:50]}...")
        
        filter_debug = {"min_replies_failed": 0, "min_likes_failed": 0, "dislike_count_failed": 0, "recent_only_failed": 0, "excluded_failed": 0}
        page_filtered_count = 0
        for item in page_items:
            thread_id = str(item.get("thread_id", ""))
            no_of_reply = item.get("no_of_reply", 0)
            like_count = int(item.get("like_count", 0))
            dislike_count = int(item.get("dislike_count", 0))
            last_reply_time = int(item.get("last_reply_time", 0))
            
            if not thread_id:
                logger.warning(f"帖子缺少 thread_id: {item}")
                continue
            if no_of_reply < min_replies:
                filter_debug["min_replies_failed"] += 1
                continue
            if like_count < min_likes:
                filter_debug["min_likes_failed"] += 1
                continue
            if dislike_count > dislike_count_max:
                filter_debug["dislike_count_failed"] += 1
                continue
            if recent_only and last_reply_time < today_timestamp:
                filter_debug["recent_only_failed"] += 1
                continue
            if thread_id in exclude_thread_ids:
                filter_debug["excluded_failed"] += 1
                continue
            filtered_items.append(item)
            page_filtered_count += 1
        
        logger.info(f"頁面 {page} 篩選: 總數={len(page_items)}, 符合條件={page_filtered_count}, "
                    f"篩選失敗詳情: 回覆數不足={filter_debug['min_replies_failed']}, "
                    f"點讚數不足={filter_debug['min_likes_failed']}, "
                    f"負評過多={filter_debug['dislike_count_failed']}, "
                    f"非近期帖子={filter_debug['recent_only_failed']}, "
                    f"被排除={filter_debug['excluded_failed']}")
        
        if len(filtered_items) >= 90:
            filtered_items = filtered_items[:90]
            break
    
    logger.info(f"總抓取: cat_id={cat_id}, 總抓取數={total_fetched}, 篩選後帖子數={len(filtered_items)}")
    
    if not filtered_items:
        logger.warning(f"無有效帖子: 問題={user_question}, 分類={selected_cat}, 篩選條件={filters}")
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": rate_limit_info + ["無有效帖子，篩選條件過嚴或數據缺失"],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }
    
    title_screening = await screen_thread_titles(
        user_query=user_question,
        thread_titles=filtered_items[:90],
        post_limit=post_limit
    )
    top_thread_ids = title_screening.get("top_thread_ids", [])
    need_replies = title_screening.get("need_replies", True)
    screening_reason = title_screening.get("reason", "未知")
    logger.info(f"Grok 3 標題篩選: top_thread_ids={top_thread_ids}, need_replies={need_replies}, 理由={screening_reason}")
    
    if not top_thread_ids and filtered_items:
        theme = analysis.get("theme", "熱門帖子")
        keywords = {
            "熱門帖子": ["熱門", "最多", "人氣", "討論"],
            "感動": ["溫馨", "感人", "互助", "愛", "緣分"],
            "搞笑": ["幽默", "搞亂", "on9", "爆笑"],
            "財經": ["股票", "投資", "經濟"],
            "時事": ["新聞", "政治", "事件"]
        }.get(theme, [])
        
        theme_items = [
            item for item in filtered_items
            if any(keyword in item["title"] for keyword in keywords) or not keywords
        ]
        if theme_items:
            top_thread_ids = [item["thread_id"] for item in random.sample(theme_items, min(post_limit, len(theme_items)))]
        else:
            top_thread_ids = [item["thread_id"] for item in random.sample(filtered_items, min(post_limit, len(filtered_items)))]
        logger.warning(f"標題篩選無結果，隨機選擇帖子: top_thread_ids={top_thread_ids}, 主題={theme}")
    
    candidate_data = {}
    candidate_threads = [
        item for item in filtered_items
        if str(item["thread_id"]) in map(str, top_thread_ids)
    ][:post_limit]
    
    if not candidate_threads and top_thread_ids:
        logger.warning(f"無帖子匹配top_thread_ids: {top_thread_ids}")
        candidate_threads = filtered_items[:post_limit]
    
    if not candidate_threads:
        candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items))) if filtered_items else []
        logger.info(f"無候選帖子，隨機選擇備用: 數量={len(candidate_threads)}")
    
    if need_replies:
        for item in candidate_threads:
            thread_id = str(item["thread_id"])
            logger.info(f"開始抓取候選帖子首頁回覆: thread_id={thread_id}")
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=25,
                fetch_last_pages=0
            )
            
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"帖子無效或無權訪問: thread_id={thread_id}")
                continue
            
            sorted_replies = sorted(
                replies,
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:25]
            
            candidate_data[thread_id] = {
                "thread_id": thread_id,
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [
                    {
                        "msg": clean_html(reply["msg"]),
                        "like_count": reply.get("like_count", 0),
                        "dislike_count": reply.get("dislike_count", 0),
                        "reply_time": reply.get("reply_time", 0)
                    }
                    for reply in sorted_replies
                ]
            }
            logger.info(f"抓取候選回覆: thread_id={thread_id}, 回覆數={len(replies)}")
    
    final_threads = [
        item for item in filtered_items
        if str(item["thread_id"]) in map(str, top_thread_ids)
    ][:post_limit]
    
    if not final_threads:
        final_threads = candidate_threads[:post_limit]
        logger.info(f"無匹配top_thread_ids，使用候選帖子: 數量={len(final_threads)}")
    
    for item in final_threads:
        thread_id = str(item["thread_id"])
        logger.info(f"開始抓取最終帖子回覆: thread_id={thread_id}")
        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id,
            cat_id=cat_id,
            request_counter=request_counter,
            last_reset=last_reset,
            rate_limit_until=rate_limit_until,
            max_replies=reply_limit,
            fetch_last_pages=2
        )
        
        request_counter = thread_result.get("request_counter", request_counter)
        last_reset = thread_result.get("last_reset", last_reset)
        rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(thread_result.get("rate_limit_info", []))
        
        replies = thread_result.get("replies", [])
        if not replies and thread_result.get("total_replies", 0) >= min_replies:
            logger.warning(f"帖子無效或無權訪問: thread_id={thread_id}")
            continue
        
        sorted_replies = sorted(
            replies,
            key=lambda x: x.get("like_count", 0),
            reverse=True
        )[:reply_limit]
        
        thread_data.append({
            "thread_id": thread_id,
            "title": item["title"],
            "no_of_reply": item.get("no_of_reply", 0),
            "last_reply_time": item.get("last_reply_time", 0),
            "like_count": item.get("like_count", 0),
            "dislike_count": item.get("dislike_count", 0),
            "replies": [
                {
                    "msg": clean_html(reply["msg"]),
                    "like_count": reply.get("like_count", 0),
                    "dislike_count": reply.get("dislike_count", 0),
                    "reply_time": reply.get("reply_time", 0)
                }
                for reply in sorted_replies
            ]
        })
        
        logger.info(f"最終帖子數據: thread_id={thread_id}, 回覆數={len(replies)}")
        await asyncio.sleep(5)
    
    if not thread_data:
        logger.warning(f"最終無有效帖子: 問題={user_question}, 分類={selected_cat}, "
                      f"篩選條件={filters}, candidate_thread_ids={candidate_thread_ids}, top_thread_ids={top_thread_ids}")
        rate_limit_info.append("最終無有效帖子，可能由於篩選條件過嚴或數據缺失")
    
    return {
        "selected_cat": selected_cat,
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }
