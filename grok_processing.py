import aiohttp
import asyncio
import json
import re
import datetime
import time
import logging
import streamlit as st
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from reddit_api import get_reddit_topic_list, get_reddit_thread_content
from logging_config import configure_logger
from dynamic_prompt_utils import build_dynamic_prompt, parse_query, extract_keywords

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "grok_processing.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 120000
API_TIMEOUT = 90
MAX_CACHE_SIZE = 100

cache_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(5)

async def analyze_and_screen(user_query, source_name, source_id, source_type="reddit", conversation_context=None):
    conversation_context = conversation_context or []
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {
            "direct_response": True,
            "intents": [{"intent": "general_query", "confidence": 0.5, "reason": "缺少 API 密鑰"}],
            "theme": "一般",
            "source_type": source_type,
            "source_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "filters": {},
            "processing": {"intents": ["general_query"]},
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "缺少 API 密鑰",
            "theme_keywords": []
        }
    
    logger.info(f"開始語義分析：查詢={user_query}")
    parsed_query = await parse_query(user_query, conversation_context, GROK3_API_KEY, source_type)
    intents = parsed_query["intents"]
    query_keywords = parsed_query["keywords"]
    top_thread_ids = parsed_query["thread_ids"]
    reason = parsed_query["reason"]
    confidence = parsed_query["confidence"]
    
    if not intents:
        logger.warning(f"意圖分析失敗：查詢={user_query}, 原因=無法識別有效意圖，回退到默認意圖 contextual_analysis")
        intents = [{"intent": "contextual_analysis", "confidence": 0.5, "reason": "無法識別有效意圖，回退到默認意圖"}]
        reason = "無法識別有效意圖，回退到默認意圖"
        confidence = 0.5
    
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "一般")
    historical_keywords = context_summary.get("keywords", [])
    
    has_high_confidence_list_titles = any(i["intent"] == "list_titles" and i["confidence"] >= 0.9 for i in intents)
    is_vague = len(query_keywords) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題"]) and not has_high_confidence_list_titles
    
    if is_vague and historical_theme != "一般":
        intents = [{"intent": "contextual_analysis", "confidence": 0.7, "reason": f"問題模糊，延續歷史主題：{historical_theme}"}]
        reason = f"問題模糊，延續歷史主題：{historical_theme}"
    elif is_vague:
        intents = [{"intent": "contextual_analysis", "confidence": 0.7, "reason": "問題模糊，默認總結帖子"}]
        reason = "問題模糊，默認總結帖子"
    
    theme = historical_theme if is_vague else (query_keywords[0] if query_keywords else "一般")
    theme_keywords = list(historical_keywords if is_vague else query_keywords)
    
    post_limit = 15 if any(i["intent"] == "list_titles" for i in intents) else 20
    data_type = "both" if not all(i["intent"] in ["general_query"] for i in intents) else "none"
    
    if any(i["intent"] == "follow_up" for i in intents):
        post_limit = len(top_thread_ids) or 5
        data_type = "replies"
    
    logger.info(f"語義分析結果：intents={[i['intent'] for i in intents]}, confidence={confidence}, reason={reason}")
    return {
        "direct_response": all(i["intent"] in ["general_query"] for i in intents),
        "intents": intents,
        "theme": theme,
        "source_type": source_type,
        "source_ids": [source_id],
        "data_type": data_type,
        "post_limit": post_limit,
        "filters": {"min_replies": 10, "min_likes": 0, "sort": "popular", "keywords": theme_keywords},
        "processing": {"intents": [i["intent"] for i in intents], "top_thread_ids": top_thread_ids},
        "candidate_thread_ids": top_thread_ids,
        "top_thread_ids": top_thread_ids,
        "needs_advanced_analysis": confidence < 0.7,
        "reason": reason,
        "theme_keywords": theme_keywords
    }

async def prioritize_threads_with_grok(user_query, threads, source_name, source_id, source_type="reddit", intents=["contextual_analysis"]):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {"top_thread_ids": [], "reason": "缺少 API 密鑰", "intent_breakdown": []}

    if any(intent == "follow_up" for intent in intents):
        referenced_thread_ids = []
        context = st.session_state.get("conversation_context", [])
        if context:
            last_response = context[-1].get("content", "")
            matches = re.findall(r"\[帖子 ID: [a-zA-Z0-9]+\]", last_response)
            referenced_thread_ids = [tid.strip("[]").split(": ")[1] for tid in matches if any(str(t["thread_id"]) == tid.strip("[]").split(": ")[1] for t in threads)]
        if referenced_thread_ids:
            logger.info(f"追問檢測到參考帖子：thread_ids={referenced_thread_ids}")
            return {
                "top_thread_ids": referenced_thread_ids[:5],
                "reason": "使用追問的參考帖子 ID",
                "intent_breakdown": [{"intent": "follow_up", "thread_ids": referenced_thread_ids[:5]}]
            }

    threads = [{"thread_id": str(t["thread_id"]), **t} for t in threads]
    prompt = f"""
你是帖子優先級排序助手，請根據用戶查詢和多個意圖，從提供的帖子中選出最多20個最相關的帖子。
查詢：{user_query}
意圖：{json.dumps(intents, ensure_ascii=False)}
討論區：{source_name} (ID: {source_id})
來源類型：{source_type}
帖子數據：
{json.dumps([{"thread_id": str(t["thread_id"]), "title": clean_html(t["title"]), "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0), "last_reply_time": t.get("last_reply_time", 0)} for t in threads], ensure_ascii=False)}
排序標準：
- 語義相關性（50%）：標題和查詢的語義相似度。
- 熱門度（30%）：回覆數（權重0.6）+點讚數（權重0.4）。
- 時間（20%）：若意圖含 time_sensitive_analysis，最近24小時的帖子（last_reply_time）優先級提升50%。
輸出格式：{{
  "top_thread_ids": ["id1", "id2", ...],
  "reason": "排序原因",
  "intent_breakdown": [
    {{"intent": "意圖1", "thread_ids": ["id1", "id2"]}},
    ...
  ]
}}
reason 簡潔，不超50字。
"""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code != 200:
                        logger.debug(f"帖子優先級排序失敗：狀態碼={status_code}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
                        logger.debug(f"帖子優先級排序失敗：缺少有效回應，嘗試次數={attempt + 1}")
                        continue
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    response_content = data["choices"][0]["message"]["content"]
                    logger.info(
                        f"Grok3 API 調用：函數=prioritize_threads_with_grok, "
                        f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                        f"輸出 token={completion_tokens}, 回應={response_content}, "
                        f"提示長度={len(prompt)} 字符"
                    )
                    try:
                        result = json.loads(response_content)
                        top_thread_ids = result.get("top_thread_ids", [])
                        reason = result.get("reason", "無排序原因")
                        intent_breakdown = result.get("intent_breakdown", [])
                        top_thread_ids = [str(tid) for tid in top_thread_ids]
                        thread_id_set = {str(t["thread_id"]) for t in threads}
                        valid_thread_ids = list(dict.fromkeys([tid for tid in top_thread_ids if tid in thread_id_set]))
                        invalid_thread_ids = [tid for tid in top_thread_ids if tid not in thread_id_set]
                        logger.info(
                            f"thread_id 驗證：有效 thread_ids 數量={len(valid_thread_ids)}, "
                            f"無效 thread_ids 數量={len(invalid_thread_ids)}"
                        )
                        logger.info(f"成功解析 JSON，回應包含 {len(valid_thread_ids)} 個有效 thread_ids")
                        return {
                            "top_thread_ids": valid_thread_ids,
                            "reason": reason,
                            "intent_breakdown": intent_breakdown
                        }
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"無法解析優先級排序結果：原始回應={response_content}, "
                            f"token 使用：輸入={prompt_tokens}, 輸出={completion_tokens}, 錯誤：{str(e)}"
                        )
                        try:
                            match = re.search(r'"top_thread_ids":\s*\[(.*?)\]', response_content, re.DOTALL)
                            if match:
                                ids_str = match.group(1)
                                ids = [str(id.strip().strip('"')) for id in ids_str.split(',') if id.strip()]
                                thread_id_set = {str(t["thread_id"]) for t in threads}
                                valid_thread_ids = list(dict.fromkeys([tid for tid in ids if tid in thread_id_set]))
                                invalid_thread_ids = [tid for tid in ids if tid not in thread_id_set]
                                logger.info(
                                    f"從不完整回應提取：有效 thread_ids 數量={len(valid_thread_ids)}, "
                                    f"無效 thread_ids 數量={len(invalid_thread_ids)}"
                                )
                                if valid_thread_ids:
                                    logger.info(f"成功從不完整回應中提取 {len(valid_thread_ids)} 個 thread_ids")
                                    return {
                                        "top_thread_ids": valid_thread_ids,
                                        "reason": "從不完整回應中提取的 thread_ids",
                                        "intent_breakdown": []
                                    }
                        except Exception as extract_e:
                            logger.warning(f"無法從不完整回應中提取 thread_ids：{str(extract_e)}")
                        return {"top_thread_ids": [], "reason": f"無法解析 API 回應：{str(e)}", "intent_breakdown": []}
        except Exception as e:
            logger.debug(f"帖子優先級排序錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            sorted_threads = sorted(
                threads,
                key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                reverse=True
            )
            valid_thread_ids = [str(t["thread_id"]) for t in sorted_threads[:20]]
            logger.info(f"API 調用失敗，回退到熱門度排序，返回 {len(valid_thread_ids)} 個 thread_ids")
            return {
                "top_thread_ids": valid_thread_ids,
                "reason": f"優先級排序失敗（{str(e)}），回退到熱門度排序",
                "intent_breakdown": []
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_source, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, source_id=None, source_type="reddit"):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 10, "min_likes": 0}
    
    if isinstance(selected_source, str):
        if "Reddit" in selected_source:
            source_name = selected_source.replace("Reddit - ", "").strip()
            source_type = "reddit"
        elif "LIHKG" in selected_source:
            source_name = selected_source.replace("LIHKG - ", "").strip()
            source_type = "lihkg"
        else:
            source_name = selected_source
            source_type = source_type or "reddit"
        selected_source = {"source_name": source_name, "source_type": source_type}
    elif not isinstance(selected_source, dict):
        logger.error(f"無效的 selected_source 格式：{type(selected_source)}")
        yield f"錯誤：無效的討論區格式（{type(selected_source)}）。請聯繫支持。"
        return
    
    if not thread_data:
        error_msg = f"在 {selected_source['source_name']} 中未找到符合條件的帖子（篩選：{json.dumps(filters)}）。建議嘗試其他關鍵詞或討論區！"
        logger.warning(f"無匹配帖子：{error_msg}")
        yield error_msg
        return

    if not isinstance(processing, dict):
        logger.error(f"無效的處理數據格式：預期 dict，得到 {type(processing)}")
        yield f"錯誤：無效的處理數據格式（{type(processing)}）。請聯繫支持。"
        return
    intents = processing.get('intents', ['contextual_analysis'])

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        yield "錯誤：缺少 API 密鑰"
        return
    
    intent_word_ranges = {
        "contextual_analysis": (700, 2800),
        "semantic_query": (700, 2800),
        "time_sensitive_analysis": (700, 2800),
        "follow_up": (1000, 5000),
        "fetch_thread_by_id": (700, 2800),
        "list_titles": (140, 400)
    }
    
    total_min_tokens = 0
    total_max_tokens = 0
    for intent in intents:
        word_min, word_max = intent_word_ranges.get(intent, (700, 2800))
        total_min_tokens += int(word_min / 0.8)
        total_max_tokens += int(word_max / 0.8)
    
    prompt_length = len(json.dumps(thread_data, ensure_ascii=False)) + len(user_query) + 1000
    length_factor = min(prompt_length / GROK3_TOKEN_LIMIT, 1.0)
    target_tokens = total_min_tokens + (total_max_tokens - total_min_tokens) * length_factor
    target_tokens = min(max(int(target_tokens), total_min_tokens), total_max_tokens)

    max_tokens_limit = 8000
    max_tokens = min(target_tokens + 500, max_tokens_limit)
    max_replies_per_thread = 300 if any(intent == "follow_up" for intent in intents) else 100
    max_comments = 300 if source_type == "reddit" and any(intent == "follow_up" for intent in intents) else 100

    if any(intent == "list_titles" for intent in intents):
        max_replies_per_thread = 10

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    
    if any(intent in ["follow_up", "fetch_thread_by_id"] for intent in intents):
        reply_count_prompt = f"""
你是資料抓取助手，請根據問題和意圖決定每個帖子應下載的回覆數量（100、200、250、300、500 條）。
僅以 JSON 格式回應，禁止生成自然語言或其他格式的內容。
問題：{user_query}
意圖：{json.dumps(intents, ensure_ascii=False)}
若問題需要深入分析（如追問、特定帖子ID），建議較多回覆（200-500）。
若意圖含 time_sensitive_analysis，優先最新回覆，建議100-250條。
默認：100 條。
輸出格式：{{"replies_per_thread": 100, "reason": "決定原因"}}
"""
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": reply_count_prompt}],
            "max_tokens": 100,
            "temperature": 0.5
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code == 200:
                        data = await response.json()
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        response_content = data["choices"][0]["message"]["content"]
                        logger.info(
                            f"Grok3 API 調用：函數=stream_grok3_response (reply_count), "
                            f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                            f"輸出 token={completion_tokens}, 回應={response_content}, "
                            f"提示長度={len(reply_count_prompt)} 字符"
                        )
                        try:
                            result = json.loads(response_content)
                            max_replies_per_thread = min(result.get("replies_per_thread", 100), 500)
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON 解析失敗：{str(e)}，原始回應={response_content}")
                    else:
                        logger.warning(f"無法確定每帖子回覆數，狀態碼={status_code}")
        except Exception as e:
            logger.warning(f"每帖子回覆數選擇失敗：{str(e)}")

    thread_data_dict = {}
    if isinstance(thread_data, list):
        thread_data_dict = {str(data["thread_id"]): data for data in thread_data if isinstance(data, dict) and "thread_id" in data}
    elif isinstance(thread_data, dict):
        thread_data_dict = thread_data
    else:
        logger.error(f"無效的 thread_data 格式：預期 list 或 dict，得到 {type(thread_data)}")
        yield f"錯誤：無效的 thread_data 格式（{type(thread_data)}）。請聯繫支持。"
        return
    
    if any(intent in ["follow_up", "fetch_thread_by_id"] for intent in intents):
        referenced_thread_ids = []
        if any(intent == "follow_up" for intent in intents):
            referenced_thread_ids = processing.get("top_thread_ids", [])
            if not referenced_thread_ids:
                last_response = conversation_context[-1].get("content", "") if conversation_context else ""
                matches = re.findall(r"\[帖子 ID: [a-zA-Z0-9]+\]", last_response)
                referenced_thread_ids = [tid.strip("[]").split(": ")[1] for tid in matches if str(tid.strip("[]").split(": ")[1]) in thread_data_dict]
        else:
            top_thread_ids = processing.get("top_thread_ids", [])
            referenced_thread_ids = [tid for tid in top_thread_ids if str(tid) in thread_data_dict]
        
        if not referenced_thread_ids and any(intent == "follow_up" for intent in intents):
            keyword_result = await extract_keywords(user_query, conversation_context, GROK3_API_KEY)
            theme_keywords = keyword_result["keywords"]
            async with request_semaphore:
                if source_type == "lihkg":
                    supplemental_result = await get_lihkg_topic_list(
                        cat_id=source_id,
                        start_page=1,
                        max_pages=2
                    )
                else:
                    supplemental_result = await get_reddit_topic_list(
                        subreddit=source_id,
                        start_page=1,
                        max_pages=2
                    )
            supplemental_threads = supplemental_result.get("items", [])
            filtered_supplemental = [
                item for item in supplemental_threads
                if any(kw.lower() in item["title"].lower() for kw in theme_keywords)
            ][:5]
            referenced_thread_ids = [str(item["thread_id"]) for item in filtered_supplemental]
        
        prioritized_thread_data = {tid: thread_data_dict[tid] for tid in map(str, referenced_thread_ids) if tid in thread_data_dict}
        supplemental_thread_data = {tid: data for tid, data in thread_data_dict.items() if tid not in map(str, referenced_thread_ids)}
        thread_data_dict = {**prioritized_thread_data, **supplemental_thread_data}

    filtered_thread_data = {}
    total_replies_count = 0
    
    for tid, data in thread_data_dict.items():
        try:
            replies = data.get("replies", [])
            if not isinstance(replies, list):
                logger.warning(f"無效的回覆格式，帖子 ID={tid}：預期 list，得到 {type(replies)}")
                replies = []
            filtered_replies = []
            for r in replies:
                if not isinstance(r, dict) or not r.get("msg"):
                    continue
                cleaned_msg = clean_html(r["msg"])
                if len(cleaned_msg.strip()) <= 7 or cleaned_msg in ["[圖片]", "[無內容]", "[表情符號]"]:
                    continue
                filtered_replies.append({
                    "reply_id": r.get("reply_id"),
                    "msg": cleaned_msg,
                    "like_count": r.get("like_count", 0),
                    "reply_time": r.get("reply_time", 0),
                    "dislike_count": r.get("dislike_count", 0) if source_type == "lihkg" else 0
                })
            
            sorted_replies = sorted(
                filtered_replies,
                key=lambda x: (
                    x.get("like_count", 0) * 0.3 +
                    (1 if any(intent == "time_sensitive_analysis" for intent in intents) and isinstance(x.get("reply_time", 0), (int, float)) and x.get("reply_time", 0) >= time.time() - 86400 else 0) * 0.4 +
                    0.3
                ),
                reverse=True
            )[:50]
            
            clustered_replies = []
            seen_msgs = set()
            for reply in sorted_replies:
                msg = reply["msg"]
                is_unique = True
                for seen_msg in seen_msgs:
                    if len(set(msg.split()) & set(seen_msg.split())) / len(set(msg.split()) | set(seen_msg.split())) > 0.7:
                        is_unique = False
                        break
                if is_unique:
                    seen_msgs.add(msg)
                    clustered_replies.append(reply)
                if len(clustered_replies) >= max_replies_per_thread:
                    break
            
            total_replies_count += len(clustered_replies)
            filtered_thread_data[tid] = {
                "thread_id": data.get("thread_id", tid),
                "title": data.get("title", ""),
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": clustered_replies,
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(clustered_replies)
            }
        except Exception as e:
            logger.error(f"處理帖子 ID={tid} 失敗：{str(e)}")
            yield f"錯誤：處理帖子（ID={tid}）失敗（{str(e)}）。請聯繫支持。"
            return
    
    if total_replies_count < max_replies_per_thread and any(intent in ["follow_up", "fetch_thread_by_id"] for intent in intents):
        for tid, data in filtered_thread_data.items():
            if data["total_fetched_replies"] < max_replies_per_thread:
                async with request_semaphore:
                    if source_type == "lihkg":
                        content_result = await get_lihkg_thread_content(
                            thread_id=tid,
                            cat_id=source_id,
                            max_replies=max_replies_per_thread - data["total_fetched_replies"],
                            fetch_last_pages=1,
                            start_page=max(data["fetched_pages"], default=0) + 1
                        )
                    else:
                        content_result = await get_reddit_thread_content(
                            post_id=tid,
                            subreddit=source_id,
                            max_comments=max_comments
                        )
                if content_result.get("replies"):
                    total_replies = content_result.get("total_replies", data["no_of_reply"])
                    cleaned_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "reply_time": reply.get("reply_time", 0)
                        }
                        for reply in content_result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                        and len(clean_html(reply.get("msg")).strip()) > 7
                    ]
                    filtered_additional_replies = [
                        r for r in cleaned_replies
                        if len(r["msg"].strip()) > 7
                    ]
                    updated_data = {
                        "thread_id": data.get("thread_id", tid),
                        "title": data.get("title", ""),
                        "no_of_reply": total_replies,
                        "last_reply_time": content_result.get("last_reply_time", data["last_reply_time"]),
                        "like_count": data.get("like_count", 0),
                        "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                        "replies": data.get("replies", []) + filtered_additional_replies,
                        "fetched_pages": list(set(data.get("fetched_pages", []) + content_result.get("fetched_pages", []))),
                        "total_fetched_replies": len(data.get("replies", []) + filtered_additional_replies)
                    }
                    filtered_thread_data[tid] = updated_data
                    total_replies_count += len(filtered_additional_replies)
                    async with cache_lock:
                        st.session_state.thread_cache[tid] = {
                            "data": updated_data,
                            "timestamp": time.time()
                        }

    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata:
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": [],
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": 0
            } for tid, data in filtered_thread_data.items()
        }
        total_replies_count = 0
    
    prompt = await build_dynamic_prompt(
        query=user_query,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=list(filtered_thread_data.values()),
        filters=filters,
        intent=intents[0],
        selected_source=selected_source,
        grok3_api_key=GROK3_API_KEY
    )
    
    prompt_length = len(prompt)
    estimated_tokens = prompt_length // 4
    logger.info(
        f"生成提示：函數=stream_grok3_response, 查詢={user_query}, "
        f"提示長度={prompt_length} 字符, 估計 token={estimated_tokens}, "
        f"thread_data 帖子數={len(filtered_thread_data)}, 總回覆數={total_replies_count}"
    )
    
    if prompt_length > GROK3_TOKEN_LIMIT:
        logger.warning(f"提示長度超過限制：初始長度={prompt_length} > {GROK3_TOKEN_LIMIT}，縮減數據")
        max_replies_per_thread = max_replies_per_thread // 2
        total_replies_count = 0
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": data["replies"][:max_replies_per_thread],
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(data["replies"][:max_replies_per_thread])
            } for tid, data in filtered_thread_data.items()
        }
        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
        prompt = await build_dynamic_prompt(
            query=user_query,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=list(filtered_thread_data.values()),
            filters=filters,
            intent=intents[0],
            selected_source=selected_source,
            grok3_api_key=GROK3_API_KEY
        )
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4
        logger.info(
            f"縮減後提示：函數=stream_grok3_response, 查詢={user_query}, "
            f"提示長度={prompt_length} 字符, 估計 token={estimated_tokens}, "
            f"thread_data 帖子數={len(filtered_thread_data)}, 總回覆數={total_replies_count}"
        )
        target_tokens = total_min_tokens + (total_replies_count / 500) * (total_max_tokens - total_min_tokens) * 0.9
        target_tokens = min(max(int(target_tokens), total_min_tokens), max_tokens_limit)
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": (
            "你是社交媒體討論區（包括 Reddit 和 LIHKG）的數據助手，以繁體中文回答，"
            "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
            "禁止使用 [post_id: ...] 格式。根據用戶意圖動態選擇回應格式（例如段落、表格），"
            "確保結構清晰、內容連貫、不重複，且適合查詢的需求。"
        )},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }
    
    response_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                status_code = response.status
                if status_code != 200:
                    logger.error(f"回應生成失敗：狀態碼={status_code}")
                    yield f"錯誤：生成回應失敗（狀態碼 {status_code}）。請稍後重試。"
                    return
                
                async for line in response.content:
                    if line and not line.isspace():
                        line_str = line.decode('utf-8').strip()
                        if line_str == "data: [DONE]":
                            logger.info(
                                f"Grok3 API 調用：函數=stream_grok3_response, "
                                f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                                f"輸出 token={completion_tokens}, 回應長度={len(response_content)}"
                            )
                            break
                        if line_str.startswith("data:"):
                            try:
                                chunk = json.loads(line_str[6:])
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    if "Content Moderation" in content or "Blocked" in content:
                                        logger.warning(f"檢測到內容審核：{content}")
                                        raise ValueError("檢測到內容審核")
                                    cleaned_content = clean_response(content)
                                    response_content += cleaned_content
                                    yield cleaned_content
                                usage = chunk.get("usage", {})
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                            except json.JSONDecodeError:
                                logger.warning(f"流式數據 JSON 解碼錯誤")
                                continue
        except Exception as e:
            logger.error(f"回應生成失敗：{str(e)}")
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
        finally:
            await session.close()

async def process_user_question(user_query, selected_source, source_id, source_type="reddit", analysis=None, request_counter=0, last_reset=0, rate_limit_until=0, conversation_context=None, progress_callback=None):
    if source_type == "lihkg":
        configure_lihkg_api_logger()
    else:
        configure_reddit_api_logger()
    
    if isinstance(selected_source, str):
        if "Reddit" in selected_source:
            source_name = selected_source.replace("Reddit - ", "").strip()
            source_type = "reddit"
        elif "LIHKG" in selected_source:
            source_name = selected_source.replace("LIHKG - ", "").strip()
            source_type = "lihkg"
        else:
            source_name = selected_source
            source_type = source_type or "reddit"
        selected_source = {"source_name": source_name, "source_type": source_type}
    elif not isinstance(selected_source, dict):
        logger.error(f"無效的 selected_source 格式：{type(selected_source)}")
        return {
            "selected_source": {"source_name": "未知", "source_type": source_type},
            "thread_data": [],
            "rate_limit_info": [{"message": f"無效的討論區格式：{type(selected_source)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    try:
        clean_cache()
        
        if rate_limit_until > time.time():
            logger.warning(f"速率限制生效，直到 {datetime.datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
            return {
                "selected_source": selected_source,
                "thread_data": [],
                "rate_limit_info": [{"message": "速率限制生效", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        try:
            GROK3_API_KEY = st.secrets["grok3key"]
        except KeyError as e:
            logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
            return {
                "selected_source": selected_source,
                "thread_data": [],
                "rate_limit_info": [{"message": "缺少 API 密鑰"}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        if not analysis:
            analysis = await analyze_and_screen(user_query, selected_source["source_name"], source_id, source_type, conversation_context)
        
        post_limit = min(analysis.get("post_limit", 20), 20)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 10)
        top_thread_ids = list(set(analysis.get("top_thread_ids", [])))
        intents = analysis.get("intents", [{"intent": "contextual_analysis", "confidence": 0.5, "reason": "默認意圖"}])
        
        logger.info(f"處理用戶問題：intents={[i['intent'] for i in intents]}, source_type={source_type}, source_id={source_id}, post_limit={post_limit}, top_thread_ids={top_thread_ids}")
        
        keyword_result = await extract_keywords(user_query, conversation_context, GROK3_API_KEY)
        fetch_last_pages = 1 if keyword_result.get("time_sensitive", False) else 0
        
        max_comments = 300 if source_type == "reddit" and any(i["intent"] == "follow_up" for i in intents) else 100
        max_replies = 300 if any(i["intent"] == "follow_up" for i in intents) else 100
        
        if any(i["intent"] in ["fetch_thread_by_id", "follow_up"] for i in intents) and top_thread_ids:
            thread_data = []
            rate_limit_info = []
            processed_thread_ids = set()
            
            candidate_threads = [{"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0} for tid in top_thread_ids]
            
            tasks = []
            for idx, thread_id in enumerate(top_thread_ids):
                thread_id_str = str(thread_id)
                if thread_id_str in processed_thread_ids:
                    logger.debug(f"跳過重複的 thread_id={thread_id_str}")
                    continue
                processed_thread_ids.add(thread_id_str)
                async with cache_lock:
                    if thread_id_str in st.session_state.thread_cache and st.session_state.thread_cache[thread_id_str]["data"].get("replies"):
                        cached_data = st.session_state.thread_cache[thread_id_str]["data"]
                        thread_data.append(cached_data)
                        logger.debug(f"緩存命中：thread_id={thread_id_str}")
                        continue
                if source_type == "lihkg":
                    tasks.append(get_lihkg_thread_content(
                        thread_id=thread_id_str,
                        cat_id=source_id,
                        max_replies=max_replies,
                        fetch_last_pages=fetch_last_pages,
                        specific_pages=[],
                        start_page=1
                    ))
                else:
                    tasks.append(get_reddit_thread_content(
                        post_id=thread_id_str,
                        subreddit=source_id,
                        max_comments=max_comments
                    ))
            
            if tasks:
                content_results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(content_results):
                    if isinstance(result, Exception):
                        logger.warning(f"無法抓取帖子 {candidate_threads[idx]['thread_id']}：{str(result)}")
                        continue
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    
                    thread_id = str(candidate_threads[idx]["thread_id"])
                    if result.get("title"):
                        total_replies = result.get("total_replies", candidate_threads[idx]["no_of_reply"])
                        if total_replies == 0:
                            total_replies = candidate_threads[idx]["no_of_reply"]
                        filtered_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "reply_time": reply.get("reply_time", 0)
                            }
                            for reply in result.get("replies", [])
                            if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                            and len(clean_html(reply.get("msg")).strip()) > 7
                        ]
                        thread_info = {
                            "thread_id": thread_id,
                            "title": result.get("title"),
                            "no_of_reply": total_replies,
                            "last_reply_time": result.get("last_reply_time", 0),
                            "like_count": result.get("like_count", 0),
                            "dislike_count": result.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": filtered_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(filtered_replies)
                        }
                        thread_data.append(thread_info)
                        async with cache_lock:
                            st.session_state.thread_cache[thread_id] = {
                                "data": thread_info,
                                "timestamp": time.time()
                            }
            
            if len(thread_data) < 5 and any(i["intent"] == "follow_up" for i in intents):
                keyword_result = await extract_keywords(user_query, conversation_context, GROK3_API_KEY)
                theme_keywords = keyword_result["keywords"]
                
                async with request_semaphore:
                    if source_type == "lihkg":
                        supplemental_result = await get_lihkg_topic_list(
                            cat_id=source_id,
                            start_page=1,
                            max_pages=2
                        )
                    else:
                        supplemental_result = await get_reddit_topic_list(
                            subreddit=source_id,
                            start_page=1,
                            max_pages=2
                        )
                supplemental_threads = supplemental_result.get("items", [])
                filtered_supplemental = [
                    item for item in supplemental_threads
                    if str(item["thread_id"]) not in top_thread_ids
                    and any(kw.lower() in item["title"].lower() for kw in theme_keywords)
                ][:5 - len(thread_data)]
                request_counter = supplemental_result.get("request_counter", request_counter)
                last_reset = supplemental_result.get("last_reset", last_reset)
                rate_limit_until = supplemental_result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(supplemental_result.get("rate_limit_info", []))
                
                supplemental_tasks = []
                for item in filtered_supplemental:
                    thread_id = str(item["thread_id"])
                    if thread_id in processed_thread_ids:
                        logger.debug(f"跳過重複的補充 thread_id={thread_id}")
                        continue
                    processed_thread_ids.add(thread_id)
                    if source_type == "lihkg":
                        supplemental_tasks.append(get_lihkg_thread_content(
                            thread_id=thread_id,
                            cat_id=source_id,
                            max_replies=max_replies,
                            fetch_last_pages=fetch_last_pages,
                            specific_pages=[],
                            start_page=1
                        ))
                    else:
                        supplemental_tasks.append(get_reddit_thread_content(
                            post_id=thread_id,
                            subreddit=source_id,
                            max_comments=max_comments
                        ))
                
                if supplemental_tasks:
                    supplemental_results = await asyncio.gather(*supplemental_tasks, return_exceptions=True)
                    for idx, result in enumerate(supplemental_results):
                        if isinstance(result, Exception):
                            logger.warning(f"無法抓取補充帖子 {filtered_supplemental[idx]['thread_id']}：{str(result)}")
                            continue
                        request_counter = result.get("request_counter", request_counter)
                        last_reset = result.get("last_reset", last_reset)
                        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                        rate_limit_info.extend(result.get("rate_limit_info", []))
                        
                        thread_id = str(filtered_supplemental[idx]["thread_id"])
                        if result.get("title"):
                            total_replies = result.get("total_replies", filtered_supplemental[idx].get("no_of_reply", 0))
                            if total_replies == 0:
                                total_replies = filtered_supplemental[idx].get("no_of_reply", 0)
                            filtered_replies = [
                                {
                                    "reply_id": reply.get("reply_id"),
                                    "msg": clean_html(reply.get("msg", "[無內容]")),
                                    "like_count": reply.get("like_count", 0),
                                    "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                    "reply_time": reply.get("reply_time", 0)
                                }
                                for reply in result.get("replies", [])
                                if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                                and len(clean_html(reply.get("msg")).strip()) > 7
                            ]
                            thread_info = {
                                "thread_id": thread_id,
                                "title": result.get("title"),
                                "no_of_reply": total_replies,
                                "last_reply_time": result.get("last_reply_time", 0),
                                "like_count": filtered_supplemental[idx].get("like_count", 0),
                                "dislike_count": filtered_supplemental[idx].get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "replies": filtered_replies,
                                "fetched_pages": result.get("fetched_pages", []),
                                "total_fetched_replies": len(filtered_replies)
                            }
                            thread_data.append(thread_info)
                            async with cache_lock:
                                st.session_state.thread_cache[thread_id] = {
                                    "data": thread_info,
                                    "timestamp": time.time()
                                }
            
            logger.info(
                f"最終 thread_data：{[{'thread_id': data['thread_id'], 'replies_count': len(data['replies'])} for data in thread_data]}"
            )
            
            return {
                "selected_source": selected_source,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        thread_data = []
        rate_limit_info = []
        candidate_threads = []
        processed_thread_ids = set()
        
        if top_thread_ids:
            candidate_threads = [
                {"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0}
                for tid in top_thread_ids
            ]
        else:
            initial_threads = []
            for page in range(1, 4):
                async with request_semaphore:
                    if source_type == "lihkg":
                        result = await get_lihkg_topic_list(
                            cat_id=source_id,
                            start_page=page,
                            max_pages=1
                        )
                    else:
                        result = await get_reddit_topic_list(
                            subreddit=source_id,
                            start_page=page,
                            max_pages=1
                        )
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                items = result.get("items", [])
                initial_threads.extend(items)
                if not items:
                    logger.warning(f"未抓取到分類 ID={source_id}，頁面={page} 的帖子")
                if len(initial_threads) >= 150:
                    initial_threads = initial_threads[:150]
                    break
                if progress_callback:
                    progress_callback(f"已抓取第 {page}/3 頁帖子", 0.1 + 0.2 * (page / 3))
            
            filtered_items = [
                item for item in initial_threads
                if item.get("no_of_reply", 0) >= min_replies
            ]
            
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                async with cache_lock:
                    if thread_id not in st.session_state.thread_cache:
                        cache_data = {
                            "thread_id": thread_id,
                            "title": item["title"],
                            "no_of_reply": item.get("no_of_reply", 0),
                            "last_reply_time": item.get("last_reply_time", 0),
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": [],
                            "fetched_pages": []
                        }
                        st.session_state.thread_cache[thread_id] = {
                            "data": cache_data,
                            "timestamp": time.time()
                        }
            
            if any(i["intent"] == "time_sensitive_analysis" for i in intents):
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: x.get("last_reply_time", 0),
                    reverse=True
                )
                candidate_threads = sorted_items[:post_limit]
            else:
                if filtered_items:
                    prioritization = await prioritize_threads_with_grok(
                        user_query, filtered_items, selected_source["source_name"], source_id, source_type, [i["intent"] for i in intents]
                    )
                    top_thread_ids = prioritization.get("top_thread_ids", [])
                    if not top_thread_ids:
                        sorted_items = sorted(
                            filtered_items,
                            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                            reverse=True
                        )
                        candidate_threads = sorted_items[:post_limit]
                    else:
                        candidate_threads = [
                            item for item in filtered_items
                            if str(item["thread_id"]) in map(str, top_thread_ids)
                        ][:post_limit]
        
        if progress_callback:
            progress_callback("正在抓取帖子內容", 0.3)
        
        tasks = []
        for idx, item in enumerate(candidate_threads):
            thread_id = str(item["thread_id"])
            if thread_id in processed_thread_ids:
                logger.debug(f"跳過重複的 thread_id={thread_id}")
                continue
            processed_thread_ids.add(thread_id)
            async with cache_lock:
                if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
                    cached_data = st.session_state.thread_cache[thread_id]["data"]
                    thread_data.append(cached_data)
                    logger.debug(f"緩存命中：thread_id={thread_id}")
                    continue
            if source_type == "lihkg":
                tasks.append((idx, get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=source_id,
                    max_replies=max_replies,
                    fetch_last_pages=fetch_last_pages,
                    specific_pages=[],
                    start_page=1
                )))
            else:
                tasks.append((idx, get_reddit_thread_content(
                    post_id=thread_id,
                    subreddit=source_id,
                    max_comments=max_comments
                )))
        
        if tasks:
            content_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            for task_idx, result in enumerate(content_results):
                idx = tasks[task_idx][0]
                if isinstance(result, Exception):
                    logger.warning(f"無法抓取帖子 {candidate_threads[idx]['thread_id']} 內容：{str(result)}")
                    continue
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                
                thread_id = str(candidate_threads[idx]["thread_id"])
                if result.get("title"):
                    total_replies = result.get("total_replies", candidate_threads[idx]["no_of_reply"])
                    if total_replies == 0:
                        total_replies = candidate_threads[idx]["no_of_reply"]
                    filtered_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "reply_time": reply.get("reply_time", 0)
                        }
                        for reply in result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                        and len(clean_html(reply.get("msg")).strip()) > 7
                    ]
                    thread_info = {
                        "thread_id": thread_id,
                        "title": result.get("title"),
                        "no_of_reply": total_replies,
                        "last_reply_time": result.get("last_reply_time", 0),
                        "like_count": candidate_threads[idx].get("like_count", 0),
                        "dislike_count": candidate_threads[idx].get("dislike_count", 0) if source_type == "lihkg" else 0,
                        "replies": filtered_replies,
                        "fetched_pages": result.get("fetched_pages", []),
                        "total_fetched_replies": len(filtered_replies)
                    }
                    thread_data.append(thread_info)
                    async with cache_lock:
                        st.session_state.thread_cache[thread_id] = {
                            "data": thread_info,
                            "timestamp": time.time()
                        }
        
        if progress_callback:
            progress_callback("正在準備數據", 0.5)
        
        logger.info(
            f"最終 thread_data：{[{'thread_id': data['thread_id'], 'replies_count': len(data['replies'])} for data in thread_data]}"
        )
        
        return {
            "selected_source": selected_source,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(f"處理用戶問題失敗：{str(e)}")
        return {
            "selected_source": selected_source,
            "thread_data": [],
            "rate_limit_info": [{"message": f"處理錯誤：{str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
