"""
Grok 3 API 處理模組，負責動態推斷意圖並生成回應。
使用單一通用提示詞，依賴 Grok 3 處理所有意圖，無需預定義。
主要函數：
- process_user_question：抓取數據，調用 Grok 3，生成標準 JSON 回應。
- truncate_data：截斷數據，控制提示詞長度。
- cache_response / get_cached_response：快取回應，減少 API 調用。
"""

import json
import aiohttp
import logging
import time
import streamlit as st
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content, get_category_name

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
logger.handlers.clear()
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90  # 秒

# 通用提示詞
GENERAL_PROMPT = """
你是LIHKG論壇的集體意見代表，以繁體中文回答，模擬論壇用戶的語氣。根據用戶問題和提供的數據，執行以下任務：
1. 理解問題的語義，推斷用戶意圖（例如列出帖子標題、總結話題、提取關鍵詞、分析情緒或其他）。
2. 根據推斷的意圖，處理以下數據：
   - 帖子元數據：{metadata}
   - 回覆內容：{thread_data}
3. 返回JSON格式：
```json
{
  "intent": {string} (推斷的意圖名稱，例如 "list_titles", "summarize_posts", "analyze_sentiment"),
  "response": {string} (自然語言回應),
  "raw_result": {object} (意圖特定的結構化數據，可選)
}
```
要求：
- 意圖名稱應簡潔且一致，例如用 'list_titles' 表示列出標題，'analyze_sentiment' 表示情緒分析。
- 回應應清晰、符合問題需求，字數根據任務適配（簡單任務200字，複雜任務400-600字）。
- 對於分析任務（如情緒分析），提供具體證據，例如引用帖子標題或回覆內容。
- 若數據不足，返回："在{cat_name}中未找到符合條件的帖子。"
- 若無法推斷意圖，返回通用回應並設置 intent 為 "general_query"。
- 確保JSON格式嚴格有效。

問題：{query}
分類：{cat_name}（cat_id={cat_id})
對話歷史：{conversation_context}
"""

def truncate_data(metadata, thread_data, max_replies_per_thread=20):
    """截斷數據以控制提示詞長度"""
    truncated_metadata = metadata[:10]  # 限制帖子數
    truncated_thread_data = {
        tid: {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": sorted(
                [r for r in data.get("replies", []) if r.get("msg")],
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:max_replies_per_thread],
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items()
    }
    return truncated_metadata, truncated_thread_data

def cache_response(user_query, result):
    """快取回應，1小時有效"""
    st.session_state.response_cache[user_query] = {
        "result": result,
        "timestamp": time.time()
    }
    logger.info(f"Cached response for query: {user_query}")

def get_cached_response(user_query):
    """獲取快取回應"""
    cached = st.session_state.response_cache.get(user_query)
    if cached and time.time() - cached["timestamp"] < 3600:
        logger.info(f"Using cached response for query: {user_query}")
        return cached["result"]
    return None

async def retry_with_simplified_prompt(user_query, selected_cat, cat_id, conversation_context):
    """重試簡化提示詞"""
    prompt = f"""
    你是LIHKG論壇助手，以繁體中文回答。簡化回答以下問題，200字以內：
    問題：{user_query}
    分類：{selected_cat}（cat_id={cat_id})
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    格式為JSON：
    ```json
    {{"intent": "general_query", "response": "...", "raw_result": {{}}}}
    ```
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {
            "intent": "general_query",
            "response": "錯誤：缺少 API 密鑰",
            "raw_result": {}
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": "你是由 xAI 創建的 Grok 3，以繁體中文回答。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                logger.info(f"Retry succeeded for query: {user_query}")
                return result
    except Exception as e:
        logger.error(f"Retry failed: {str(e)}")
        return {
            "intent": "general_query",
            "response": "無法處理查詢，請稍後重試。",
            "raw_result": {}
        }

async def process_user_question(user_question, selected_cat, cat_id, post_limit, reply_limit, request_counter, last_reset, rate_limit_until, conversation_context, progress_callback=None):
    """
    處理用戶問題，抓取 LIHKG 數據並調用 Grok 3 生成回應。
    返回標準 JSON 格式，包含推斷的意圖和回應。
    """
    try:
        logger.info(
            json.dumps({
                "event": "process_user_question",
                "query": user_question,
                "category": selected_cat,
                "cat_id": cat_id
            }, ensure_ascii=False)
        )

        # 檢查快取
        if cached_result := get_cached_response(user_question):
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                **cached_result
            }

        # 檢查速率限制
        if rate_limit_until > time.time():
            logger.warning(f"Rate limit active until {rate_limit_until}")
            result = {
                "intent": "general_query",
                "response": f"速率限制中，請在 {datetime.fromtimestamp(rate_limit_until):%Y-%m-%d %H:%M:%S} 後重試。",
                "raw_result": {}
            }
            cache_response(user_question, result)
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                **result
            }

        # 抓取帖子數據
        if progress_callback:
            progress_callback("正在抓取帖子列表", 0.2)
        
        topic_result = await get_lihkg_topic_list(
            cat_id=cat_id,
            start_page=1,
            max_pages=3,
            request_counter=request_counter,
            last_reset=last_reset,
            rate_limit_until=rate_limit_until
        )
        request_counter = topic_result.get("request_counter", request_counter)
        last_reset = topic_result.get("last_reset", last_reset)
        rate_limit_until = topic_result.get("rate_limit_until", request_counter)
        rate_limit_info = topic_result.get("rate_limit_info", [])
        initial_threads = topic_result.get("items", [])

        if progress_callback:
            progress_callback("正在篩選帖子", 0.4)

        # 篩選帖子
        filtered_items = [
            {
                "thread_id": str(item["thread_id"]),
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0)
            } for item in initial_threads
            if item.get("no_of_reply", 0) >= 50 and item.get("like_count", 0) >= 10
        ]
        sorted_items = sorted(
            filtered_items,
            key=lambda x: x["no_of_reply"] * 0.6 + x["like_count"] * 0.4,
            reverse=True
        )
        top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]

        # 更新緩存
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            if thread_id not in st.session_state.thread_cache:
                st.session_state.thread_cache[thread_id] = {
                    "data": {
                        "thread_id": thread_id,
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item.get("last_reply_time", 0),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [],
                        "fetched_pages": []
                    },
                    "timestamp": time.time()
                }

        # 抓取帖子內容
        thread_data = {}
        if progress_callback:
            progress_callback("正在抓取帖子內容", 0.6)
        
        for idx, thread_id in enumerate(top_thread_ids):
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
            sorted_replies = sorted(
                [r for r in replies if r.get("msg")],
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:reply_limit]
            
            thread_data[thread_id] = {
                "thread_id": thread_id,
                "title": thread_result.get("title", ""),
                "no_of_reply": thread_result.get("total_replies", 0),
                "last_reply_time": sorted_replies[0]["reply_time"] if sorted_replies else "0",
                "like_count": sum(r.get("like_count", 0) for r in sorted_replies),
                "dislike_count": sum(r.get("dislike_count", 0) for r in sorted_replies),
                "replies": [
                    {
                        "msg": r["msg"],
                        "like_count": r.get("like_count", 0),
                        "dislike_count": r.get("dislike_count", 0),
                        "reply_time": r.get("reply_time", 0)
                    } for r in sorted_replies
                ],
                "fetched_pages": thread_result.get("fetched_pages", [])
            }
            
            st.session_state.thread_cache[thread_id]["data"].update({
                "replies": thread_data[thread_id]["replies"],
                "fetched_pages": thread_data[thread_id]["fetched_pages"]
            })
            st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
            
            if progress_callback:
                progress_callback(f"已抓取帖子 {idx + 1}/{len(top_thread_ids)}", 0.6 + 0.2 * ((idx + 1) / len(top_thread_ids)))

        # 截斷數據
        metadata, thread_data = truncate_data(filtered_items, thread_data)
        
        # 構造提示詞
        prompt = GENERAL_PROMPT.format(
            query=user_question,
            cat_name=selected_cat,
            cat_id=cat_id,
            conversation_context=json.dumps(conversation_context, ensure_ascii=False),
            metadata=json.dumps(metadata, ensure_ascii=False),
            thread_data=json.dumps(thread_data, ensure_ascii=False)
        )
        
        # 檢查提示詞長度
        if len(prompt) > GROK3_TOKEN_LIMIT:
            logger.warning("Prompt too long, reducing data")
            metadata, thread_data = truncate_data(metadata, thread_data, max_replies_per_thread=10)
            prompt = GENERAL_PROMPT.format(
                query=user_question,
                cat_name=selected_cat,
                cat_id=cat_id,
                conversation_context=json.dumps(conversation_context, ensure_ascii=False),
                metadata=json.dumps(metadata, ensure_ascii=False),
                thread_data=json.dumps(thread_data, ensure_ascii=False)
            )
        
        if progress_callback:
            progress_callback("正在生成回應", 0.8)

        # 調用 Grok 3 API
        try:
            GROK3_API_KEY = st.secrets["grok3key"]
        except KeyError as e:
            logger.error(f"Grok 3 API key missing: {str(e)}")
            result = {
                "intent": "general_query",
                "response": "錯誤：缺少 API 密鑰",
                "raw_result": {}
            }
            cache_response(user_question, result)
            return {
                "selected_cat": selected_cat,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                **result
            }
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
        payload = {
            "model": "grok-3-beta",
            "messages": [
                {"role": "system", "content": "你是由 xAI 創建的 Grok 3，代表 LIHKG 論壇的集體意見，以繁體中文回答。"},
                *conversation_context,
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 600,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code != 200:
                        logger.error(f"API request failed with status {status_code}")
                        result = await retry_with_simplified_prompt(user_question, selected_cat, cat_id, conversation_context)
                        cache_response(user_question, result)
                        return {
                            "selected_cat": selected_cat,
                            "thread_data": thread_data,
                            "rate_limit_info": rate_limit_info,
                            "request_counter": request_counter,
                            "last_reset": last_reset,
                            "rate_limit_until": rate_limit_until,
                            **result
                        }
                    
                    data = await response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    
                    # 驗證結果格式
                    if not isinstance(result, dict) or "intent" not in result or "response" not in result:
                        logger.error("Invalid response format")
                        result = await retry_with_simplified_prompt(user_question, selected_cat, cat_id, conversation_context)
                    
                    # 標準化意圖名稱
                    intent_mapping = {
                        "sentiment_analysis": "analyze_sentiment",
                        "emotion_analysis": "analyze_sentiment",
                        "show_titles": "list_titles",
                        "summary": "summarize_posts"
                    }
                    result["intent"] = intent_mapping.get(result["intent"], result["intent"])
                    
                    # 檢查回應質量
                    if len(result["response"]) < 50:
                        logger.warning(f"Response too short for query: {user_question}")
                        result = await retry_with_simplified_prompt(user_question, selected_cat, cat_id, conversation_context)
                    
                    logger.info(
                        json.dumps({
                            "event": "grok3_api_call",
                            "action": "完成回應生成",
                            "query": user_question,
                            "intent": result["intent"],
                            "response_length": len(result["response"])
                        }, ensure_ascii=False)
                    )
                    cache_response(user_question, result)
                    return {
                        "selected_cat": selected_cat,
                        "thread_data": thread_data,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until,
                        **result
                    }
            
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Grok 3 response error: {str(e)}")
                result = await retry_with_simplified_prompt(user_question, selected_cat, cat_id, conversation_context)
                cache_response(user_question, result)
                return {
                    "selected_cat": selected_cat,
                    "thread_data": thread_data,
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": rate_limit_until,
                    **result
                }
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                # 備用回應
                if filtered_items:
                    result = {
                        "intent": "list_titles",
                        "response": "近期帖子：\n" + "\n".join(
                            f"- 帖子ID: {item['thread_id']} 標題: {item['title']}"
                            for item in filtered_items[:5]
                        ),
                        "raw_result": {}
                    }
                    cache_response(user_question, result)
                    return {
                        "selected_cat": selected_cat,
                        "thread_data": thread_data,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until,
                        **result
                    }
                result = {
                    "intent": "general_query",
                    "response": "無法連接API，請稍後重試。",
                    "raw_result": {}
                }
                cache_response(user_question, result)
                return {
                    "selected_cat": selected_cat,
                    "thread_data": [],
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": rate_limit_until,
                    **result
                }
    
    except Exception as e:
        logger.error(
            json.dumps({
                "event": "processing_error",
                "query": user_question,
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e)
            }, ensure_ascii=False)
        )
        result = {
            "intent": "general_query",
            "response": f"處理失敗：{str(e)}",
            "raw_result": {}
        }
        cache_response(user_question, result)
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": [{"message": f"Processing failed: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            **result
        }