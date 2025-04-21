"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
包含數據處理邏輯（進階分析、緩存管理）和輔助函數。
主要函數：
- analyze_and_screen：分析問題，識別細粒度意圖，動態設置篩選條件。
- stream_grok3_response：生成流式回應，根據意圖和分類動態選擇模板。
- process_user_question：處理用戶問題，抓取並分析帖子。
- clean_html：清理 HTML 標籤。
"""

import aiohttp
import asyncio
import json
import re
import logging
import streamlit as st
import time
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

# 檔案處理器：寫入 app.log
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台處理器：輸出到 stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90  # 秒

def clean_html(text):
    """
    清理 HTML 標籤，規範化文本。
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}", extra={"function": "clean_html"})
        return text

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，識別細粒度意圖，動態設置篩選條件。
    使用 Grok 3 提取主題和關鍵詞，減少硬編碼。
    """
    conversation_context = conversation_context or []
    prompt = f"""
    你是 LIHKG 論壇的集體意見代表，根據用戶問題和提供的數據，以繁體中文回覆，模擬論壇用戶的語氣。輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else '無標題數據'}
    {'元數據：' + json.dumps(metadata, ensure_ascii=False) if metadata else '無元數據'}
    {'回覆數據：' + json.dumps(thread_data, ensure_ascii=False) if thread_data else '無回覆數據'}

    步驟：
    1. 分析問題意圖，動態分類：
       - list_titles：僅列出帖子標題。
       - summarize_posts：總結帖子內容，引用高關注回覆。
       - analyze_sentiment：分析帖子情緒（正面、負面、中立）。
       - compare_categories：比較多個討論區的話題或趨勢。
       - general_query：一般問題，無需帖子數據。
       - find_themed：尋找特定主題的帖子，根據問題中的情感、形容詞或主題詞（例如「搞笑」「感動」「白痴」）設置。
    2. 若問題包含「分析」，優先選擇 summarize_posts 或 analyze_sentiment：
       - 若要求內容總結，選 summarize_posts。
       - 若要求觀點或情緒，選 analyze_sentiment。
    3. 若問題指定主題（例如「感動」「on9」「白痴」），設置 intent 為 find_themed，並提取主題詞存入 theme：
       - 提取問題中的核心形容詞或情感詞作為 theme（例如「感動」「on9」）。
       - 使用自然語言理解生成與主題相關的關鍵詞列表（theme_keywords），包括同義詞、俚語和相關詞（例如「感動」可包括「感人」「催淚」）。
       - 若無法明確主題，設置 theme 為「相關」，theme_keywords 為問題中的主要名詞。
    4. 若問題涉及「熱門」或排序（如「最」），從帖子標題或元數據中選最多10個帖子ID（top_thread_ids）：
       - 排序標準：回覆數(no_of_reply)*0.6 + 點讚數(like_count)*0.4。
       - 若指定主題，優先匹配標題或回覆中包含 theme_keywords 的帖子。
    5. 動態確定：
       - post_limit（1-20）：根據問題複雜度。
       - reply_limit（0-500）：若需分析內容或情緒，設置足夠回覆數。
       - filters：根據討論區和主題設置（例如 min_replies=50-200, min_likes=10-100；若主題為搞笑或輕鬆，可降低 min_likes）。
    6. 若為進階分析（is_advanced=True），檢查數據是否足夠，設置 needs_advanced_analysis。
    7. 提供候選帖子ID（candidate_thread_ids）和處理方式（processing，例如 summarize, sentiment, themed）。
    8. 若無法生成 top_thread_ids，提供原因（reason）。

    輸出格式：
    {{
        "direct_response": boolean,
        "intent": string,
        "theme": string,
        "category_ids": array,
        "data_type": string (title_only, replies, both),
        "post_limit": integer,
        "reply_limit": integer,
        "filters": object,
        "processing": string,
        "candidate_thread_ids": array,
        "top_thread_ids": array,
        "needs_advanced_analysis": boolean,
        "reason": string,
        "theme_keywords": array
    }}
    示例（主題為感動）：
    {{
        "direct_response": false,
        "intent": "find_themed",
        "theme": "感動",
        "category_ids": ["{cat_id}"],
        "data_type": "both",
        "post_limit": 5,
        "reply_limit": 100,
        "filters": {{"min_replies": 50, "min_likes": 10}},
        "processing": "themed",
        "candidate_thread_ids": [],
        "top_thread_ids": ["3915677", "3910235"],
        "needs_advanced_analysis": false,
        "reason": "",
        "theme_keywords": ["感動", "感人", "催淚", "溫馨"]
    }}
    若無數據，設置合理參數並說明：
    {{
        "direct_response": false,
        "intent": "find_themed",
        "theme": "感動",
        "category_ids": ["{cat_id}"],
        "data_type": "both",
        "post_limit": 5,
        "reply_limit": 100,
        "filters": {{"min_replies": 50, "min_likes": 10}},
        "processing": "themed",
        "candidate_thread_ids": [],
        "top_thread_ids": [],
        "needs_advanced_analysis": true,
        "reason": "無帖子數據，建議抓取更多帖子",
        "theme_keywords": ["感動", "感人", "催淚", "溫馨"]
    }}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}", extra={"function": "analyze_and_screen"})
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": "",
            "category_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "reply_limit": 0,
            "filters": {},
            "processing": "general",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "Missing API key",
            "theme_keywords": []
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": "你是由 xAI 創建的 Grok 3，代表 LIHKG 論壇的集體意見，以繁體中文回答。根據問題語義和提供數據直接回應，無需提及身份或語氣。"},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.7
    }
    
    logger.info(
        json.dumps({
            "event": "grok3_api_call",
            "function": "analyze_and_screen",
            "action": "发起問題意圖分析",
            "query": user_query,
            "prompt_length": len(prompt)
        }, ensure_ascii=False),
        extra={"function": "analyze_and_screen"}
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    response_text = await response.text()
                    if status_code != 200:
                        logger.warning(
                            json.dumps({
                                "event": "grok3_api_call",
                                "function": "analyze_and_screen",
                                "action": "問題意圖分析失敗",
                                "query": user_query,
                                "status": "failed",
                                "status_code": status_code,
                                "response_text": response_text[:500],
                                "attempt": attempt + 1
                            }, ensure_ascii=False),
                            extra={"function": "analyze_and_screen"}
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {
                            "direct_response": True,
                            "intent": "general_query",
                            "theme": "",
                            "category_ids": [],
                            "data_type": "none",
                            "post_limit": 5,
                            "reply_limit": 0,
                            "filters": {},
                            "processing": "general",
                            "candidate_thread_ids": [],
                            "top_thread_ids": [],
                            "needs_advanced_analysis": False,
                            "reason": f"API request failed with status {status_code}",
                            "theme_keywords": []
                        }
                    
                    data = await response.json()
                    if "choices" not in data or not data["choices"]:
                        logger.warning(
                            json.dumps({
                                "event": "grok3_api_call",
                                "function": "analyze_and_screen",
                                "action": "問題意圖分析失敗",
                                "query": user_query,
                                "status": "failed",
                                "status_code": status_code,
                                "error": "Missing 'choices' in response",
                                "attempt": attempt + 1
                            }, ensure_ascii=False),
                            extra={"function": "analyze_and_screen"}
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {
                            "direct_response": True,
                            "intent": "general_query",
                            "theme": "",
                            "category_ids": [],
                            "data_type": "none",
                            "post_limit": 5,
                            "reply_limit": 0,
                            "filters": {},
                            "processing": "general",
                            "candidate_thread_ids": [],
                            "top_thread_ids": [],
                            "needs_advanced_analysis": False,
                            "reason": "Invalid API response: missing 'choices'",
                            "theme_keywords": []
                        }
                    
                    result = json.loads(data["choices"][0]["message"]["content"])
                    result.setdefault("direct_response", False)
                    result.setdefault("intent", "general_query")
                    result.setdefault("theme", "")
                    result.setdefault("category_ids", [cat_id] if not result.get("direct_response") else [])
                    result.setdefault("data_type", "both")
                    result.setdefault("post_limit", 5)
                    result.setdefault("reply_limit", 100)
                    result.setdefault("filters", {"min_replies": 50, "min_likes": 10})
                    result.setdefault("processing", "summarize")
                    result.setdefault("candidate_thread_ids", [])
                    result.setdefault("top_thread_ids", [])
                    result.setdefault("needs_advanced_analysis", False)
                    result.setdefault("reason", "")
                    result.setdefault("theme_keywords", [])
                    logger.info(
                        json.dumps({
                            "event": "grok3_api_call",
                            "function": "analyze_and_screen",
                            "action": "完成問題意圖分析",
                            "query": user_query,
                            "status": "success",
                            "status_code": status_code,
                            "intent": result["intent"],
                            "theme": result["theme"],
                            "needs_advanced_analysis": result["needs_advanced_analysis"],
                            "filters": result["filters"],
                            "top_thread_ids": result["top_thread_ids"],
                            "theme_keywords": result["theme_keywords"]
                        }, ensure_ascii=False),
                        extra={"function": "analyze_and_screen"}
                    )
                    return result
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            status_code = getattr(e, 'status', None) or "unknown"
            logger.warning(
                json.dumps({
                    "event": "grok3_api_call",
                    "function": "analyze_and_screen",
                    "action": "問題意圖分析失敗",
                    "query": user_query,
                    "status": "failed",
                    "status_code": status_code,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "attempt": attempt + 1
                }, ensure_ascii=False),
                extra={"function": "analyze_and_screen"}
            )
            if attempt < max_retries - 1:
                simplified_prompt = f"""
                你是 LIHKG 論壇助手，分析用戶問題，輸出 JSON。
                問題：{user_query}
                分類：{cat_name}（cat_id={cat_id})
                步驟：
                1. 判斷意圖（例如：列出帖子、總結、情緒分析、一般問題、尋找特定主題）。
                2. 若問題指定主題，設置 intent="find_themed"，提取主題詞存入 theme，生成相關關鍵詞存入 theme_keywords。
                3. 設置帖子數量（post_limit，1-20）、回覆數量（reply_limit，0-500）。
                4. 設置篩選條件（filters，例如 min_replies=50-200, min_likes=10-100）。
                5. 若有帖子標題，選最多10個帖子ID（top_thread_ids），按回覆數*0.6+點讚數*0.4排序，優先匹配主題相關詞。
                輸出：
                {{"direct_response": boolean, "intent": string, "theme": string, "category_ids": array, "data_type": string, "post_limit": integer, "reply_limit": integer, "filters": object, "processing": string, "candidate_thread_ids": array, "top_thread_ids": array, "needs_advanced_analysis": boolean, "reason": string, "theme_keywords": array}}
                """
                messages[-1]["content"] = simplified_prompt
                payload["max_tokens"] = 300
                logger.info(
                    json.dumps({
                        "event": "grok3_api_call",
                        "function": "analyze_and_screen",
                        "action": "重試問題意圖分析（簡化提示詞）",
                        "query": user_query,
                        "prompt_length": len(simplified_prompt)
                    }, ensure_ascii=False),
                    extra={"function": "analyze_and_screen"}
                )
                await asyncio.sleep(2)
                continue
            return {
                "direct_response": True,
                "intent": "general_query",
                "theme": "",
                "category_ids": [],
                "data_type": "none",
                "post_limit": 5,
                "reply_limit": 0,
                "filters": {},
                "processing": "general",
                "candidate_thread_ids": [],
                "top_thread_ids": [],
                "needs_advanced_analysis": False,
                "reason": f"Analysis failed after {max_retries} attempts: {str(e)}",
                "theme_keywords": []
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None):
    """
    使用 Grok 3 API 生成流式回應，根據意圖和分類動態選擇模板。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 50, "min_likes": 10}
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}", extra={"function": "stream_grok3_response"})
        yield "錯誤: 缺少 API 密鑰"
        return
    
    # 驗證輸入數據
    logger.info(
        json.dumps({
            "event": "data_validation",
            "function": "stream_grok3_response",
            "query": user_query,
            "metadata_count": len(metadata) if metadata else 0,
            "thread_data_count": len(thread_data) if thread_data else 0
        }, ensure_ascii=False),
        extra={"function": "stream_grok3_response"}
    )
    
    # 動態調整回覆數量，確保提示詞不過長
    max_replies_per_thread = 20  # 每個帖子最多取20條回覆
    filtered_thread_data = {
        tid: {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": sorted(data.get("replies", []), key=lambda x: x.get("like_count", 0), reverse=True)[:max_replies_per_thread],
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items()
    }
    
    # 記錄過濾後的回覆數量和點讚數
    for tid in filtered_thread_data:
        replies = filtered_thread_data[tid]["replies"]
        like_counts = [r.get("like_count", 0) for r in replies]
        logger.info(
            json.dumps({
                "event": "filtered_thread_data",
                "function": "stream_grok3_response",
                "thread_id": tid,
                "reply_count": len(replies),
                "like_counts_summary": {
                    "min": min(like_counts) if like_counts else 0,
                    "max": max(like_counts) if like_counts else 0,
                    "avg": sum(like_counts) / len(like_counts) if like_counts else 0
                }
            }, ensure_ascii=False),
            extra={"function": "stream_grok3_response"}
        )
    
    # 若filtered_thread_data無回覆，回退到原始數據的前10條回覆
    if not any(data["replies"] for data in filtered_thread_data.values()):
        logger.warning(
            json.dumps({
                "event": "data_validation",
                "function": "stream_grok3_response",
                "query": user_query,
                "reason": "Filtered thread data has no replies, using raw thread data"
            }, ensure_ascii=False),
            extra={"function": "stream_grok3_response"}
        )
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": data.get("replies", [])[:10],
                "fetched_pages": data.get("fetched_pages", [])
            } for tid, data in thread_data.items()
        }
    
    prompt_templates = {
        "list": f"""
        你是 LIHKG 論壇的數據助手，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子元數據：{json.dumps(metadata, ensure_ascii=False)}
        篩選條件：{json.dumps(filters, ensure_ascii=False)}
        任務：
        1. 列出帖子標題，格式為：
           - 帖子 ID: [thread_id] 標題: [title]
        2. 若無符合條件的帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子（篩選：回覆數≥{filters.get('min_replies', 0)}，點讚數≥{filters.get('min_likes', 0)}）。」
        3. 若無任何帖子，回答：「在 {selected_cat} 中目前無可用帖子。」
        輸出：標題列表或無帖子提示
        """,
        "summarize": f"""
        你是 LIHKG 論壇的集體意見代表，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        任務：
        1. 分析每個帖子的主題和核心觀點，引用高關注回覆（若無高關注回覆，引用前幾條回覆）。
        2. 總結熱門話題的趨勢（例如社會議題、娛樂八卦）。
        3. 提取用戶關注點和討論熱度原因。
        4. 若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子（篩選：回覆數≥{filters.get('min_replies', 0)}，點讚數≥{filters.get('min_likes', 0)}）。」
        5. 若回覆數據不足，基於帖子標題生成簡化總結。
        6. 字數：400-600字。
        輸出：詳細總結或無帖子提示
        """,
        "sentiment": f"""
        你是 LIHKG 論壇的集體意見代表，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        任務：
        1. 分析每個帖子的情緒分佈（正面、負面、中立），聚焦高關注回覆（若無高關注回覆，分析前幾條回覆）。
        2. 量化情緒比例（例如 正面40%，負面30%，中立30%）。
        3. 說明情緒背後的原因。
        4. 若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子（篩選：回覆數≥{filters.get('min_replies', 0)}，點讚數≥{filters.get('min_likes', 0)}）。」
        5. 字數：300-500字。
        輸出：情緒分析或無帖子提示
        """,
        "introduce": f"""
        你是 Grok 3，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        回答：「我是 Grok 3，由 xAI 創建的智能助手，專為解答問題和分析 LIHKG 論壇數據設計。有什麼可以幫您的？」（50-100 字）。
        輸出：自我介紹
        """,
        "general": f"""
        你是 Grok 3，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        根據問題語義提供直接回應，聚焦問題核心（50-100 字）。
        若問題涉及 LIHKG 論壇數據但無具體要求，回答：「請提供更明確的查詢以分析 {selected_cat} 的帖子，例如具體主題或分析類型。」
        若無帖子數據，回答：「在 {selected_cat} 中未找到符合條件的帖子（篩選：回覆數≥{filters.get('min_replies', 0)}，點讚數≥{filters.get('min_likes', 0)}）。」
        輸出：直接回應或無帖子提示
        """
    }
    
    intent = processing.get('intent', 'general') if isinstance(processing, dict) else 'general'
    if user_query.lower() in ["你是誰？", "你是誰", "who are you?", "who are you"] or "你是誰" in user_query.lower():
        intent = "introduce"
    
    # 確保 summarize_posts 意圖優先使用 summarize 模板
    if intent == "summarize_posts" and metadata and thread_data:
        prompt = prompt_templates["summarize"]
    else:
        prompt = prompt_templates.get(intent, prompt_templates["general"])
    
    # 檢查提示詞長度和內容
    prompt_length = len(prompt)
    if prompt_length < 500:
        logger.warning(
            json.dumps({
                "event": "prompt_validation",
                "function": "stream_grok3_response",
                "query": user_query,
                "prompt_length": prompt_length,
                "reason": "Prompt too short, likely missing data",
                "metadata_summary": {
                    "count": len(metadata) if metadata else 0,
                    "sample": metadata[:1] if metadata else []
                },
                "thread_data_summary": {
                    "count": len(thread_data) if thread_data else 0,
                    "sample": {tid: {"title": data["title"], "reply_count": len(data.get("replies", []))} for tid, data in list(thread_data.items())[:1]} if thread_data else {}
                }
            }, ensure_ascii=False),
            extra={"function": "stream_grok3_response"}
        )
        # 回退到簡化提示詞
        prompt = f"""
        你是 LIHKG 論壇的集體意見代表，以繁體中文回答。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        任務：
        1. 基於帖子標題生成簡化總結，概述熱門話題。
        2. 若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子（篩選：回覆數≥{filters.get('min_replies', 0)}，點讚數≥{filters.get('min_likes', 0)}）。」
        3. 字數：200-300字。
        輸出：簡化總結或無帖子提示
        """
        logger.info(
            json.dumps({
                "event": "prompt_fallback",
                "function": "stream_grok3_response",
                "query": user_query,
                "reason": "Using simplified prompt due to insufficient data",
                "new_prompt_length": len(prompt)
            }, ensure_ascii=False),
            extra={"function": "stream_grok3_response"}
        )
    
    # 記錄最終提示詞內容（僅限調試）
    logger.debug(
        json.dumps({
            "event": "prompt_content",
            "function": "stream_grok3_response",
            "query": user_query,
            "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt
        }, ensure_ascii=False),
        extra={"function": "stream_grok3_response"}
    )
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": "你是由 xAI 創建的 Grok 3，代表 LIHKG 論壇的集體意見，以繁體中文回答。根據問題語義和提供數據直接回應，無需提及身份或語氣。"},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": True
    }
    
    logger.info(
        json.dumps({
            "event": "grok3_api_call",
            "function": "stream_grok3_response",
            "action": "发起回應生成",
            "query": user_query,
            "prompt_length": len(prompt)
        }, ensure_ascii=False),
        extra={"function": "stream_grok3_response"}
    )
    
    response_content = ""
    async with aiohttp.ClientSession() as session:
        try:
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        status_code = response.status
                        if status_code != 200:
                            response_text = await response.text()
                            logger.warning(
                                json.dumps({
                                    "event": "grok3_api_call",
                                    "function": "stream_grok3_response",
                                    "action": "回應生成失敗",
                                    "query": user_query,
                                    "status": "failed",
                                    "status_code": status_code,
                                    "response_text": response_text[:500],
                                    "attempt": attempt + 1
                                }, ensure_ascii=False),
                                extra={"function": "stream_grok3_response"}
                            )
                            if attempt < 2:
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            yield f"錯誤：API 請求失敗（狀態碼 {status_code}）。請稍後重試。"
                            return
                        
                        async for line in response.content:
                            if line and not line.isspace():
                                line_str = line.decode('utf-8').strip()
                                if line_str == "data: [DONE]":
                                    break
                                if line_str.startswith("data: "):
                                    try:
                                        chunk = json.loads(line_str[6:])
                                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if content:
                                            if "###" in content:
                                                logger.warning(
                                                    f"Detected ### in response chunk: {content}",
                                                    extra={"function": "stream_grok3_response"}
                                                )
                                                if "Content Moderation" in content or "Blocked" in content:
                                                    raise ValueError("Content moderation detected")
                                            response_content += content
                                            yield content
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON decode error in stream chunk: {str(e)}", extra={"function": "stream_grok3_response"})
                                        continue
                        if not response_content:
                            logger.warning(f"No content generated for query: {user_query}")
                            response_content = "無法生成回應，請稍後重試。"
                            yield response_content
                        logger.info(
                            json.dumps({
                                "event": "grok3_api_call",
                                "function": "stream_grok3_response",
                                "action": "完成回應生成",
                                "query": user_query,
                                "status": "success",
                                "status_code": status_code,
                                "response_length": len(response_content)
                            }, ensure_ascii=False),
                            extra={"function": "stream_grok3_response"}
                        )
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    status_code = getattr(e, 'status', None) or "unknown"
                    logger.warning(
                        json.dumps({
                            "event": "grok3_api_call",
                            "function": "stream_grok3_response",
                            "action": "回應生成失敗",
                            "query": user_query,
                            "status": "failed",
                            "status_code": status_code,
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "attempt": attempt + 1
                        }, ensure_ascii=False),
                        extra={"function": "stream_grok3_response"}
                    )
                    if attempt < 2:
                        max_replies_per_thread = max_replies_per_thread // 2
                        filtered_thread_data = {
                            tid: {
                                "thread_id": data["thread_id"],
                                "title": data["title"],
                                "no_of_reply": data.get("no_of_reply", 0),
                                "last_reply_time": data.get("last_reply_time", 0),
                                "like_count": data.get("like_count", 0),
                                "dislike_count": data.get("dislike_count", 0),
                                "replies": sorted(data.get("replies", []), key=lambda x: x.get("like_count", 0), reverse=True)[:max_replies_per_thread],
                                "fetched_pages": data.get("fetched_pages", [])
                            } for tid, data in thread_data.items()
                        }
                        prompt = prompt_templates["summarize"].replace(
                            json.dumps(filtered_thread_data, ensure_ascii=False),
                            json.dumps(filtered_thread_data, ensure_ascii=False)
                        )
                        payload["messages"][-1]["content"] = prompt
                        logger.info(
                            json.dumps({
                                "event": "grok3_api_call",
                                "function": "stream_grok3_response",
                                "action": "重試回應生成（減少回覆數據）",
                                "query": user_query,
                                "prompt_length": len(prompt)
                            }, ensure_ascii=False),
                            extra={"function": "stream_grok3_response"}
                        )
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試。"
                    return
        except Exception as e:
            status_code = "unknown"
            logger.error(
                json.dumps({
                    "event": "grok3_api_call",
                    "function": "stream_grok3_response",
                    "action": "回應生成異常",
                    "query": user_query,
                    "status": "failed",
                    "status_code": status_code,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }, ensure_ascii=False),
                extra={"function": "stream_grok3_response"}
            )
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
        finally:
            await session.close()

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    """
    處理用戶問題，抓取並分析 LIHKG 帖子。
    使用 Grok 3 進行主題匹配，減少硬編碼。
    """
    try:
        logger.info(
            json.dumps({
                "event": "process_user_question",
                "function": "process_user_question",
                "query": user_question,
                "category": selected_cat,
                "cat_id": cat_id,
                "intent": analysis.get("intent", "unknown"),
                "theme": analysis.get("theme", "")
            }, ensure_ascii=False),
            extra={"function": "process_user_question"}
        )
        
        if rate_limit_until > time.time():
            logger.warning(f"Rate limit active until {rate_limit_until}", extra={"function": "process_user_question"})
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        if analysis.get("direct_response", False) or analysis.get("intent") == "general_query":
            logger.info(
                json.dumps({
                    "event": "skip_thread_fetch",
                    "function": "process_user_question",
                    "query": user_question,
                    "reason": "Direct response or general query, no thread fetching needed",
                    "intent": analysis.get("intent", "unknown")
                }, ensure_ascii=False),
                extra={"function": "process_user_question"}
            )
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        if progress_callback:
            progress_callback("正在抓取帖子列表", 0.1)
        
        post_limit = min(analysis.get("post_limit", 5), 20)
        reply_limit = analysis.get("reply_limit", 100)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 0)
        min_likes = filters.get("min_likes", 0)
        candidate_thread_ids = analysis.get("candidate_thread_ids", [])
        top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
        theme = analysis.get("theme", "")
        theme_keywords = analysis.get("theme_keywords", [])
        
        thread_data = []
        rate_limit_info = []
        previous_thread_ids = previous_thread_ids or []
        
        initial_threads = []
        for page in range(1, 4):
            result = await get_lihkg_topic_list(
                cat_id=cat_id,
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
            items = result.get("items", [])
            initial_threads.extend(items)
            logger.info(
                json.dumps({
                    "event": "thread_fetch",
                    "function": "process_user_question",
                    "cat_id": cat_id,
                    "page": page,
                    "items_fetched": len(items)
                }, ensure_ascii=False),
                extra={"function": "process_user_question"}
            )
            if not items:
                logger.warning(f"No threads fetched for cat_id={cat_id}, page={page}", extra={"function": "process_user_question"})
            if len(initial_threads) >= 90:
                initial_threads = initial_threads[:90]
                break
            if progress_callback:
                progress_callback(f"已抓取第 {page}/3 頁帖子", 0.1 + 0.2 * (page / 3))
        
        if progress_callback:
            progress_callback("正在篩選帖子", 0.3)
        
        async def match_theme_with_grok(thread):
            """
            篩選帖子，基於回覆數和點讚數，準備後續整體比較。
            """
            thread_id = str(thread["thread_id"])
            no_of_reply = thread.get("no_of_reply", 0)
            like_count = int(thread.get("like_count", 0))
            
            if no_of_reply < min_replies or like_count < min_likes or thread_id in previous_thread_ids:
                return None
            
            return {
                "thread": thread,
                "score": no_of_reply * 0.6 + like_count * 0.4
            }
        
        # 並行處理篩選
        filtered_items = []
        tasks = [match_theme_with_grok(thread) for thread in initial_threads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result:
                filtered_items.append(result)
        
        # 按得分排序
        filtered_items = sorted(filtered_items, key=lambda x: x["score"], reverse=True)[:post_limit]
        filtered_threads = [item["thread"] for item in filtered_items]
        
        logger.info(
            json.dumps({
                "event": "thread_filtering_summary",
                "function": "process_user_question",
                "initial_threads": len(initial_threads),
                "filtered_items": len(filtered_items),
                "filters": filters,
                "theme": theme,
                "theme_keywords": theme_keywords,
                "excluded_thread_ids": previous_thread_ids
            }, ensure_ascii=False),
            extra={"function": "process_user_question"}
        )
        
        # 若無符合條件的帖子，放寬篩選條件
        if not filtered_items:
            logger.warning(
                f"No threads meet filters, trying relaxed filters",
                extra={"function": "process_user_question"}
            )
            filtered_threads = []
            for item in initial_threads:
                if str(item["thread_id"]) not in previous_thread_ids:
                    no_of_reply = item.get("no_of_reply", 0)
                    like_count = int(item.get("like_count", 0))
                    if no_of_reply >= min_replies // 2 and like_count >= min_likes // 2:
                        score = no_of_reply * 0.6 + like_count * 0.4
                        filtered_threads.append({"thread": item, "score": score})
            filtered_threads = sorted(filtered_threads, key=lambda x: x["score"], reverse=True)[:post_limit]
            filtered_threads = [item["thread"] for item in filtered_threads]
        
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
        
        if not top_thread_ids and filtered_threads:
            top_thread_ids = [str(item["thread_id"]) for item in filtered_threads]
            logger.info(f"Generated top_thread_ids: {top_thread_ids}", extra={"function": "process_user_question"})
        
        if progress_callback:
            progress_callback("正在抓取帖子內容", 0.6)
        
        # 抓取帖子內容
        for idx, item in enumerate(filtered_threads):
            thread_id = str(item["thread_id"])
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
            logger.info(
                json.dumps({
                    "event": "thread_fetch_raw",
                    "function": "process_user_question",
                    "thread_id": thread_id,
                    "raw_replies": len(replies)
                }, ensure_ascii=False),
                extra={"function": "process_user_question"}
            )
            sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
            thread_data.append({
                "thread_id": thread_id,
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
                "fetched_pages": thread_result.get("fetched_pages", [1])
            })
            st.session_state.thread_cache[thread_id]["data"].update({
                "replies": thread_data[-1]["replies"],
                "fetched_pages": thread_data[-1]["fetched_pages"]
            })
            st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
            logger.info(
                json.dumps({
                    "event": "thread_data_update",
                    "function": "process_user_question",
                    "thread_id": thread_id,
                    "replies_stored": len(sorted_replies),
                    "fetched_pages": thread_result.get("fetched_pages", [])
                }, ensure_ascii=False),
                extra={"function": "process_user_question"}
            )
            if progress_callback:
                progress_callback(f"已抓取帖子 {idx + 1}/{len(filtered_threads)}", 0.6 + 0.3 * ((idx + 1) / len(filtered_threads)))
            await asyncio.sleep(1)
        
        if thread_data:
            logger.info(
                json.dumps({
                    "event": "top_thread_selection",
                    "function": "process_user_question",
                    "query": user_question,
                    "top_thread_ids": [item["thread_id"] for item in thread_data],
                    "details": [
                        {
                            "thread_id": str(item["thread_id"]),
                            "title": item["title"],
                            "no_of_reply": item.get("no_of_reply", 0),
                            "like_count": item.get("like_count", 0)
                        } for item in thread_data
                    ]
                }, ensure_ascii=False),
                extra={"function": "process_user_question"}
            )
        
        logger.info(f"Processing completed: {len(thread_data)} threads for query: {user_question}", extra={"function": "process_user_question"})
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(
            json.dumps({
                "event": "processing_error",
                "function": "process_user_question",
                "query": user_question,
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e)
            }, ensure_ascii=False),
            extra={"function": "process_user_question"}
        )
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data if 'thread_data' in locals() else [],
            "rate_limit_info": rate_limit_info if 'rate_limit_info' in locals() else [{"message": f"Processing failed: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis or {}
        }
