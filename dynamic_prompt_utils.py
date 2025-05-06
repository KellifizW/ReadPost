import json
import logging
import re
import aiohttp
import asyncio
from logging_config import configure_logger
import uuid

# Configure logging
logger = configure_logger(__name__, "dynamic_prompt_utils.log")

# Configuration parameters
CONFIG = {
    "max_prompt_length": 120000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "default_intents": ["contextual_analysis", "semantic_query", "recommend_threads"],
    "error_prompt_template": "在 {selected_cat} 中未找到符合條件的帖子（篩選：{filters}）。建議嘗試其他關鍵詞或討論區！",
    "min_keywords": 1,
    "max_keywords": 8,  # Increased keyword count
    "intent_confidence_threshold": 0.75,
    "default_word_ranges": {
        "contextual_analysis": (500, 800),
        "semantic_query": (500, 800),
        "time_sensitive_analysis": (500, 800),
        "summarize_posts": (600, 1000),
        "analyze_sentiment": (420, 700),
        "follow_up": (1000, 5000),
        "fetch_thread_by_id": (500, 800),
        "general_query": (280, 560),
        "list_titles": (280, 560),
        "find_themed": (420, 1000),
        "fetch_dates": (280, 800),
        "search_keywords": (420, 1000),
        "recommend_threads": (280, 800)
    }
}

async def parse_query(query, conversation_context, grok3_api_key, source_type="lihkg"):
    """
    Parse user query to identify intents, keywords, and relevant thread IDs.
    Supports Reddit and LIHKG, handles time-sensitive queries and follow-ups.
    """
    conversation_context = conversation_context or []
    
    # Extract keywords and related terms
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    keywords = keyword_result.get("keywords", [])
    time_sensitive = keyword_result.get("time_sensitive", False)
    related_terms = keyword_result.get("related_terms", [])
    
    # Check for specific thread ID query
    id_match = re.search(r'(?:ID|帖子)\s*([a-zA-Z0-9]+)', query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        logger.info(f"Detected specific thread ID query: thread_id={thread_id}")
        return {
            "intents": [{"intent": "fetch_thread_by_id", "confidence": 0.95, "reason": f"Detected explicit thread ID {thread_id}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"Detected explicit thread ID {thread_id}",
            "confidence": 0.95
        }
    
    # Check for follow-up query
    thread_id, title, _, match_reason = await extract_relevant_thread(
        conversation_context, query, grok3_api_key
    )
    if thread_id:
        logger.info(f"Detected follow-up query: thread_id={thread_id}, reason={match_reason}")
        return {
            "intents": [{"intent": "follow_up", "confidence": 0.90, "reason": f"Detected follow-up, matched thread ID {thread_id}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"Detected follow-up, matched thread ID {thread_id}, reason: {match_reason}",
            "confidence": 0.90
        }
    
    # Check for time-sensitive query
    time_triggers = ["今晚", "今日", "最近", "今個星期"]
    is_time_sensitive = time_sensitive or any(word in query for word in time_triggers)
    if is_time_sensitive:
        logger.info(f"Detected time-sensitive query: query={query}")
        return {
            "intents": [
                {"intent": "time_sensitive_analysis", "confidence": 0.85, "reason": "Time-sensitive query, prioritize recent posts"},
                {"intent": "semantic_query", "confidence": 0.80, "reason": "Time-sensitive query, requires semantic analysis"}
            ],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "recent",
            "thread_ids": [],
            "reason": "Detected time-sensitive query, prioritize recent posts and semantic analysis",
            "confidence": 0.85
        }
    
    # Check for vague query or list_titles trigger words
    trigger_words = ["列出", "標題", "清單", "所有標題"]
    detected_triggers = [word for word in trigger_words if word in query]
    logger.info(f"Query trigger word detection: query={query}, detected triggers={detected_triggers}")
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦"])
    multi_intent_indicators = ["並且", "同時", "總結並", "列出並", "分析並"]
    has_multi_intent = any(indicator in query for indicator in multi_intent_indicators)
    
    if detected_triggers:
        logger.info(f"Detected list_titles trigger words: {detected_triggers}")
        return {
            "intents": [{"intent": "list_titles", "confidence": 0.95, "reason": f"Detected list_titles trigger words: {detected_triggers}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [],
            "reason": f"Detected list_titles trigger words: {detected_triggers}",
            "confidence": 0.95
        }
    
    # Perform semantic intent analysis
    prompt = f"""
    你是語義分析助手，請分析以下查詢並分類最多2個意圖，考慮對話歷史和關鍵詞。
    查詢：{query}
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    關鍵詞：{json.dumps(keywords, ensure_ascii=False)}
    相關詞：{json.dumps(related_terms, ensure_ascii=False)}
    數據來源：{source_type}
    是否模糊查詢：{is_vague}
    是否包含多意圖指示詞：{has_multi_intent}
    意圖描述：
    {json.dumps({
        "list_titles": "列出帖子標題或清單（觸發詞：列出、標題、清單、所有標題）",
        "contextual_analysis": "綜合總結帖子觀點、推薦相關帖子、分析情緒",
        "semantic_query": "根據語義分析查詢，匹配相關討論",
        "time_sensitive_analysis": "處理時間敏感查詢，優先最新帖子",
        "summarize_posts": "總結帖子內容或討論",
        "analyze_sentiment": "分析帖子或回覆的情緒",
        "general_query": "模糊或非討論區相關問題",
        "find_themed": "尋找特定主題的帖子",
        "fetch_dates": "提取帖子或回覆的日期資料",
        "search_keywords": "根據關鍵詞搜索帖子",
        "recommend_threads": "推薦相關或熱門帖子",
        "follow_up": "追問之前回應的帖子內容",
        "fetch_thread_by_id": "根據明確的帖子 ID 抓取內容"
    }, ensure_ascii=False, indent=2)}
    輸出格式：{{
      "intents": [
        {{"intent": "意圖1", "confidence": 0.0-1.0, "reason": "匹配原因"}},
        ...
      ],
      "reason": "整體匹配原因"
    }}
    若查詢包含「列出」「標題」「清單」「所有標題」，優先返回 list_titles 意圖，信心值設為 0.95。
    若查詢提及「Reddit」或「LIHKG」或子版/分類，優先 contextual_analysis 和 semantic_query。
    若查詢包含時間敏感詞（如「今晚」「今日」），優先 time_sensitive_analysis。
    若查詢模糊且無多意圖指示詞，僅返回1個高信心意圖（優先 list_titles 若包含觸發詞，否則 recommend_threads）。
    若查詢明確或有對話歷史支持，可返回最多2個意圖，但僅包含信心值高於 {CONFIG['intent_confidence_threshold']} 的意圖。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok3_api_key}"
    }
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是語義分析助手，以繁體中文回答，專注於理解用戶意圖。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.5
    }
    
    for attempt in range(CONFIG["max_parse_retries"]):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=CONFIG["parse_timeout"]
                ) as response:
                    if response.status != 200:
                        logger.debug(f"Intent analysis failed: status={response.status}, attempt={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"Intent analysis failed: no choices, attempt={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intents = result.get("intents", [{"intent": "recommend_threads", "confidence": 0.7, "reason": "Vague query, default to recommending popular threads"}])
                    reason = result.get("reason", "Semantic matching")
                    
                    # Filter low-confidence intents
                    intents = [i for i in intents if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                    if not intents:
                        intents = [{"intent": "recommend_threads", "confidence": 0.7, "reason": "No high-confidence intents, default to recommend"}]
                    
                    # For vague queries, limit to a single intent
                    if is_vague and not has_multi_intent:
                        intents = [max(intents, key=lambda x: x["confidence"])]
                    
                    time_range = "recent" if is_time_sensitive else "all"
                    logger.info(f"Intent analysis completed: intents={[i['intent'] for i in intents]}, reason={reason}, confidence={max(i['confidence'] for i in intents)}")
                    return {
                        "intents": intents[:2],
                        "keywords": keywords,
                        "related_terms": related_terms,
                        "time_range": time_range,
                        "thread_ids": [],
                        "reason": reason,
                        "confidence": max(intent["confidence"] for intent in intents) if intents else 0.7
                    }
        except Exception as e:
            logger.debug(f"Intent analysis error: {str(e)}, attempt={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    # Fallback to default intent
    logger.warning(f"Intent analysis failed, falling back to default intent: recommend_threads")
    return {
        "intents": [{"intent": "recommend_threads", "confidence": 0.5, "reason": "Intent analysis failed, default to recommending popular threads"}],
        "keywords": keywords,
        "related_terms": related_terms,
        "time_range": "all",
        "thread_ids": [],
        "reason": "Intent analysis failed, default to recommending popular threads",
        "confidence": 0.5
    }

async def extract_keywords(query, conversation_context, grok3_api_key):
    """
    Extract keywords and related terms from the query, considering Reddit and LIHKG context.
    Detects time-sensitive queries.
    """
    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何"]
    
    prompt = f"""
請從以下查詢提取 {CONFIG['min_keywords']}-{CONFIG['max_keywords']} 個核心關鍵詞（優先名詞或核心動詞，排除通用詞如 {generic_terms}）。
同時生成最多8個語義相關詞（同義詞或討論區術語，如 Reddit 的 YOLO、DD 或 LIHKG 的吹水、高登）。
若查詢包含時間性詞語（如「今晚」「今日」「最近」），設置 time_sensitive 為 true。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
返回格式：
{{
  "keywords": ["關鍵詞1", "關鍵詞2", ...],
  "related_terms": ["相關詞1", "相關詞2", ...],
  "reason": "提取邏輯（70字以內）",
  "time_sensitive": true/false
}}
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok3_api_key}"
    }
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是語義分析助手，以繁體中文回答，專注於提取 Reddit 和 LIHKG 討論區關鍵詞。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.3
    }
    
    for attempt in range(CONFIG["max_parse_retries"]):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=CONFIG["parse_timeout"]
                ) as response:
                    if response.status != 200:
                        logger.debug(f"Keyword extraction failed: status={response.status}, attempt={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"Keyword extraction failed: no choices, attempt={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
                    related_terms = result.get("related_terms", [])
                    return {
                        "keywords": keywords[:CONFIG["max_keywords"]],
                        "related_terms": related_terms[:8],
                        "reason": result.get("reason", "No reason provided")[:70],
                        "time_sensitive": result.get("time_sensitive", False)
                    }
        except Exception as e:
            logger.debug(f"Keyword extraction error: {str(e)}, attempt={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    return {
        "keywords": [],
        "related_terms": [],
        "reason": "Extraction failed",
        "time_sensitive": False
    }

async def extract_relevant_thread(conversation_context, query, grok3_api_key):
    """
    Extract relevant thread from conversation history based on semantic similarity.
    Returns thread_id, title, keywords, and reason for match.
    """
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None, "No conversation history"
    
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = set(keyword_result["keywords"] + keyword_result["related_terms"])
    
    follow_up_phrases = ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩"]
    is_follow_up_query = any(phrase in query for phrase in follow_up_phrases)
    
    # Perform semantic similarity analysis
    prompt = f"""
比較以下查詢和對話歷史，找出最相關的帖子 ID，基於語義相似度。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
關鍵詞：{json.dumps(query_keywords, ensure_ascii=False)}
輸出格式：{{
  "thread_id": "帖子 ID",
  "title": "標題",
  "reason": "匹配原因"
}}
若無匹配，返回空對象 {{}}
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok3_api_key}"
    }
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是語義分析助手，專注於匹配 Reddit 和 LIHKG 帖子。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=CONFIG["parse_timeout"]
            ) as response:
                if response.status != 200:
                    logger.debug(f"Thread extraction failed: status={response.status}")
                    return None, None, None, f"API request failed: status={response.status}"
                data = await response.json()
                if not data.get("choices"):
                    logger.debug("Thread extraction failed: no choices")
                    return None, None, None, "No choices in API response"
                result = json.loads(data["choices"][0]["message"]["content"])
                thread_id = result.get("thread_id")
                title = result.get("title")
                reason = result.get("reason", "No reason provided")
                if thread_id and is_follow_up_query:
                    return thread_id, title, query_keywords, reason
                return None, None, None, "No relevant thread matched"
    except Exception as e:
        logger.debug(f"Thread extraction error: {str(e)}")
        return None, None, None, f"Error during thread extraction: {str(e)}"

async def build_dynamic_prompt(parsed_query, threads_data, query, source_type="lihkg"):
    """
    Build a dynamic prompt for Grok-3 based on parsed query, thread data, and original query.
    Returns a structured prompt with clear response format instructions.
    """
    intents = parsed_query.get("intents", [])
    keywords = parsed_query.get("keywords", [])
    related_terms = parsed_query.get("related_terms", [])
    time_range = parsed_query.get("time_range", "all")
    thread_ids = parsed_query.get("thread_ids", [])
    
    # Handle None intent case
    primary_intent = intents[0]["intent"] if intents else "summarize_posts"
    if not intents:
        logger.warning(f"No intents provided for query={query}, defaulting to summarize_posts")
    
    # Log input parameters
    logger.info(f"Building dynamic prompt: query={query}, intent={primary_intent}, source_type={source_type}")
    
    # Determine word range for response
    word_range = CONFIG["default_word_ranges"].get(primary_intent, (500, 800))
    
    # Base prompt structure
    prompt = f"""
你是社交媒體分析助手，專注於 {source_type} 的討論區（Reddit 或 LIHKG）。
用戶查詢：{query}
根據以下解析的查詢和帖子數據，生成結構化回應，確保內容清晰、簡潔，且不重複。
查詢意圖：{json.dumps(intents, ensure_ascii=False)}
關鍵詞：{json.dumps(keywords, ensure_ascii=False)}
相關詞：{json.dumps(related_terms, ensure_ascii=False)}
時間範圍：{time_range}
帖子 ID：{json.dumps(thread_ids, ensure_ascii=False)}
帖子數據：{json.dumps(threads_data, ensure_ascii=False)}
回應要求：
1. 總字數控制在 {word_range[0]}-{word_range[1]} 字。
2. 結構化回應：
   - **簡介**：概述查詢背景和主要發現（約 50-100 字）。
   - **主體**：按主題分段，總結相關討論、觀點或情緒。每段包含 1-2 句關鍵引述（若適用）。
   - **表格**：列出相關帖子（包含 ID、標題、發帖時間、情緒分數）。
3. 若為時間敏感查詢（time_range="recent"），僅包含過去 24 小時的帖子，並在主體中明確提及回覆時間。
4. 若為追問（intent="follow_up"），聚焦於指定帖子 ID 的深入分析。
5. 若無匹配帖子，使用錯誤模板：{CONFIG['error_prompt_template']}。
6. 使用繁體中文，確保語氣自然，符合討論區文化。
"""
    
    # Intent-specific instructions
    if primary_intent == "time_sensitive_analysis":
        prompt += """
7. 優先處理最新帖子，確保回應突出「今日」「今晚」等時間敏感內容。
8. 在表格中新增「回覆時間」欄，顯示最新回覆的時間戳。
"""
    elif primary_intent == "contextual_analysis":
        prompt += """
7. 提供情緒分佈（正面、中性、負面）並推薦 2-3 個相關帖子。
8. 每段按主題分組（如「正面觀點」「負面觀點」）。
"""
    elif primary_intent == "list_titles":
        prompt += """
7. 僅列出帖子標題和 ID，格式為編號清單。
8. 無需主體或表格，直接返回清單。
"""
    elif primary_intent == "follow_up":
        prompt += """
7. 聚焦於指定帖子 ID，深入分析回覆內容、用戶觀點和情緒。
8. 提供至少 3 個關鍵回覆的引述（包含用戶名和時間戳）。
"""
    elif primary_intent == "summarize_posts":
        prompt += """
7. 總結帖子內容，突出主要討論主題和用戶觀點。
8. 提供 2-3 個關鍵回覆的引述（若適用）。
"""
    
    # Ensure prompt length is within limits
    prompt = prompt[:CONFIG["max_prompt_length"]]
    logger.info(f"Dynamic prompt built: intent={primary_intent}, word_range={word_range}, length={len(prompt)}")
    return prompt