import json
import logging
import re
import aiohttp
import asyncio
from logging_config import configure_logger
import uuid

# 配置日誌
logger = configure_logger(__name__, "dynamic_prompt_utils.log")

# 配置參數
CONFIG = {
    "max_prompt_length": 120000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "default_intents": ["recommend_threads", "summarize_posts", "contextual_analysis", "semantic_query", "time_sensitive_analysis"],
    "error_prompt_template": "在 {selected_cat} 中未找到符合條件的帖子（篩選：{filters}）。建議嘗試其他關鍵詞或討論區！",
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75,
    "default_word_ranges": {
        "summarize_posts": (600, 1000),
        "analyze_sentiment": (420, 700),
        "recommend_threads": (280, 800),
        "contextual_analysis": (500, 800),
        "semantic_query": (500, 800),
        "time_sensitive_analysis": (500, 800),
        "follow_up": (1000, 5000),
        "fetch_thread_by_id": (500, 800),
        "general_query": (280, 560),
        "list_titles": (280, 560),
        "find_themed": (420, 1000),
        "fetch_dates": (280, 800),
        "search_keywords": (420, 1000)
    }
}

async def parse_query(query, conversation_context, grok3_api_key, source_type="lihkg"):
    conversation_context = conversation_context or []
    
    # 提取關鍵詞和相關詞
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    keywords = keyword_result.get("keywords", [])
    time_sensitive = keyword_result.get("time_sensitive", False)
    related_terms = keyword_result.get("related_terms", [])
    
    # 檢查是否為特定帖子 ID 查詢（支持數字和字母數字 ID）
    id_match = re.search(r'(?:ID|帖子)\s*([a-zA-Z0-9]+)', query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        logger.info(f"檢測到特定帖子 ID 查詢：thread_id={thread_id}")
        return {
            "intents": [{"intent": "fetch_thread_by_id", "confidence": 0.95, "reason": f"檢測到明確帖子 ID {thread_id}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"檢測到明確帖子 ID {thread_id}",
            "confidence": 0.95
        }
    
    # 檢查是否為追問
    thread_id, _, _, match_reason = await extract_relevant_thread(
        conversation_context, query, grok3_api_key
    )
    if thread_id:
        logger.info(f"檢測到追問：thread_id={thread_id}, 原因={match_reason}")
        return {
            "intents": [{"intent": "follow_up", "confidence": 0.90, "reason": f"檢測到追問，匹配帖子 ID {thread_id}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"檢測到追問，匹配帖子 ID {thread_id}，原因：{match_reason}",
            "confidence": 0.90
        }
    
    # 檢查是否為時間敏感查詢
    time_triggers = ["今晚", "今日", "最近", "今個星期"]
    is_time_sensitive = time_sensitive or any(word in query for word in time_triggers)
    if is_time_sensitive:
        logger.info(f"檢測到時間敏感查詢：query={query}")
        return {
            "intents": [
                {"intent": "time_sensitive_analysis", "confidence": 0.85, "reason": "時間敏感查詢，需優先最新帖子"},
                {"intent": "semantic_query", "confidence": 0.80, "reason": "時間敏感查詢，需語義分析"}
            ],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "recent",
            "thread_ids": [],
            "reason": "檢測到時間敏感查詢，優先最新帖子和語義分析",
            "confidence": 0.85
        }
    
    # 檢查是否模糊查詢並檢測 list_titles 觸發詞
    trigger_words = ["列出", "標題", "清單", "所有標題"]
    detected_triggers = [word for word in trigger_words if word in query]
    logger.info(f"查詢觸發詞檢測：query={query}, 檢測到觸發詞={detected_triggers}")
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦"])
    multi_intent_indicators = ["並且", "同時", "總結並", "列出並", "分析並"]
    has_multi_intent = any(indicator in query for indicator in multi_intent_indicators)
    
    # 語義意圖分析
    prompt = f"""
    你是語義分析助手，請分析以下查詢並分類最多3個意圖，考慮對話歷史和關鍵詞。
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
        "summarize_posts": "總結帖子內容或討論",
        "analyze_sentiment": "分析帖子或回覆的情緒",
        "general_query": "模糊或非討論區相關問題",
        "find_themed": "尋找特定主題的帖子",
        "fetch_dates": "提取帖子或回覆的日期資料",
        "search_keywords": "根據關鍵詞搜索帖子",
        "recommend_threads": "推薦相關或熱門帖子",
        "contextual_analysis": "綜合總結帖子觀點、推薦相關帖子、分析情緒",
        "semantic_query": "根據語義分析查詢，匹配相關討論",
        "time_sensitive_analysis": "處理時間敏感查詢，優先最新帖子",
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
    若查詢明確或有對話歷史支持，可返回最多3個意圖，但僅包含信心值高於 {CONFIG['intent_confidence_threshold']} 的意圖。
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
                        logger.debug(f"意圖分析失敗：狀態碼={response.status}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"意圖分析失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intents = result.get("intents", [{"intent": "recommend_threads", "confidence": 0.7, "reason": "模糊查詢，默認推薦熱門帖子"}])
                    reason = result.get("reason", "語義匹配")
                    
                    # 強制 list_titles 若檢測到觸發詞
                    if detected_triggers:
                        intents = [{"intent": "list_titles", "confidence": 0.95, "reason": f"檢測到 list_titles 觸發詞：{detected_triggers}"}]
                        reason = f"檢測到 list_titles 觸發詞：{detected_triggers}"
                    
                    # 過濾低信心意圖
                    intents = [i for i in intents if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                    if not intents:
                        intents = [{"intent": "recommend_threads", "confidence": 0.7, "reason": "無高信心意圖，默認推薦"}]
                    
                    # 對於模糊查詢，限制為單一意圖
                    if is_vague and not has_multi_intent:
                        intents = [max(intents, key=lambda x: x["confidence"])]
                    
                    time_range = "recent" if is_time_sensitive else "all"
                    logger.info(f"意圖分析完成：intents={[i['intent'] for i in intents]}, reason={reason}, confidence={max(i['confidence'] for i in intents)}")
                    return {
                        "intents": intents[:3],
                        "keywords": keywords,
                        "related_terms": related_terms,
                        "time_range": time_range,
                        "thread_ids": [],
                        "reason": reason,
                        "confidence": max(intent["confidence"] for intent in intents) if intents else 0.7
                    }
        except Exception as e:
            logger.debug(f"意圖分析錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    # 回退到默認意圖
    logger.warning(f"意圖分析失敗，回退到默認意圖：recommend_threads")
    return {
        "intents": [{"intent": "recommend_threads", "confidence": 0.5, "reason": "意圖分析失敗，默認推薦熱門帖子"}],
        "keywords": keywords,
        "related_terms": related_terms,
        "time_range": "all",
        "thread_ids": [],
        "reason": "意圖分析失敗，默認推薦熱門帖子",
        "confidence": 0.5
    }

async def extract_keywords(query, conversation_context, grok3_api_key):
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
                        logger.debug(f"關鍵詞提取失敗：狀態碼={response.status}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"關鍵詞提取失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
                    related_terms = result.get("related_terms", [])
                    return {
                        "keywords": keywords[:CONFIG["max_keywords"]],
                        "related_terms": related_terms[:8],
                        "reason": result.get("reason", "未提供原因")[:70],
                        "time_sensitive": result.get("time_sensitive", False)
                    }
        except Exception as e:
            logger.debug(f"關鍵詞提取錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    return {
        "keywords": [],
        "related_terms": [],
        "reason": "提取失敗",
        "time_sensitive": False
    }

async def extract_relevant_thread(conversation_context, query, grok3_api_key):
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None, "無對話歷史"
    
    query_keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = set(query_keyword_result["keywords"] + query_keyword_result["related_terms"])
    
    follow_up_phrases = ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩"]
    is_follow_up_query = any(phrase in query for phrase in follow_up_phrases)
    
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
                    logger.debug(f"相關帖子提取失敗：狀態碼={response.status}")
                    return None, None, None, "API 請求失敗"
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                thread_id = result.get("thread_id")
                title = result.get("title")
                reason = result.get("reason", "未提供原因")
                if thread_id:
                    return thread_id, title, None, reason
                return None, None, None, "無匹配帖子"
    except Exception as e:
        logger.debug(f"相關帖子提取錯誤：{str(e)}")
        return None, None, None, f"提取錯誤：{str(e)}"

async def build_dynamic_prompt(query, conversation_context, metadata, thread_data, filters, intents, selected_source, grok3_api_key):
    logger.info(f"生成動態提示：查詢={query}, intents={intents}")
    
    if not isinstance(intents, list) or not intents:
        logger.warning(f"無效的 intents 格式：預期非空列表，得到 {type(intents)}，回退到 ['summarize_posts']")
        intents = ["summarize_posts"]
    
    parsed_query = await parse_query(query, conversation_context, grok3_api_key, selected_source.get("source_type", "lihkg"))
    parsed_intents = parsed_query["intents"]
    keywords = parsed_query["keywords"]
    related_terms = parsed_query["related_terms"]
    time_range = parsed_query["time_range"]
    
    # 合併並去重 intents，保留 parsed_query 中的高信心意圖
    intent_set = set(intent["intent"] for intent in parsed_intents)
    for intent in intents:
        if intent not in intent_set:
            parsed_intents.append({"intent": intent, "confidence": 0.7, "reason": "從外部傳入的意圖"})
    intents = parsed_intents[:3]  # 限制最多3個意圖
    
    system = (
        "你是社交媒體討論區（包括 LIHKG 和 Reddit）的數據助手，以繁體中文回答，"
        "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
        "禁止使用 [post_id: ...] 格式。根據用戶意圖動態選擇回應格式（例如列表、段落、表格等），"
        "確保結構清晰、內容連貫，且適合查詢的需求。"
    )
    
    context = (
        f"用戶問題：{query}\n"
        f"討論區：{selected_source}\n"
        f"對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}"
    )
    
    data = (
        f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n"
        f"帖子內容：{json.dumps(thread_data, ensure_ascii=False)}\n"
        f"篩選條件：{json.dumps(filters, ensure_ascii=False)}"
    )
    
    # 動態計算字數範圍
    word_min = 280
    word_max = 5000
    for intent_info in intents:
        intent = intent_info["intent"]
        min_w, max_w = CONFIG["default_word_ranges"].get(intent, (280, 5000))
        word_min = max(word_min, min_w)
        word_max = min(word_max, max_w)
    
    prompt_length = len(context) + len(data) + len(system) + 500
    length_factor = min(prompt_length / CONFIG["max_prompt_length"], 1.0)
    word_min = int(word_min + (word_max - word_min) * length_factor * 0.3)
    word_max = int(word_min + (word_max - word_min) * (1 + length_factor * 0.7))
    
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦"])
    
    # 動態生成指令並指定回應格式
    instruction_parts = []
    format_instructions = []
    for intent_info in intents:
        intent = intent_info["intent"]
        if intent == "summarize_posts":
            instruction_parts.append(
                f"總結最多5個帖子的討論內容，聚焦關鍵詞 {keywords}，引用高點讚回覆，簡明扼要。"
            )
            format_instructions.append(
                "使用連貫段落格式，每個帖子以 [帖子 ID: {thread_id}] 開頭，總結其核心討論。"
            )
        elif intent == "analyze_sentiment":
            instruction_parts.append(
                f"分析最多5個帖子的情緒（正面、中立、負面），提供情緒比例，融入總結。"
            )
            format_instructions.append(
                "使用段落格式，總結情緒分析結果，並以表格形式列出每個帖子的情緒比例。"
            )
        elif intent == "contextual_analysis":
            instruction_parts.append(
                f"綜合最多5個帖子的討論，聚焦關鍵詞 {keywords} 和相關詞 {related_terms}，總結主要觀點，動態識別主題（如正面、負面或其他），引用高點讚、最新和多樣化回覆，分析情緒比例。"
            )
            format_instructions.append(
                "使用連貫段落，按主題分段（每個主題引用不同帖子），結尾附情緒分佈表格，格式為 | 主題 | 比例 | 代表性帖子 |。"
            )
        elif intent == "semantic_query":
            instruction_parts.append(
                f"根據語義分析查詢，匹配最多5個相關帖子，總結討論，突出與關鍵詞 {keywords} 和相關詞 {related_terms} 的語義關聯。"
            )
            format_instructions.append(
                "使用段落格式，每帖子以 [帖子 ID: {thread_id}] 開頭，強調語義相關性。"
            )
        elif intent == "time_sensitive_analysis":
            instruction_parts.append(
                f"優先總結過去24小時的帖子，聚焦關鍵詞 {keywords} 和相關詞 {related_terms}，引用最新回覆，突出時間敏感討論。"
            )
            format_instructions.append(
                "使用段落格式，標註回覆時間，優先引用 [帖子 ID: {thread_id}] 的最新討論。"
            )
        elif intent == "follow_up":
            instruction_parts.append(
                f"深入分析對話歷史中的帖子，聚焦問題關鍵詞 {keywords} 和相關詞 {related_terms}，引用高點讚或最新回覆，補充上下文。"
            )
            format_instructions.append(
                "使用詳細段落格式，聚焦上下文連貫性，必要時分段以突出不同回覆的觀點。"
            )
        elif intent == "fetch_thread_by_id":
            instruction_parts.append(
                f"根據提供的帖子 ID 總結內容，引用高點讚或最新回覆，突出核心討論。"
            )
            format_instructions.append(
                "使用單一段落格式，詳細總結指定帖子的內容，引用 [帖子 ID: {thread_id}]。"
            )
        elif intent == "general_query" and not is_vague:
            instruction_parts.append(
                f"提供與問題相關的簡化總結，基於元數據推測話題，保持簡潔。"
            )
            format_instructions.append(
                "使用簡潔段落格式，直接回答問題，必要時引用相關帖子。"
            )
        elif intent == "list_titles":
            instruction_parts.append(
                f"列出最多15個帖子標題，聚焦關鍵詞 {keywords} 和相關詞 {related_terms}，並簡述其相關性。"
            )
            format_instructions.append(
                "使用有序列表格式，每項包含 [帖子 ID: {thread_id}]、標題和簡短相關性說明。"
            )
        elif intent == "find_themed":
            instruction_parts.append(
                f"尋找與關鍵詞 {keywords} 和相關詞 {related_terms} 相關的帖子，總結其內容，突出主題關聯。"
            )
            format_instructions.append(
                "使用段落格式，每個帖子以 [帖子 ID: {thread_id}] 開頭，強調主題相關性。"
            )
        elif intent == "fetch_dates":
            instruction_parts.append(
                f"提取最多5個帖子的發布或回覆日期，聚焦關鍵詞 {keywords} 和相關詞 {related_terms}，融入總結。"
            )
            format_instructions.append(
                "使用表格格式，列出帖子 ID、標題和日期，後附簡短總結段落。"
            )
        elif intent == "search_keywords":
            instruction_parts.append(
                f"搜索包含關鍵詞 {keywords} 和相關詞 {related_terms} 的帖子，總結其內容，強調關鍵詞匹配。"
            )
            format_instructions.append(
                "使用段落格式，突出關鍵詞匹配，每個帖子以 [帖子 ID: {thread_id}] 開頭。"
            )
        elif intent == "recommend_threads":
            instruction_parts.append(
                f"推薦2-5個熱門或相關帖子，基於回覆數和點讚數，聚焦關鍵詞 {keywords} 和相關詞 {related_terms}。"
            )
            format_instructions.append(
                "使用無序列表格式，每項包含 [帖子 ID: {thread_id}]、標題和推薦理由。"
            )
    
    combined_instruction = (
        f"根據以下意圖綜合生成一個連貫的回應（字數：{word_min}-{word_max}字）：{'；'.join(instruction_parts)}"
        f"回應格式：簡介（1句）+按主題分段（每個主題引用不同帖子和回覆）+結尾總結（如情緒分佈表格或關鍵觀點）。"
        f"選擇最適合的回應格式：{'；'.join(format_instructions)}"
        f"確保回應聚焦用戶問題，綜合所有意圖，內容不重複且流暢，引用帖子時標註 [帖子 ID: {{thread_id}}] {{標題}}。"
        f"優先考慮主要意圖（信心值最高者），其他意圖作為輔助。"
    )
    
    if time_range == "recent":
        combined_instruction += f"僅考慮過去24小時的帖子和回覆。"
    
    prompt = (
        f"[System]\n{system}\n"
        f"[Context]\n{context}\n"
        f"[Data]\n{data}\n"
        f"[Instructions]\n{combined_instruction}"
    )
    
    if len(prompt) > CONFIG["max_prompt_length"]:
        logger.warning("提示長度超過限制，縮減數據")
        thread_data = thread_data[:2]
        data = f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n篩選條件：{json.dumps(filters, ensure_ascii=False)}"
        prompt = (
            f"[System]\n{system}\n"
            f"[Context]\n{context}\n"
            f"[Data]\n{data}\n"
            f"[Instructions]\n{combined_instruction}"
        )
    
    logger.info(f"動態提示生成完成：長度={len(prompt)} 字符，intents={[i['intent'] for i in intents]}")
    return prompt

def format_context(conversation_context):
    try:
        return [
            {"role": msg["role"], "content": msg["content"][:500]}
            for msg in conversation_context
        ]
    except Exception as e:
        logger.error(f"格式化對話歷史失敗：{str(e)}")
        return []

def extract_thread_metadata(metadata):
    try:
        return [
            {
                "thread_id": item["thread_id"],
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "like_count": item.get("like_count", 0),
                "last_reply_time": item.get("last_reply_time", 0)
            }
            for item in metadata
        ]
    except Exception as e:
        logger.error(f"提取元數據失敗：{str(e)}")
        return []