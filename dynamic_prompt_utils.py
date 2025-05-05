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
    "default_intents": ["contextual_analysis", "semantic_query", "time_sensitive_analysis"],
    "error_prompt_template": "在 {selected_cat} 中未找到符合條件的帖子（篩選：{filters}）。建議嘗試其他關鍵詞或討論區！",
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75,
    "default_word_ranges": {
        "contextual_analysis": (500, 800),
        "semantic_query": (500, 800),
        "time_sensitive_analysis": (500, 800),
        "follow_up": (600, 1000),
        "fetch_thread_by_id": (400, 600),
        "list_titles": (200, 400)
    }
}

async def parse_query(query, conversation_context, grok3_api_key, source_type="reddit"):
    conversation_context = conversation_context or []
    
    # 提取語義關鍵詞和相關詞
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    keywords = keyword_result.get("keywords", [])
    time_sensitive = keyword_result.get("time_sensitive", False)
    related_terms = keyword_result.get("related_terms", [])
    
    # 檢查是否為特定帖子 ID 查詢
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
    
    # 檢查是否為標題列舉查詢
    trigger_words = ["列出", "標題", "清單", "所有標題"]
    detected_triggers = [word for word in trigger_words if word in query]
    if detected_triggers:
        logger.info(f"檢測到 list_titles 觸發詞：{detected_triggers}")
        return {
            "intents": [{"intent": "list_titles", "confidence": 0.95, "reason": f"檢測到 list_titles 觸發詞：{detected_triggers}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [],
            "reason": f"檢測到 list_titles 觸發詞：{detected_triggers}",
            "confidence": 0.95
        }
    
    # 語義意圖分析
    prompt = f"""
    你是語義分析助手，請分析以下查詢並分類最多2個意圖，考慮對話歷史和關鍵詞。
    查詢：{query}
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    關鍵詞：{json.dumps(keywords, ensure_ascii=False)}
    相關詞：{json.dumps(related_terms, ensure_ascii=False)}
    數據來源：{source_type}
    意圖描述：
    {json.dumps({
        "list_titles": "列出帖子標題或清單",
        "contextual_analysis": "綜合總結帖子觀點、推薦相關帖子、分析情緒",
        "semantic_query": "根據語義分析查詢，匹配相關討論",
        "time_sensitive_analysis": "處理時間敏感查詢，優先最新帖子",
        "follow_up": "追問之前回應的帖子內容",
        "fetch_thread_by_id": "根據明確帖子 ID 抓取內容"
    }, ensure_ascii=False, indent=2)}
    輸出格式：{{
      "intents": [
        {{"intent": "意圖1", "confidence": 0.0-1.0, "reason": "匹配原因"}},
        ...
      ],
      "reason": "整體匹配原因"
    }}
    若查詢提及「Reddit」或「LIHKG」或子版/分類，優先 contextual_analysis 和 semantic_query。
    若查詢包含時間敏感詞（如「今晚」「今日」），優先 time_sensitive_analysis。
    若對話歷史中提及帖子 ID 或上下文相關，優先 follow_up 意圖。
    僅返回信心值高於 {CONFIG['intent_confidence_threshold']} 的意圖，若無高信心意圖，則根據上下文選擇 follow_up 或 semantic_query。
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
        "max_tokens": 200,
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
                    if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
                        logger.debug(f"意圖分析失敗：缺少有效回應，嘗試次數={attempt + 1}")
                        continue
                    content = data["choices"][0]["message"]["content"]
                    try:
                        result = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON 解析失敗：{str(e)}，原始回應={content}")
                        continue
                    intents = result.get("intents", [])
                    reason = result.get("reason", "語義匹配")
                    
                    # 過濾信心值低於閾值的意圖
                    intents = [i for i in intents if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                    
                    # 若無高信心意圖，檢查上下文是否支持 follow_up
                    if not intents:
                        if conversation_context and any("帖子 ID" in msg.get("content", "") for msg in conversation_context):
                            intents = [{"intent": "follow_up", "confidence": 0.8, "reason": "無高信心意圖，檢測到上下文提及帖子 ID，選擇 follow_up"}]
                        else:
                            intents = [{"intent": "semantic_query", "confidence": 0.7, "reason": "無高信心意圖，默認語義分析"}]
                    
                    time_range = "recent" if is_time_sensitive else "all"
                    logger.info(f"意圖分析完成：intents={[i['intent'] for i in intents]}, reason={reason}, confidence={max(i['confidence'] for i in intents)}")
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
            logger.debug(f"意圖分析錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    # 最終回退邏輯：優先檢查上下文是否支持 follow_up
    if conversation_context and any("帖子 ID" in msg.get("content", "") for msg in conversation_context):
        logger.warning(f"意圖分析失敗，檢測到上下文提及帖子 ID，回退到 follow_up")
        return {
            "intents": [{"intent": "follow_up", "confidence": 0.7, "reason": "意圖分析失敗，檢測到上下文提及帖子 ID，回退到 follow_up"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [],
            "reason": "意圖分析失敗，檢測到上下文提及帖子 ID，回退到 follow_up",
            "confidence": 0.7
        }
    
    logger.warning(f"意圖分析失敗，回退到默認意圖：semantic_query")
    return {
        "intents": [{"intent": "semantic_query", "confidence": 0.5, "reason": "意圖分析失敗，默認語義分析"}],
        "keywords": keywords,
        "related_terms": related_terms,
        "time_range": "all",
        "thread_ids": [],
        "reason": "意圖分析失敗，默認語義分析",
        "confidence": 0.5
    }

async def extract_keywords(query, conversation_context, grok3_api_key):
    generic_terms = ["post", "分享", "什麼", "如何", "有咩", "點樣"]
    
    prompt = f"""
請從以下查詢提取 {CONFIG['min_keywords']}-{CONFIG['max_keywords']} 個核心關鍵詞（優先名詞或動詞，排除通用詞如 {generic_terms}）。
同時生成最多8個語義相關詞（同義詞或討論區術語，如 Reddit 的 YOLO、DD 或 LIHKG 的吹水、高登）。
若查詢包含時間性詞語（如「今晚」「今日」），設置 time_sensitive 為 true。
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
                    if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
                        logger.debug(f"關鍵詞提取失敗：缺少有效回應，嘗試次數={attempt + 1}")
                        continue
                    content = data["choices"][0]["message"]["content"]
                    try:
                        result = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON 解析失敗：{str(e)}，原始回應={content}")
                        continue
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
    
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = keyword_result["keywords"] + keyword_result["related_terms"]
    
    follow_up_phrases = ["詳情", "更多", "進一步", "為什麼", "再講", "繼續", "點解", "講多D", "仲有咩"]
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
                if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
                    logger.debug(f"相關帖子提取失敗：缺少有效回應")
                    return None, None, None, "缺少有效回應"
                content = data["choices"][0]["message"]["content"]
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON 解析失敗：{str(e)}，原始回應={content}")
                    return None, None, None, "JSON 解析失敗"
                thread_id = result.get("thread_id")
                title = result.get("title")
                reason = result.get("reason", "未提供原因")
                if thread_id:
                    return thread_id, title, None, reason
                return None, None, None, "無匹配帖子"
    except Exception as e:
        logger.debug(f"相關帖子提取錯誤：{str(e)}")
        return None, None, None, f"提取錯誤：{str(e)}"

async def build_dynamic_prompt(query, conversation_context, metadata, thread_data, filters, intent, selected_source, grok3_api_key):
    parsed_query = await parse_query(query, conversation_context, grok3_api_key, selected_source.get("source_type", "reddit"))
    intents = parsed_query["intents"]
    keywords = parsed_query["keywords"]
    time_range = parsed_query["time_range"]
    
    system = (
        "你是社交媒體討論區（包括 Reddit 和 LIHKG）的數據助手，以繁體中文回答，"
        "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
        "禁止使用 [post_id: ...] 格式。根據用戶意圖動態選擇回應格式（例如段落、表格），"
        "確保結構清晰、內容連貫、不重複，且適合查詢的需求。"
    )
    
    context = (
        f"用戶問題：{query}\n"
        f"討論區：{selected_source.get('source_name', '未知')}\n"
        f"對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}"
    )
    
    data = (
        f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n"
        f"帖子內容：{json.dumps(thread_data, ensure_ascii=False)}\n"
        f"篩選條件：{json.dumps(filters, ensure_ascii=False)}"
    )
    
    word_min = 500
    word_max = 800
    for intent_info in intents:
        intent = intent_info["intent"]
        min_w, max_w = CONFIG["default_word_ranges"].get(intent, (500, 800))
        word_min = max(word_min, min_w)
        word_max = min(word_max, max_w)
    
    prompt_length = len(context) + len(data) + len(system) + 500
    length_factor = min(prompt_length / CONFIG["max_prompt_length"], 1.0)
    word_min = int(word_min + (word_max - word_min) * length_factor * 0.3)
    word_max = int(word_min + (word_max - word_min) * (1 + length_factor * 0.7))
    
    instruction_parts = []
    format_instructions = []
    for intent_info in intents[:2]:
        intent = intent_info["intent"]
        if intent == "contextual_analysis":
            instruction_parts.append(
                f"綜合最多5個帖子的討論，聚焦關鍵詞 {keywords}，總結主要觀點，動態識別主題（如正面、負面或其他），引用高點讚、最新和多樣化回覆，分析情緒比例。"
            )
            format_instructions.append(
                "使用連貫段落，按主題分段（每個主題引用不同帖子），結尾附情緒分佈表格，格式為 | 主題 | 比例 | 代表性帖子 |。"
            )
        elif intent == "semantic_query":
            instruction_parts.append(
                f"根據語義分析查詢，匹配最多5個相關帖子，總結討論，突出與關鍵詞 {keywords} 的語義關聯。"
            )
            format_instructions.append(
                "使用段落格式，每帖子以 [帖子 ID: {thread_id}] 開頭，強調語義相關性。"
            )
        elif intent == "time_sensitive_analysis":
            instruction_parts.append(
                f"優先總結過去24小時的帖子，聚焦關鍵詞 {keywords}，引用最新回覆，突出時間敏感討論。"
            )
            format_instructions.append(
                "使用段落格式，標註回覆時間，優先引用 [帖子 ID: {thread_id}] 的最新討論。"
            )
        elif intent == "follow_up":
            instruction_parts.append(
                f"深入分析對話歷史中的帖子，聚焦問題關鍵詞 {keywords}，引用高點讚或最新回覆，補充上下文。"
            )
            format_instructions.append(
                "使用詳細段落，聚焦上下文連貫性，必要時分段以突出不同回覆的觀點。"
            )
        elif intent == "fetch_thread_by_id":
            instruction_parts.append(
                f"根據提供的帖子 ID 總結內容，引用高點讚或最新回覆，突出核心討論。"
            )
            format_instructions.append(
                "使用單一段落，詳細總結指定帖子的內容，引用 [帖子 ID: {thread_id}]。"
            )
        elif intent == "list_titles":
            instruction_parts.append(
                f"列出最多15個帖子標題，聚焦關鍵詞 {keywords}，並簡述其相關性。"
            )
            format_instructions.append(
                "使用有序列表，每項包含 [帖子 ID: {thread_id}]、標題和簡短相關性說明。"
            )
    
    combined_instruction = (
        f"根據以下意圖生成連貫回應（字數：{word_min}-{word_max}字）：{'；'.join(instruction_parts)}"
        f"回應格式：簡介（1句）+按主題分段（每個主題引用不同帖子和回覆）+情緒分佈表格。"
        f"選擇最適合的格式：{'；'.join(format_instructions)}"
        f"確保內容不重複、流暢，每帖子引用一次，引用格式為 [帖子 ID: {{thread_id}}] {{標題}}。"
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
