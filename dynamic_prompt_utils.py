import json
import logging
import re
import aiohttp
import asyncio
from logging_config import configure_logger

# 配置日誌
logger = configure_logger(__name__, "dynamic_prompt_utils.log")

# 配置參數
CONFIG = {
    "max_prompt_length": 100000,  # 提示最大長度（token）
    "max_parse_retries": 3,       # 查詢解析最大重試次數
    "parse_timeout": 90,          # 查詢解析超時時間（秒）
    "default_intents": [
        "summarize_posts", "analyze_sentiment", "general_query", 
        "follow_up", "fetch_thread_by_id", "search_keywords", 
        "find_themed", "fetch_dates", "recommend_threads", 
        "compare_threads", "monitor_events"
    ],  # 擴展默認意圖
    "error_prompt_template": "在 {selected_cat} 中未找到符合條件的帖子（篩選：{filters}）。建議嘗試其他關鍵詞或討論區！",
    "min_keywords": 1,            # 最小關鍵詞數量
    "max_keywords": 3,            # 最大關鍵詞數量
    "default_word_ranges": {
        "summarize_posts": (600, 1000),
        "analyze_sentiment": (420, 700),
        "follow_up": (700, 2100),
        "fetch_thread_by_id": (500, 800),
        "general_query": (280, 560),
        "search_keywords": (420, 1000),
        "find_themed": (420, 1000),
        "fetch_dates": (280, 800),
        "recommend_threads": (280, 800),
        "compare_threads": (560, 1200),
        "monitor_events": (420, 1000)
    }
}

async def parse_query(query, conversation_context, grok3_api_key, source_type="lihkg"):
    """
    解析用戶查詢，提取意圖（支持多意圖）、關鍵詞、時間範圍和上下文。
    返回結構化 JSON 結果。
    """
    conversation_context = conversation_context or []
    
    # 提取關鍵詞
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    keywords = keyword_result.get("keywords", [])
    time_sensitive = keyword_result.get("time_sensitive", False)
    
    # 檢查是否為特定帖子 ID 查詢
    id_match = re.search(r'(?:ID|帖子)\s*(\d+)', query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        return {
            "intents": ["fetch_thread_by_id"],
            "keywords": keywords,
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
        return {
            "intents": ["follow_up"],
            "keywords": keywords,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"檢測到追問，匹配帖子 ID {thread_id}，原因：{match_reason}",
            "confidence": 0.90
        }
    
    # 語義意圖分析（支持多意圖）
    prompt = f"""
你是語義分析助手，請分析以下查詢並分類意圖（可返回多個意圖），考慮對話歷史和關鍵詞。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
關鍵詞：{json.dumps(keywords, ensure_ascii=False)}
數據來源：{source_type}
意圖描述：
{json.dumps({
    "list_titles": "列出帖子標題或清單",
    "summarize_posts": "總結帖子內容或討論",
    "analyze_sentiment": "分析帖子或回覆的情緒",
    "general_query": "模糊或非討論區相關問題",
    "find_themed": "尋找特定主題的帖子",
    "fetch_dates": "提取帖子或回覆的日期資料",
    "search_keywords": "根據關鍵詞搜索帖子",
    "recommend_threads": "推薦相關或熱門帖子",
    "follow_up": "追問之前回應的帖子內容",
    "fetch_thread_by_id": "根據明確的帖子 ID 抓取內容",
    "compare_threads": "比較多個帖子的內容或觀點",
    "monitor_events": "監控特定事件或話題的討論"
}, ensure_ascii=False, indent=2)}
示例：
1. 查詢：「網民點睇某股票？有咩分析？」 -> intents: ["summarize_posts", "analyze_sentiment"], reason: "要求總結網民觀點並分析情緒"
2. 查詢：「最近有咩熱門話題？」 -> intents: ["recommend_threads"], reason: "尋求熱門帖子推薦"
輸出格式：{{
  "intents": ["意圖1", "意圖2", ...],
  "confidence": 0.0-1.0,
  "reason": "匹配原因"
}}
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok3_api_key}"
    }
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是語義分析助手，以繁體中文回答，專注於理解用戶意圖，支持多意圖分類。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 250,  # 增加 max_tokens 以支持多意圖
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
                    logger.info(f"意圖分析回應：{json.dumps(data, ensure_ascii=False)}")
                    if not data.get("choices"):
                        logger.debug(f"意圖分析失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intents = result.get("intents", ["summarize_posts"])
                    confidence = result.get("confidence", 0.7)
                    reason = result.get("reason", "語義匹配")
                    
                    time_range = "recent" if time_sensitive else "all"
                    # 二次確認低置信度結果
                    if confidence < 0.9 and attempt < CONFIG["max_parse_retries"] - 1:
                        logger.debug(f"置信度低（{confidence}），重試意圖分析")
                        continue
                    return {
                        "intents": intents,
                        "keywords": keywords,
                        "time_range": time_range,
                        "thread_ids": [],
                        "reason": reason,
                        "confidence": confidence
                    }
        except Exception as e:
            logger.debug(f"意圖分析錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    # 回退到默認意圖
    return {
        "intents": ["summarize_posts"],
        "keywords": keywords,
        "time_range": "all",
        "thread_ids": [],
        "reason": "意圖分析失敗，默認總結帖子",
        "confidence": 0.5
    }

async def extract_keywords(query, conversation_context, grok3_api_key):
    """
    提取查詢中的關鍵詞和時間敏感性。
    """
    prompt = f"""
請從以下查詢提取 {CONFIG['min_keywords']}-{CONFIG['max_keywords']} 個核心關鍵詞（僅保留名詞或核心動詞）。
若查詢包含時間性詞語（如「今晚」「今日」「最近」），設置 time_sensitive 為 true。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
返回格式：
{{
  "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
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
            {"role": "system", "content": "你是語義分析助手，以繁體中文回答，專注於提取關鍵詞。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
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
                        logger.debug(f"關鍵詞提取失敗：狀態碼={response.status}，� Attempt={attempt + 1}")
                        continue
                    data = await response.json()
                    logger.info(f"關鍵詞提取回應：{json.dumps(data, ensure_ascii=False)}")
                    if not data.get("choices"):
                        logger.debug(f"關鍵詞提取失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    return {
                        "keywords": result.get("keywords", [])[:CONFIG["max_keywords"]],
                        "reason": result.get("reason", "未提供原因")[:70],
                        "time_sensitive": result.get("time_sensitive", False)
                    }
        except Exception as e:
            logger.debug(f"關鍵詞提取錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    return {"keywords": [], "reason": "提取失敗", "time_sensitive": False}

async def extract_relevant_thread(conversation_context, query, grok3_api_key):
    """
    從對話歷史中提取相關帖子 ID。
    """
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None, "無對話歷史"
    
    query_keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = set(query_keyword_result["keywords"])
    
    follow_up_phrases = ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩", "係講D咩"]
    is_follow_up_query = any(phrase in query for phrase in follow_up_phrases)
    
    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: (\d+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keyword_result = await extract_keywords(title, conversation_context, grok3_api_key)
                title_keywords = set(title_keyword_result["keywords"])
                common_keywords = query_keywords.intersection(title_keywords)
                content_contains_keywords = any(kw.lower() in message["content"].lower() for kw in query_keywords)
                if common_keywords or content_contains_keywords or is_follow_up_query:
                    return thread_id, title, message["content"], f"關鍵詞匹配：{common_keywords or query_keywords}, 追問詞：{is_follow_up_query}"
    
    return None, None, None, "無匹配貼文"

async def build_dynamic_prompt(query, conversation_context, metadata, thread_data, filters, intent, selected_source, grok3_api_key):
    """
    動態構建提示，根據查詢解析結果、上下文和數據生成，支持多意圖。
    """
    parsed_query = await parse_query(query, conversation_context, grok3_api_key)
    intents = parsed_query.get("intents", [intent or "summarize_posts"])
    keywords = parsed_query["keywords"]
    time_range = parsed_query["time_range"]
    
    # 系統指令
    system = (
        "你是社交媒體討論區（包括 LIHKG 和 Reddit）的數據助手，以繁體中文回答，"
        "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
        "禁止使用 [post_id: ...] 格式。"
    )
    
    # 上下文
    context = (
        f"用戶問題：{query}\n"
        f"討論區：{selected_source}\n"
        f"對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}"
    )
    
    # 數據
    data = (
        f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n"
        f"帖子內容：{json.dumps(thread_data, ensure_ascii=False)}\n"
        f"篩選條件：{json.dumps(filters, ensure_ascii=False)}"
    )
    
    # 動態指導語
    instructions = ["任務："]
    word_min, word_max = (420, 1000)  # 默認字數範圍
    for intent in intents:
        if intent in CONFIG["default_word_ranges"]:
            intent_min, intent_max = CONFIG["default_word_ranges"][intent]
            word_min = max(word_min, intent_min)
            word_max = max(word_max, intent_max)
    
    # 多意圖指令生成
    for intent in intents:
        if intent == "summarize_posts":
            instructions.append(
                f"總結最多5個帖子的討論內容，聚焦關鍵詞 {keywords}，引用高點讚回覆。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "analyze_sentiment":
            instructions.append(
                f"分析最多5個帖子的情緒（正面、中立、負面），提供情緒比例和代表性回覆。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "follow_up":
            instructions.append(
                f"深入分析對話歷史中的帖子，聚焦問題關鍵詞 {keywords}，引用高點讚或最新回覆。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "fetch_thread_by_id":
            instructions.append(
                f"根據提供的帖子 ID 總結內容，引用高點讚或最新回覆。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "general_query":
            instructions.append(
                f"提供與問題相關的簡化總結或回答，基於元數據推測話題。"
            )
        elif intent == "search_keywords":
            instructions.append(
                f"根據關鍵詞 {keywords} 搜索相關帖子，總結匹配內容。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "find_themed":
            instructions.append(
                f"尋找與主題 {keywords} 相關的帖子，總結討論內容。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "fetch_dates":
            instructions.append(
                f"提取最多5個帖子的日期資料，聚焦關鍵詞 {keywords}。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "recommend_threads":
            instructions.append(
                f"推薦與關鍵詞 {keywords} 相關的熱門或高質量帖子。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "compare_threads":
            instructions.append(
                f"比較最多5個帖子的內容或觀點，聚焦關鍵詞 {keywords}。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        elif intent == "monitor_events":
            instructions.append(
                f"監控與關鍵詞 {keywords} 相關的事件或話題討論，提供最新進展。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
        else:
            instructions.append(
                f"根據意圖 {intent} 處理數據，聚焦關鍵詞 {keywords}，提供相關總結或分析。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。"
            )
    
    instructions.append(f"總字數：{word_min}-{word_max}字。")
    
    # 錯誤處理
    instructions.append(
        f"若無匹配帖子，回應：「{CONFIG['error_prompt_template'].format(selected_cat=selected_source, filters=json.dumps(filters))}」"
    )
    
    # 時間範圍
    if time_range != "all":
        instructions.append(f"僅考慮 {time_range} 內的帖子。")
    
    prompt = (
        f"[System]\n{system}\n"
        f"[Context]\n{context}\n"
        f"[Data]\n{data}\n"
        f"[Instructions]\n{'\n'.join(instructions)}"
    )
    
    if len(prompt) > CONFIG["max_prompt_length"]:
        logger.warning("提示長度超過限制，縮減數據")
        thread_data = thread_data[:2]  # 限制帖子數量
        data = f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n帖子內容：{json.dumps(thread_data, ensure_ascii=False)}\n篩選條件：{json.dumps(filters, ensure_ascii=False)}"
        prompt = (
            f"[System]\n{system}\n"
            f"[Context]\n{context}\n"
            f"[Data]\n{data}\n"
            f"[Instructions]\n{'\n'.join(instructions)}"
        )
    
    return prompt

def format_context(conversation_context):
    """
    格式化對話歷史，簡化為結構化 JSON。
    """
    try:
        return [
            {"role": msg["role"], "content": msg["content"][:500]}
            for msg in conversation_context
        ]
    except Exception as e:
        logger.error(f"格式化對話歷史失敗：{str(e)}")
        return []

def extract_thread_metadata(metadata):
    """
    提取帖子元數據的關鍵字段。
    """
    try:
        return [
            {
                "thread_id": item["thread_id"],
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "like_count": item.get("like_count", 0)
            }
            for item in metadata
        ]
    except Exception as e:
        logger.error(f"提取元數據失敗：{str(e)}")
        return []