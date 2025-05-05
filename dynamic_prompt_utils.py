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
    "max_prompt_length": 100000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "default_intents": ["recommend_threads", "summarize_posts"],
    "error_prompt_template": "在 {selected_cat} 中未找到符合條件的帖子（篩選：{filters}）。建議嘗試其他關鍵詞或討論區！",
    "min_keywords": 1,
    "max_keywords": 3,
    "intent_confidence_threshold": 0.75,
    "default_word_ranges": {
        "summarize_posts": (600, 1000),
        "analyze_sentiment": (420, 700),
        "follow_up": (700, 2100),
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
    解析用戶查詢，提取最多3個意圖，優先選擇高信心意圖，對於模糊查詢限制意圖數量。
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
            "intents": [{"intent": "fetch_thread_by_id", "confidence": 0.95, "reason": f"檢測到明確帖子 ID {thread_id}"}],
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
            "intents": [{"intent": "follow_up", "confidence": 0.90, "reason": f"檢測到追問，匹配帖子 ID {thread_id}"}],
            "keywords": keywords,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"檢測到追問，匹配帖子 ID {thread_id}，原因：{match_reason}",
            "confidence": 0.90
        }
    
    # 檢查查詢是否模糊
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦"])
    multi_intent_indicators = ["並且", "同時", "總結並", "列出並", "分析並"]
    has_multi_intent = any(indicator in query for indicator in multi_intent_indicators)
    
    # 語義意圖分析
    prompt = f"""
你是語義分析助手，請分析以下查詢並分類最多3個意圖，考慮對話歷史和關鍵詞。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
關鍵詞：{json.dumps(keywords, ensure_ascii=False)}
數據來源：{source_type}
是否模糊查詢：{is_vague}
是否包含多意圖指示詞：{has_multi_intent}
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
    "fetch_thread_by_id": "根據明確的帖子 ID 抓取內容"
}, ensure_ascii=False, indent=2)}
輸出格式：{{
  "intents": [
    {{"intent": "意圖1", "confidence": 0.0-1.0, "reason": "匹配原因"}},
    ...
  ],
  "reason": "整體匹配原因"
}}
若查詢模糊且無多意圖指示詞，僅返回1個高信心意圖（優先 recommend_threads 或 summarize_posts）。
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
                    
                    # 過濾低信心意圖
                    intents = [i for i in intents if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                    if not intents:
                        intents = [{"intent": "recommend_threads", "confidence": 0.7, "reason": "無高信心意圖，默認推薦"}]
                    
                    # 對於模糊查詢，限制為單一意圖
                    if is_vague and not has_multi_intent:
                        intents = [max(intents, key=lambda x: x["confidence"])]
                    
                    time_range = "recent" if time_sensitive else "all"
                    return {
                        "intents": intents[:3],
                        "keywords": keywords,
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
    return {
        "intents": [{"intent": "recommend_threads", "confidence": 0.5, "reason": "意圖分析失敗，默認推薦熱門帖子"}],
        "keywords": keywords,
        "time_range": "all",
        "thread_ids": [],
        "reason": "意圖分析失敗，默認推薦熱門帖子",
        "confidence": 0.5
    }

async def extract_keywords(query, conversation_context, grok3_api_key):
    """
    提取查詢中的關鍵詞，過濾通用詞並考慮語義意圖。
    """
    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何"]
    
    prompt = f"""
請從以下查詢提取 {CONFIG['min_keywords']}-{CONFIG['max_keywords']} 個核心關鍵詞（優先名詞或核心動詞，排除通用詞如 {generic_terms}）。
若查詢包含時間性詞語（如「今晚」「今日」「最近」），設置 time_sensitive 為 true。
若查詢模糊（如「有咩post 分享」），根據對話歷史推測語義意圖（如尋找熱門或可分享內容）。
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
                        logger.debug(f"關鍵詞提取失敗：狀態碼={response.status}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"關鍵詞提取失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
                    return {
                        "keywords": keywords[:CONFIG["max_keywords"]],
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
    動態構建提示，根據查詢解析結果、上下文和數據生成，限制模糊查詢的多意圖輸出。
    """
    parsed_query = await parse_query(query, conversation_context, grok3_api_key)
    intents = parsed_query["intents"]
    keywords = parsed_query["keywords"]
    time_range = parsed_query["time_range"]
    
    # 系統指令
    system = (
        "你是社交媒體討論區（包括 LIHKG 和 Reddit）的數據助手，以繁體中文回答，"
        "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
        "禁止使用 [post_id: ...] 格式。每個任務的回應以 ### 標題分隔，例如 ### 總結, ### 情緒分析。"
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
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦"])
    
    for intent_info in intents[:3]:
        intent = intent_info["intent"]
        word_min, word_max = CONFIG["default_word_ranges"].get(intent, (420, 1000))
        if intent == "summarize_posts":
            instructions.append(
                f"### 總結\n總結最多5個帖子的討論內容，聚焦關鍵詞 {keywords}，引用高點讚回覆。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "analyze_sentiment":
            instructions.append(
                f"### 情緒分析\n分析最多5個帖子的情緒（正面、中立、負面），提供情緒比例。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "follow_up":
            instructions.append(
                f"### 追問\n深入分析對話歷史中的帖子，聚焦問題關鍵詞 {keywords}，引用高點讚或最新回覆。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "fetch_thread_by_id":
            instructions.append(
                f"### 特定帖子\n根據提供的帖子 ID 總結內容，引用高點讚或最新回覆。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "general_query" and not is_vague:
            instructions.append(
                f"### 一般查詢\n提供與問題相關的簡化總結或回答，基於元數據推測話題。"
                f"字數：{word_min}-{word_max}字。"
            )
        elif intent == "list_titles":
            instructions.append(
                f"### 列出標題\n列出最多5個帖子的標題，聚焦關鍵詞 {keywords}。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "find_themed":
            instructions.append(
                f"### 主題搜索\n尋找與關鍵詞 {keywords} 相關的帖子，總結其內容。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "fetch_dates":
            instructions.append(
                f"### 日期提取\n提取最多5個帖子的發布或回覆日期，聚焦關鍵詞 {keywords}。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "search_keywords":
            instructions.append(
                f"### 關鍵詞搜索\n搜索包含關鍵詞 {keywords} 的帖子，總結其內容。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
        elif intent == "recommend_threads":
            instructions.append(
                f"### 推薦帖子\n推薦2-3個熱門或相關帖子，基於回覆數和點讚數，聚焦熱門或可分享內容。"
                f"每個帖子標註 [帖子 ID: {{thread_id}}] {{標題}}。字數：{word_min}-{word_max}字。"
            )
    
    # 錯誤處理
    instructions.append(
        f"### 錯誤處理\n若無匹配帖子，回應：「{CONFIG['error_prompt_template'].format(selected_cat=selected_source, filters=json.dumps(filters))}」"
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
        thread_data = thread_data[:2]
        data = f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n篩選條件：{json.dumps(filters, ensure_ascii=False)}"
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
