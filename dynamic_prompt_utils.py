import json
import logging
import re
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from logging_config import configure_logger

# 配置日誌
logger = configure_logger(__name__, "dynamic_prompt_utils.log")

# 配置參數
CONFIG = {
    "max_prompt_length": 120000,
    "parse_timeout": 90,
    "default_intents": ["recommend_threads", "summarize_posts", "time_sensitive_analysis", "rank_topics"],
    "error_prompt_template": "在 {selected_cat} 中未找到符合條件的帖子（篩選：{filters}）。建議嘗試其他關鍵詞或討論區！",
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75,
    "default_word_ranges": {
        "summarize_posts": (700, 1500),
        "analyze_sentiment": (700, 1500),
        "follow_up": (1000, 2500),
        "fetch_thread_by_id": (500, 1000),
        "general_query": (500, 1000),
        "list_titles": (500, 1500),
        "find_themed": (700, 1500),
        "fetch_dates": (500, 1000),
        "search_keywords": (700, 1500),
        "recommend_threads": (500, 1500),
        "time_sensitive_analysis": (500, 1000),
        "contextual_analysis": (700, 1500),
        "rank_topics": (500, 1000)
    }
}

# Pydantic 模型
class Intent(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

class QueryAnalysis(BaseModel):
    intents: List[Intent] = Field(default_factory=lambda: [Intent(intent="summarize_posts", confidence=0.5, reason="默認意圖")])
    keywords: List[str] = Field(default_factory=list)
    related_terms: List[str] = Field(default_factory=list)
    time_range: str = "all"
    thread_ids: List[str] = Field(default_factory=list)
    reason: str = "未提供原因"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

class KeywordResult(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    related_terms: List[str] = Field(default_factory=list)
    reason: str = "未提供原因"
    time_sensitive: bool = False

class ThreadMatch(BaseModel):
    thread_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    reason: str = "未提供原因"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, json.JSONDecodeError)),
    before_sleep=lambda retry_state: logger.debug(f"重試 {retry_state.attempt_number} 次")
)
async def parse_query(query, conversation_context, grok3_api_key, source_type="lihkg"):
    conversation_context = conversation_context or []
    
    if not isinstance(query, str):
        logger.error(f"無效查詢類型：預期 str，得到 {type(query)}")
        return QueryAnalysis(
            intents=[Intent(intent="summarize_posts", confidence=0.5, reason="無效查詢類型")],
            reason="無效查詢類型"
        )
    
    # 提取關鍵詞和相關詞
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key, source_type)
    keywords = keyword_result.keywords
    related_terms = keyword_result.related_terms
    time_sensitive = keyword_result.time_sensitive
    
    # 檢查特定帖子 ID
    id_match = re.search(r'(?:ID|帖子)\s*([a-zA-Z0-9]+)', query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        logger.info(f"檢測到特定帖子 ID 查詢：thread_id={thread_id}")
        return QueryAnalysis(
            intents=[Intent(intent="fetch_thread_by_id", confidence=0.95, reason=f"檢測到明確帖子 ID {thread_id}")],
            keywords=keywords,
            related_terms=related_terms,
            thread_ids=[thread_id],
            reason=f"檢測到明確帖子 ID {thread_id}",
            confidence=0.95
        )
    
    # 檢查追問
    thread_match = await extract_relevant_thread(conversation_context, query, grok3_api_key)
    if thread_match.thread_id:
        logger.info(f"檢測到追問：thread_id={thread_match.thread_id}, 原因={thread_match.reason}")
        return QueryAnalysis(
            intents=[Intent(intent="follow_up", confidence=0.90, reason=f"檢測到追問，匹配帖子 ID {thread_match.thread_id}")],
            keywords=keywords,
            related_terms=related_terms,
            thread_ids=[thread_match.thread_id],
            reason=f"檢測到追問，匹配帖子 ID {thread_match.thread_id}，原因：{thread_match.reason}",
            confidence=0.90
        )
    
    # 檢查情緒分析
    sentiment_triggers = ["情緒", "氣氛", "指數", "正負面", "sentiment", "mood"]
    if any(word in query for word in sentiment_triggers):
        logger.info(f"檢測到情緒分析查詢：query={query}")
        return QueryAnalysis(
            intents=[
                Intent(intent="analyze_sentiment", confidence=0.90, reason="檢測到情緒分析關鍵詞"),
                Intent(intent="summarize_posts", confidence=0.80, reason="情緒分析查詢，輔以總結")
            ],
            keywords=keywords,
            related_terms=related_terms,
            time_range="recent" if time_sensitive else "all",
            reason="檢測到情緒分析查詢，優先分析情緒比例",
            confidence=0.90
        )
    
    # 檢查時間敏感查詢
    time_triggers = ["今晚", "今日", "最近", "今個星期"]
    is_time_sensitive = time_sensitive or any(word in query for word in time_triggers)
    if is_time_sensitive:
        logger.info(f"檢測到時間敏感查詢：query={query}")
        return QueryAnalysis(
            intents=[
                Intent(intent="time_sensitive_analysis", confidence=0.85, reason="時間敏感查詢，優先最新帖子"),
                Intent(intent="summarize_posts", confidence=0.80, reason="時間敏感查詢，輔以總結")
            ],
            keywords=keywords,
            related_terms=related_terms,
            time_range="recent",
            reason="檢測到時間敏感查詢，優先最新帖子",
            confidence=0.85
        )
    
    # 檢查標題列舉查詢
    trigger_words = ["列出", "標題", "清單", "所有標題"]
    detected_triggers = [word for word in trigger_words if word in query]
    if detected_triggers:
        logger.info(f"檢測到 list_titles 觸發詞：{detected_triggers}")
        return QueryAnalysis(
            intents=[Intent(intent="list_titles", confidence=0.95, reason=f"檢測到 list_titles 觸發詞：{detected_triggers}")],
            keywords=keywords,
            related_terms=related_terms,
            reason=f"檢測到 list_titles 觸發詞：{detected_triggers}",
            confidence=0.95
        )
    
    # 檢查排序查詢
    ranking_triggers = ["熱門", "最多", "關注", "流行"]
    if any(word in query for word in ranking_triggers):
        logger.info(f"檢測到排序查詢：query={query}")
        return QueryAnalysis(
            intents=[
                Intent(intent="rank_topics", confidence=0.85, reason="檢測到排序關鍵詞"),
                Intent(intent="summarize_posts", confidence=0.80, reason="排序查詢，輔以總結")
            ],
            keywords=keywords,
            related_terms=related_terms,
            time_range="recent" if time_sensitive else "all",
            reason="檢測到排序查詢，優先排序話題",
            confidence=0.85
        )
    
    # 檢查平台特定查詢
    platform_triggers = ["Reddit", "LIHKG", "子版", "討論區"]
    if any(word in query for word in platform_triggers):
        logger.info(f"檢測到平台特定查詢：query={query}")
        return QueryAnalysis(
            intents=[
                Intent(intent="contextual_analysis", confidence=0.85, reason="平台特定查詢，需語義分析"),
                Intent(intent="summarize_posts", confidence=0.80, reason="平台特定查詢，輔以總結")
            ],
            keywords=keywords,
            related_terms=related_terms,
            reason="檢測到平台特定查詢，優先語義分析",
            confidence=0.85
        )
    
    # 語義意圖分析
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦", "熱門", "最多", "關注"])
    has_multi_intent = any(indicator in query for indicator in ["並且", "同時", "總結並", "列出並", "分析並"])
    
    prompt = f"""
你是一個語義分析助手，請分析查詢並分類最多4個意圖，考慮對話歷史、關鍵詞和相關詞。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
關鍵詞：{json.dumps(keywords, ensure_ascii=False)}
相關詞：{json.dumps(related_terms, ensure_ascii=False)}
數據來源：{source_type}
是否模糊查詢：{is_vague}
是否包含多意圖指示詞：{has_multi_intent}
特別注意：
- 支持粵語及口語化表達（如「既」「係咩」）。
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
    "time_sensitive_analysis": "處理時間敏感查詢，優先最新帖子",
    "contextual_analysis": "綜合總結帖子觀點、推薦相關帖子、分析情緒",
    "rank_topics": "按關注度排序話題或帖子"
}, ensure_ascii=False, indent=2)}
輸出格式：{{
  "intents": [
    {{"intent": "意圖1", "confidence": 0.0-1.0, "reason": "匹配原因"}},
    ...
  ],
  "reason": "整體匹配原因"
}}
僅返回信心值高於 {CONFIG['intent_confidence_threshold']} 的意圖。
"""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {grok3_api_key}"}
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
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=CONFIG["parse_timeout"]
        ) as response:
            if response.status != 200:
                raise aiohttp.ClientError(f"狀態碼={response.status}")
            data = await response.json()
            result = json.loads(data["choices"][0]["message"]["content"])
            intents = [Intent(**i) for i in result.get("intents", [])]
            intents = [i for i in intents if i.confidence >= CONFIG["intent_confidence_threshold"]]
            if not intents:
                intents = [Intent(intent="summarize_posts", confidence=0.7, reason="無高信心意圖，默認總結")]
            if is_vague and not has_multi_intent:
                intents = [max(intents, key=lambda x: x.confidence)]
            return QueryAnalysis(
                intents=intents[:4],
                keywords=keywords,
                related_terms=related_terms,
                time_range="recent" if is_time_sensitive else "all",
                reason=result.get("reason", "語義匹配"),
                confidence=max(i.confidence for i in intents)
            )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, json.JSONDecodeError)),
    before_sleep=lambda retry_state: logger.debug(f"重試 {retry_state.attempt_number} 次")
)
async def extract_keywords(query, conversation_context, grok3_api_key, source_type="lihkg"):
    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何", "係咩"]
    platform_terms = {
        "lihkg": ["吹水", "高登", "連登"],
        "reddit": ["YOLO", "DD", "subreddit"]
    }
    
    prompt = f"""
請從查詢提取 {CONFIG['min_keywords']}-{CONFIG['max_keywords']} 個核心關鍵詞（優先名詞或核心動詞，排除通用詞如 {generic_terms}）。
生成最多8個語義相關詞（同義詞或討論區術語，如 {platform_terms[source_type]}）。
若查詢包含時間性詞語（如「今晚」「今日」「最近」），設置 time_sensitive 為 true。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
數據來源：{source_type}
返回格式：
{{
  "keywords": ["關鍵詞1", "關鍵詞2", ...],
  "related_terms": ["相關詞1", "相關詞2", ...],
  "reason": "提取邏輯（70字以內）",
  "time_sensitive": true/false
}}
"""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {grok3_api_key}"}
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": f"你是語義分析助手，以繁體中文回答，專注於提取 {source_type} 討論區關鍵詞。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.3
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=CONFIG["parse_timeout"]
        ) as response:
            if response.status != 200:
                raise aiohttp.ClientError(f"狀態碼={response.status}")
            data = await response.json()
            result = json.loads(data["choices"][0]["message"]["content"])
            keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
            return KeywordResult(
                keywords=keywords[:CONFIG["max_keywords"]],
                related_terms=result.get("related_terms", [])[:8],
                reason=result.get("reason", "未提供原因")[:70],
                time_sensitive=result.get("time_sensitive", False)
            )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, json.JSONDecodeError)),
    before_sleep=lambda retry_state: logger.debug(f"重試 {retry_state.attempt_number} 次")
)
async def extract_relevant_thread(conversation_context, query, grok3_api_key):
    if not conversation_context or len(conversation_context) < 2:
        return ThreadMatch(reason="無對話歷史")
    
    query_keywords = (await extract_keywords(query, conversation_context, grok3_api_key)).keywords
    follow_up_phrases = ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩"]
    is_follow_up_query = any(phrase in query for phrase in follow_up_phrases)
    
    # 關鍵詞匹配
    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: ([a-zA-Z0-9]+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keywords = (await extract_keywords(title, conversation_context, grok3_api_key)).keywords
                common_keywords = set(query_keywords).intersection(set(title_keywords))
                content_contains_keywords = any(kw.lower() in message["content"].lower() for kw in query_keywords)
                if common_keywords or content_contains_keywords or is_follow_up_query:
                    return ThreadMatch(
                        thread_id=thread_id,
                        title=title,
                        content=message["content"],
                        reason=f"關鍵詞匹配：{common_keywords or query_keywords}, 追問詞：{is_follow_up_query}"
                    )
    
    # 語義相似度分析
    prompt = f"""
比較查詢和對話歷史，找出最相關的帖子 ID，基於語義相似度。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
關鍵詞：{json.dumps(query_keywords, ensure_ascii=False)}
輸出格式：{{
  "thread_id": "帖子 ID",
  "title": "標題",
  "reason": "匹配原因"
}}
若無匹配，返回 {{}}
"""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {grok3_api_key}"}
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是語義分析助手，專注於匹配 Reddit 和 LIHKG 帖子。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=CONFIG["parse_timeout"]
        ) as response:
            if response.status != 200:
                raise aiohttp.ClientError(f"狀態碼={response.status}")
            data = await response.json()
            result = json.loads(data["choices"][0]["message"]["content"])
            return ThreadMatch(**result) if result.get("thread_id") else ThreadMatch(reason="無匹配帖子")

async def build_dynamic_prompt(query, conversation_context, metadata, thread_data, filters, intent, selected_source, grok3_api_key):
    if isinstance(selected_source, str):
        source_name = selected_source
        source_type = "reddit" if "reddit" in source_name.lower() else "lihkg"
        selected_source = {"source_name": source_name, "source_type": source_type}
    elif not isinstance(selected_source, dict):
        logger.warning(f"無效的 selected_source 類型：{type(selected_source)}")
        selected_source = {"source_name": "未知", "source_type": "lihkg"}
    
    parsed_query = await parse_query(query, conversation_context, grok3_api_key, selected_source.get("source_type", "lihkg"))
    intents = parsed_query.intents
    keywords = parsed_query.keywords
    related_terms = parsed_query.related_terms
    time_range = parsed_query.time_range
    
    system = (
        "你是社交媒體討論區（包括 LIHKG 和 Reddit）的數據助手，以繁體中文回答，"
        "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
        "禁止使用 [post_id: ...] 格式。根據用戶意圖動態選擇回應格式（例如列表、段落、表格等），"
        "確保結構清晰、內容連貫，且適合查詢的需求。"
    )
    
    context = (
        f"用戶問題：{query}\n"
        f"討論區：{selected_source.get('source_name', '未知')}\n"
        f"對話歷史：{json.dumps([{'role': msg['role'], 'content': msg['content'][:500]} for msg in conversation_context], ensure_ascii=False)}"
    )
    
    data = (
        f"帖子元數據：{json.dumps([{'thread_id': item['thread_id'], 'title': item['title'], 'no_of_reply': item.get('no_of_reply', 0), 'like_count': item.get('like_count', 0), 'last_reply_time': item.get('last_reply_time', 0)} for item in metadata], ensure_ascii=False)}\n"
        f"帖子內容：{json.dumps(thread_data, ensure_ascii=False)}\n"
        f"篩選條件：{json.dumps(filters, ensure_ascii=False)}"
    )
    
    word_min = max(min_w for i in intents for intnt, (min_w, max_w) in CONFIG["default_word_ranges"].items() if intnt == i.intent)
    word_max = min(max_w for i in intents for intnt, (min_w, max_w) in CONFIG["default_word_ranges"].items() if intnt == i.intent)
    
    prompt_length = len(context) + len(data) + len(system) + 500
    length_factor = min(prompt_length / CONFIG["max_prompt_length"], 1.0)
    context_factor = len(conversation_context) / 10
    keyword_factor = len(keywords) / CONFIG["max_keywords"]
    word_min = int(word_min + (word_max - word_min) * (length_factor * 0.3 + context_factor * 0.2 + keyword_factor * 0.1))
    word_max = int(word_min + (word_max - word_min) * (1 + length_factor * 0.5 + context_factor * 0.1 + keyword_factor * 0.1))
    
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦", "熱門", "最多", "關注"])
    
    instructions = []
    formats = []
    source_type = selected_source.get("source_type", "lihkg")
    
    if len(intents) > 1:
        instructions.append(
            f"綜合{len(intents)}個意圖，生成連貫回應，聚焦關鍵詞 {keywords} 和相關詞 {related_terms}，"
            f"按主題組織討論，引用高點讚或最新回覆。"
        )
        formats.append(
            "使用連貫段落，按主題分段，融合多個意圖，必要時使用表格或列表。"
        )
    
    for intent_info in intents[:4]:
        if intent_info.intent == "summarize_posts":
            instructions.append(f"總結最多5個帖子的討論，聚焦關鍵詞 {keywords}。")
            formats.append("融入段落，總結核心討論，引用 [帖子 ID: {thread_id}]。")
        elif intent_info.intent == "analyze_sentiment":
            instructions.append("分析最多5個帖子的情緒，提供比例，融入總結。")
            formats.append("融入段落，必要時列出 | 帖子 ID | 情緒 | 比例 |。")
        elif intent_info.intent == "follow_up":
            instructions.append(f"深入分析歷史帖子，聚焦 {keywords}，補充上下文。")
            formats.append("融入段落，分段突出回覆觀點。")
        elif intent_info.intent == "fetch_thread_by_id":
            instructions.append("根據帖子 ID 總結內容，突出核心討論。")
            formats.append("融入段落，詳細總結，引用 [帖子 ID: {thread_id}]。")
        elif intent_info.intent == "general_query" and not is_vague:
            instructions.append("提供簡化總結，基於元數據推測話題。")
            formats.append("融入段落，直接回答，必要時引用帖子。")
        elif intent_info.intent == "list_titles":
            instructions.append(f"列出最多15個標題，聚焦 {keywords}，簡述相關性。")
            formats.append("融入列表，包含 [帖子 ID: {thread_id}]、標題和相關性。")
        elif intent_info.intent == "find_themed":
            instructions.append(f"尋找與 {keywords} 和 {related_terms} 相關的帖子。")
            formats.append("融入段落，強調主題，引用 [帖子 ID: {thread_id}]。")
        elif intent_info.intent == "fetch_dates":
            instructions.append(f"提取最多5個帖子的日期，聚焦 {keywords}。")
            formats.append("融入表格，列出 | 帖子 ID | 標題 | 日期 |，附總結。")
        elif intent_info.intent == "search_keywords":
            instructions.append(f"搜索包含 {keywords} 和 {related_terms} 的帖子。")
            formats.append("融入段落，突出匹配，引用 [帖子 ID: {thread_id}]。")
        elif intent_info.intent == "recommend_threads":
            instructions.append(f"推薦2-5個熱門帖子，基於回覆和點讚，聚焦 {keywords}。")
            formats.append("融入列表，包含 [帖子 ID: {thread_id}]、標題和理由。")
        elif intent_info.intent == "time_sensitive_analysis":
            instructions.append(f"總結24小時內帖子，聚焦 {keywords}，突出敏感討論。")
            formats.append("融入段落，標註時間，引用 [帖子 ID: {thread_id}]。")
        elif intent_info.intent == "contextual_analysis":
            instructions.append("綜合5個帖子，總結觀點，分析情緒比例。")
            formats.append("融入段落，結尾附 | 主題 | 比例 | 代表性帖子 |。")
        elif intent_info.intent == "rank_topics":
            instructions.append("按關注度排序5個帖子，簡述熱度原因。")
            formats.append("融入列表，包含 [帖子 ID: {thread_id}]、標題和熱度原因。")
    
    combined_instruction = (
        f"生成連貫回應（{word_min}-{word_max}字）：{'；'.join(instructions)}"
        f"回應格式：簡介+按主題分段+{'；'.join(formats)}"
        f"聚焦用戶問題，融合意圖，引用 [帖子 ID: {{thread_id}}] {{標題}}。"
        f"針對 {source_type} 平台，融入平台上下文。"
    )
    
    if time_range == "recent":
        combined_instruction += "僅考慮24小時內的帖子和回覆。"
    
    prompt = (
        f"[System]\n{system}\n"
        f"[Context]\n{context}\n"
        f"[Data]\n{data}\n"
        f"[Instructions]\n{combined_instruction}"
    )
    
    if len(prompt) > CONFIG["max_prompt_length"]:
        logger.warning("提示長度超過限制，縮減數據")
        thread_data = thread_data[:2]
        data = f"帖子元數據：{json.dumps([{'thread_id': item['thread_id'], 'title': item['title'], 'no_of_reply': item.get('no_of_reply', 0), 'like_count': item.get('like_count', 0), 'last_reply_time': item.get('last_reply_time', 0)} for item in metadata], ensure_ascii=False)}\n篩選條件：{json.dumps(filters, ensure_ascii=False)}"
        prompt = (
            f"[System]\n{system}\n"
            f"[Context]\n{context}\n"
            f"[Data]\n{data}\n"
            f"[Instructions]\n{combined_instruction}"
        )
    
    logger.info(f"生成提示：查詢={query}, 提示長度={len(prompt)} 字符, intents={[i.intent for i in intents]}")
    return prompt
