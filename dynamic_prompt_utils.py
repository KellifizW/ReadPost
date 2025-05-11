import aiohttp
import asyncio
import json
import logging
import re
from logging_config import configure_logger
import streamlit as st
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict
from jinja2 import Template
from tenacity import retry, stop_after_attempt, wait_fixed

logger = configure_logger(__name__, "dynamic_prompt_utils.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"

# Centralized configuration
CONFIG = {
    "max_prompt_length": 150000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75,
}

# Pydantic models for INTENT_CONFIG
class ProcessingConfig(BaseModel):
    post_limit: int = 5
    data_type: str = "both"
    max_replies: int = 150
    sort: str = "confidence"
    min_replies: int = 10
    sort_override: Optional[Dict[str, str]] = None

class IntentConfig(BaseModel):
    triggers: Dict[str, any]
    word_range: Tuple[int, int]
    prompt_instruction: str
    prompt_format: str
    processing: ProcessingConfig = ProcessingConfig()

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types like regex

# Default intent configuration without triggers
DEFAULT_INTENT = IntentConfig(
    word_range=(700, 3000),
    prompt_instruction="Summarize up to 5 threads, focusing on key points and keywords.",
    prompt_format="Paragraphs summarizing discussions, citing [帖子 ID: {thread_id}].",
    processing=ProcessingConfig(),
    triggers={"keywords": [], "confidence": 0.5}  # Placeholder, will be overridden
)

# Intent configuration using Pydantic
INTENT_CONFIG = {
    "summarize_posts": IntentConfig(
        word_range=DEFAULT_INTENT.word_range,
        prompt_instruction=DEFAULT_INTENT.prompt_instruction,
        prompt_format=DEFAULT_INTENT.prompt_format,
        processing=DEFAULT_INTENT.processing,
        triggers={"keywords": ["總結", "摘要", "概覽"], "confidence": 0.7},
    ),
    "analyze_sentiment": IntentConfig(
        word_range=DEFAULT_INTENT.word_range,
        prompt_instruction="Analyze sentiment of up to 5 threads (positive/neutral/negative).",
        prompt_format="Paragraphs with sentiment summary, optional table: | ID | Sentiment | Ratio |.",
        processing=ProcessingConfig(max_replies=300),
        triggers={"keywords": ["情緒", "氣氛", "指數", "正負面", "sentiment", "mood"], "confidence": 0.90},
    ),
    "follow_up": IntentConfig(
        triggers={"keywords": ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩"], "confidence": 0.90},
        word_range=(1000, 4000),
        prompt_instruction="Deep dive into referenced threads, supplementing with context.",
        prompt_format="Paragraphs with cohesive follow-up, segmented by viewpoints.",
        processing=ProcessingConfig(post_limit=2, data_type="replies", max_replies=400, min_replies=5),
    ),
    "fetch_thread_by_id": IntentConfig(
        triggers={"regex": r"(?:帖子\s*ID\s*[:=]?\s*|ID\s*[:=]?\s*)(\w+)", "confidence": 0.95},
        word_range=(500, 2000),
        prompt_instruction="Summarize specified thread, highlighting core discussions.",
        prompt_format="Paragraphs summarizing thread, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(post_limit=1, data_type="replies", max_replies=400, min_replies=0),
    ),
    "general_query": IntentConfig(
        word_range=(500, 2000),
        prompt_instruction="Provide concise summary based on metadata, keeping it brief.",
        prompt_format="Paragraphs answering query, citing relevant threads if needed.",
        processing=ProcessingConfig(max_replies=100),
        triggers={"keywords": [], "confidence": 0.5},
    ),
    "list_titles": IntentConfig(
        triggers={"keywords": ["列出", "標題", "清單", "所有標題"], "confidence": 0.95},
        word_range=(500, 3000),
        prompt_instruction="List up to 15 thread titles, explaining relevance.",
        prompt_format="List with [帖子 ID: {thread_id}], title, and relevance note.",
        processing=ProcessingConfig(post_limit=15, data_type="metadata", max_replies=20, min_replies=5),
    ),
    "find_themed": IntentConfig(
        word_range=DEFAULT_INTENT.word_range,
        prompt_instruction="Find threads matching themes, highlighting thematic links.",
        prompt_format=DEFAULT_INTENT.prompt_format,
        processing=ProcessingConfig(post_limit=20),
        triggers={"keywords": ["主題", "類似", "相關"], "confidence": 0.85},
    ),
    "fetch_dates": IntentConfig(
        triggers={"keywords": ["日期", "時間", "最近更新"], "confidence": 0.85},
        word_range=(500, 2000),
        prompt_instruction="Extract dates for up to 5 threads, integrating into summary.",
        prompt_format="Table: | ID | Title | Date |, followed by brief summary.",
        processing=ProcessingConfig(data_type="metadata", max_replies=100, sort="new", min_replies=5),
    ),
    "search_keywords": IntentConfig(
        word_range=DEFAULT_INTENT.word_range,
        prompt_instruction="Search threads with matching keywords, emphasizing matches.",
        prompt_format=DEFAULT_INTENT.prompt_format,
        processing=ProcessingConfig(post_limit=20),
        triggers={"keywords": ["搜索", "查找", "關鍵詞"], "confidence": 0.85},
    ),
    "recommend_threads": IntentConfig(
        triggers={"keywords": ["推薦", "建議", "熱門"], "confidence": 0.85},
        word_range=(500, 3000),
        prompt_instruction="Recommend 2-5 threads based on replies/likes, justifying choices.",
        prompt_format="List with [帖子 ID: {thread_id}], title, and recommendation reason.",
        processing=ProcessingConfig(data_type="metadata", max_replies=100),
    ),
    "time_sensitive_analysis": IntentConfig(
        triggers={"keywords": ["今晚", "今日", "最近", "今個星期"], "confidence": 0.85},
        word_range=(500, 2000),
        prompt_instruction="Summarize threads from last 24 hours, focusing on recent discussions.",
        prompt_format="Paragraphs noting reply times, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(sort="new", min_replies=5),
    ),
    "contextual_analysis": IntentConfig(
        word_range=DEFAULT_INTENT.word_range,
        prompt_instruction="Summarize 5 threads, identifying themes and sentiment ratios.",
        prompt_format="Paragraphs by theme, ending with table: | Theme | Ratio | Thread |.",
        processing=DEFAULT_INTENT.processing,
        triggers={"keywords": ["Reddit", "LIHKG", "子版", "討論區"], "confidence": 0.85},
    ),
    "rank_topics": IntentConfig(
        triggers={"keywords": ["熱門", "最多", "關注", "流行"], "confidence": 0.85},
        word_range=(500, 2000),
        prompt_instruction="Rank up to 5 threads/topics by engagement, explaining reasons.",
        prompt_format="List with [帖子 ID: {thread_id}], title, and engagement reason.",
        processing=ProcessingConfig(data_type="metadata", max_replies=100),
    ),
    "search_odd_posts": IntentConfig(
        word_range=DEFAULT_INTENT.word_range,
        prompt_instruction="Summarize up to 5 threads with odd or unusual content, highlighting peculiar aspects.",
        prompt_format=DEFAULT_INTENT.prompt_format,
        processing=DEFAULT_INTENT.processing,
        triggers={"keywords": ["離奇", "怪事", "奇聞", "詭異", "不可思議", "怪談", "荒誕"], "confidence": 0.85},
    ),
    "hypothetical_advice": IntentConfig(
        triggers={
            "keywords": ["你點睇", "你認為", "你會點", "你會如何", "如果你是", "假設你是", "你的看法", "你建議"],
            "regex": r"(你|Grok)\s*(點睇|認為|會\s*\w+|如何|是\s*.*\s*會|看法|建議)",
            "confidence": 0.90
        },
        word_range=(500, 2500),
        prompt_instruction="Answer as Grok, providing insights or advice for the hypothetical scenario or query, incorporating platform discussions if relevant.",
        prompt_format="Paragraphs with Grok's perspective, optionally citing [帖子 ID: {thread_id}] for context.",
        processing=ProcessingConfig(max_replies=100, min_replies=5),
    ),
    "risk_warning": IntentConfig(
        word_range=DEFAULT_INTENT.word_range,
        prompt_instruction="Identify and analyze risk factors (e.g., policy changes, market volatility, negative news) in financial discussions, assessing their potential impact on the specified investment.",
        prompt_format="Table: | Risk Factor | Description | Potential Impact | Thread Reference |, followed by a summary paragraph.",
        processing=ProcessingConfig(sort="hot", sort_override={"reddit": "controversial"}),
        triggers={"keywords": ["投資", "股票", "基金", "債券", "樓市", "房地產", "加密貨幣", "比特幣", "風險", "市場"], "confidence": 0.85},
    ),
}

# Jinja2 template for prompt generation
PROMPT_TEMPLATE = Template("""
[System]
You are a data assistant for {{ source_name }} ({{ source_type }}), responding in Traditional Chinese with a clear, concise tone. Cite threads using [帖子 ID: {{ thread_id }}].
[Context]
Query: {{ query }}
Conversation History: {{ history | tojson }}
Platform: {{ source_name }}
[Data]
Metadata: {{ metadata | tojson }}
Thread Data: {{ thread_data | tojson }}
Filters: {{ filters | tojson }}
[Instructions]
Generate a response ({{ word_min }}-{{ word_max }} words): {{ instruction }} Use format: Intro + thematic paragraphs + {{ format }}. For {{ source_type }}, include platform context.
""")

def log_event(level: str, message: str, function_name: str, **kwargs):
    """Unified logging function."""
    log_func = getattr(logger, level, logger.info)
    context = f"[{function_name}] {message}" + (f" | {kwargs}" if kwargs else "")
    log_func(context)

async def parse_api_response(api_response: Optional[Dict], default_output: Dict, function_name: str) -> Dict:
    """Parse API response with error handling and JSON recovery."""
    if not api_response or not api_response.get("choices"):
        log_event("warning", "API call failed or no choices", function_name)
        return default_output
    try:
        return json.loads(api_response["choices"][0]["message"]["content"])
    except json.JSONDecodeError as e:
        log_event("error", f"JSON parse error: {str(e)}", function_name)
        content = api_response["choices"][0]["message"]["content"]
        if content.endswith("..."):
            try:
                fixed_content = content.rsplit(",", 1)[0] + "]}"
                return json.loads(fixed_content)
            except json.JSONDecodeError:
                log_event("warning", "Failed to fix truncated JSON", function_name)
        return default_output

@retry(stop=stop_after_attempt(CONFIG["max_parse_retries"]), wait=wait_fixed(2))
async def call_grok3_api(
    payload: Dict, timeout: int = CONFIG["parse_timeout"], function_name: str = "unknown"
) -> Optional[Dict]:
    """Unified Grok 3 API call handler with retry."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.secrets['grok3key']}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=timeout) as response:
            if response.status != 200 or not (data := await response.json()).get("choices"):
                log_event("warning", f"API call failed: status={response.status}, no choices", function_name)
                raise ValueError("API call failed")
            log_event(
                "info", "API call succeeded", function_name,
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0)
            )
            return data

async def extract_keywords(
    query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg"
) -> Dict:
    """Extract semantic keywords and related terms from query."""
    conversation_context = conversation_context or []
    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何", "係咩"]
    platform_terms = {
        "lihkg": ["吹水", "高登", "連登"],
        "reddit": ["YOLO", "DD", "subreddit"],
    }
    prompt = f"""
Extract {CONFIG['min_keywords']}-{CONFIG['max_keywords']} core keywords (prioritize nouns/verbs, exclude generic terms: {generic_terms}).
Generate up to 8 related terms (synonyms or platform terms: {platform_terms[source_type]}).
Set time_sensitive to true if query contains time-sensitive words (e.g., 今晚, 今日, 最近).
Query: {query}
Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}
Source: {source_type}
Output Format: {{"keywords": [], "related_terms": [], "reason": "logic (max 70 chars)", "time_sensitive": true/false}}
"""
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": f"Semantic analysis assistant for {source_type} keywords."},
            *conversation_context,
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }
    default_output = {
        "keywords": [],
        "related_terms": [],
        "reason": "API call failed",
        "time_sensitive": False,
    }
    api_response = await call_grok3_api(payload, function_name="extract_keywords")
    result = await parse_api_response(api_response, default_output, "extract_keywords")
    
    keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
    log_event("info", f"Extracted keywords: {keywords}", "extract_keywords")
    return {
        "keywords": keywords[:CONFIG["max_keywords"]],
        "related_terms": result.get("related_terms", [])[:8],
        "reason": result.get("reason", "No reason provided")[:70],
        "time_sensitive": result.get("time_sensitive", False),
    }

async def parse_query(
    query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg"
) -> Dict:
    """Parse user query to extract intents, keywords, and thread IDs."""
    conversation_context = conversation_context or []
    if not isinstance(query, str):
        log_event("error", f"Invalid query type: expected str, got {type(query)}", "parse_query")
        return {
            "intents": [{"intent": "summarize_posts", "confidence": 0.5, "reason": "Invalid query type"}],
            "keywords": [],
            "related_terms": [],
            "time_range": "all",
            "thread_ids": [],
            "reason": "Invalid query type",
            "confidence": 0.5,
        }

    # Extract keywords using the restored function
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key, source_type)
    keywords = keyword_result.get("keywords", [])
    related_terms = keyword_result.get("related_terms", [])
    time_sensitive = keyword_result.get("time_sensitive", False)

    prompt = f"""
Analyze the query to classify up to 4 intents, considering conversation history and keywords.
Query: {query}
Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}
Keywords: {json.dumps(keywords, ensure_ascii=False)}
Related Terms: {json.dumps(related_terms, ensure_ascii=False)}
Source: {source_type}
Supported Intents: {json.dumps(list(INTENT_CONFIG.keys()), ensure_ascii=False)}
Output Format: {{"intents": [{{"intent": "intent", "confidence": 0.0-1.0, "reason": "reason"}}, ...], "reason": "overall reason"}}
Filter intents with confidence >= {CONFIG["intent_confidence_threshold"]}.
"""
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "You are a semantic analysis assistant, responding in Traditional Chinese."},
            *conversation_context,
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.5,
    }
    default_output = {
        "intents": [{"intent": "summarize_posts", "confidence": 0.7, "reason": "API call failed"}],
        "reason": "API call failed"
    }
    result = await parse_api_response(
        await call_grok3_api(payload, function_name="parse_query"), default_output, "parse_query"
    )

    intents, thread_ids = [], []
    query_lower = query.lower()
    for intent, config in INTENT_CONFIG.items():
        triggers = config.triggers
        if "regex" in triggers and (match := re.search(triggers["regex"], query, re.IGNORECASE)):
            if intent == "fetch_thread_by_id":
                thread_ids.append(match.group(1))
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": f"Regex match: {match.group(0)}"})
        elif "keywords" in triggers and any(kw.lower() in query_lower for kw in triggers["keywords"]):
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": f"Keyword match: {triggers['keywords']}"})

    intents = intents or result["intents"]
    is_vague = len(keywords) < 2 and not any(
        kw in query_lower for kw in ["分析", "總結", "討論", "主題", "時事", "推薦", "熱門", "最多", "關注"]
    )
    has_multi_intent = any(ind in query_lower for ind in ["並且", "同時", "總結並", "列出並", "分析並"])
    if is_vague and not has_multi_intent:
        intents = [max(intents, key=lambda x: x["confidence"])]

    if not intents:
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "Default to summarization"}]

    result = {
        "intents": intents[:4],
        "keywords": keywords[:CONFIG["max_keywords"]],
        "related_terms": related_terms[:8],
        "time_range": "recent" if time_sensitive else "all",
        "thread_ids": thread_ids,
        "reason": result.get("reason", "Semantic matching"),
        "confidence": max(i["confidence"] for i in intents),
    }
    log_event(
        "info", "Parsed query result", "parse_query",
        intents=[i["intent"] for i in intents], keywords=keywords,
        thread_ids=thread_ids, time_range="recent" if time_sensitive else "all",
        result=result
    )
    return result

async def extract_relevant_thread(
    conversation_context: List[Dict], query: str, grok3_api_key: str
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Extract relevant thread ID from conversation history."""
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None, "No conversation history"

    # Use extract_keywords instead of parse_query to avoid redundant API calls
    query_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = query_result["keywords"] + query_result["related_terms"]

    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: ([a-zA-Z0-9]+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_result = await extract_keywords(title, conversation_context, grok3_api_key)
                title_keywords = title_result["keywords"] + title_result["related_terms"]
                common_keywords = set(query_keywords).intersection(set(title_keywords))
                content_contains_keywords = any(kw.lower() in message["content"].lower() for kw in query_keywords)
                if common_keywords or content_contains_keywords:
                    return thread_id, title, message["content"], f"Keyword match: {common_keywords or query_keywords}"

    prompt = f"""
Compare query and conversation history to find the most relevant thread ID.
Query: {query}
Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}
Keywords: {json.dumps(query_keywords, ensure_ascii=False)}
Output Format: {{"thread_id": "id", "title": "title", "reason": "reason"}}
Return empty object {{}} if no match.
"""
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "Semantic analysis assistant for thread matching."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    default_output = {"thread_id": None, "title": None, "reason": "No matching thread"}
    result = await parse_api_response(
        await call_grok3_api(payload, function_name="extract_relevant_thread"),
        default_output, "extract_relevant_thread"
    )
    return (
        result.get("thread_id"), result.get("title"), None,
        result.get("reason", "No reason provided") if result.get("thread_id") else "No matching thread"
    )

async def truncate_data(thread_data: List[Dict], max_length: int) -> List[Dict]:
    """Truncate thread data to fit prompt length."""
    if not thread_data:
        return []
    if sum(len(json.dumps(d, ensure_ascii=False)) for d in thread_data) <= max_length:
        return thread_data
    return [
        {
            "thread_id": d["thread_id"],
            "title": d["title"],
            "replies": d.get("replies", [])[:10],
            "total_fetched_replies": min(len(d.get("replies", [])), 10)
        }
        for d in thread_data[:3]
    ]

async def build_dynamic_prompt(
    query: str,
    conversation_context: List[Dict],
    metadata: List[Dict],
    thread_data: List[Dict],
    filters: Dict,
    intent: str,
    selected_source: Dict,
    grok3_api_key: str,
) -> str:
    """Build dynamic prompt using Jinja2 template."""
    # Handle None or invalid selected_source
    if not isinstance(selected_source, (str, dict)):
        log_event("warning", "Invalid selected_source, using default", "build_dynamic_prompt")
        selected_source = {"source_name": "Unknown", "source_type": "lihkg"}
    selected_source = (
        {"source_name": selected_source, "source_type": "reddit" if "reddit" in selected_source.lower() else "lihkg"}
        if isinstance(selected_source, str) else
        selected_source
    )
    source_name = selected_source.get("source_name", "Unknown")
    source_type = selected_source.get("source_type", "lihkg")

    # Handle None inputs
    conversation_context = conversation_context or []
    metadata = metadata or []
    thread_data = thread_data or []
    filters = filters or {}

    intent_config = INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"])
    word_min, word_max = intent_config.word_range
    prompt_length = len(query) + len(json.dumps(conversation_context, ensure_ascii=False)) + len(json.dumps(thread_data, ensure_ascii=False))
    length_factor = min(prompt_length / (CONFIG["max_prompt_length"] * 0.8), 1.0)
    word_min, word_max = int(word_min * (1 + length_factor * 0.7)), int(word_max * (1 + length_factor * 0.7))

    # Inline metadata extraction with validation
    validated_metadata = [
        {
            "thread_id": item["thread_id"],
            "title": item["title"],
            "no_of_reply": item.get("no_of_reply", 0),
            "like_count": item.get("like_count", 0),
            "last_reply_time": item.get("last_reply_time", 0),
        }
        for item in metadata
        if isinstance(item, dict) and "thread_id" in item and "title" in item
    ]

    thread_data = await truncate_data(thread_data, CONFIG["max_prompt_length"] - 2000)
    prompt = PROMPT_TEMPLATE.render(
        source_name=source_name,
        source_type=source_type,
        query=query,
        history=conversation_context,
        metadata=validated_metadata,
        thread_data=thread_data,
        filters=filters,
        instruction=intent_config.prompt_instruction,
        format=intent_config.prompt_format,
        word_min=word_min,
        word_max=word_max
    )
    log_event(
        "info", f"Built prompt: query={query[:50]}..., length={len(prompt)}", "build_dynamic_prompt",
        intent=intent, source_name=source_name, source_type=source_type
    )
    return prompt

def get_intent_processing_params(intent: str, source_type: str = "lihkg") -> Dict:
    """Retrieve processing parameters for a given intent."""
    params = INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"]).processing.dict()
    sort_override = params.get("sort_override", {})
    params["sort"] = sort_override.get(source_type.lower(), params["sort"])
    if source_type.lower() == "lihkg" and params["sort"] == "confidence":
        params["sort"] = "hot"
    return params