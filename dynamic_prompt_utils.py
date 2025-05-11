import aiohttp
import asyncio
import json
import logging
import re
from logging_config import configure_logger
import streamlit as st
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from jinja2 import Template
from tenacity import retry, stop_after_attempt, wait_fixed

logger = configure_logger(__name__, "dynamic_prompt_utils.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"

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
    triggers: Dict
    word_range: Tuple[int, int]
    prompt_instruction: str
    prompt_format: str
    processing: ProcessingConfig = ProcessingConfig()

DEFAULT_INTENT = IntentConfig(
    triggers={"keywords": [], "confidence": 0.5, "reason": "Default general query"},
    word_range=(500, 2000),
    prompt_instruction="Provide concise summary based on metadata, keeping it brief.",
    prompt_format="Paragraphs answering query, citing relevant threads if needed.",
    processing=ProcessingConfig(),
)

INTENT_CONFIG = {
    "summarize_posts": IntentConfig(
        triggers={"keywords": ["總結", "摘要", "概覽"], "confidence": 0.7, "reason": "Detected summarization query"},
        word_range=(700, 3000),
        prompt_instruction="Summarize up to 5 threads, focusing on key points and keywords.",
        prompt_format="Paragraphs summarizing discussions, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(),
    ),
    "analyze_sentiment": IntentConfig(
        triggers={"keywords": ["情緒", "氣氛", "指數", "正負面", "sentiment", "mood"], "confidence": 0.90, "reason": "Detected sentiment analysis query"},
        word_range=(700, 3000),
        prompt_instruction="Analyze sentiment of up to 5 threads (positive/neutral/negative).",
        prompt_format="Paragraphs with sentiment summary, optional table: | ID | Sentiment | Ratio |.",
        processing=ProcessingConfig(max_replies=300),
    ),
    "follow_up": IntentConfig(
        triggers={"keywords": ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩"], "confidence": 0.90, "reason": "Detected follow-up query"},
        word_range=(1000, 4000),
        prompt_instruction="Deep dive into referenced threads, supplementing with context.",
        prompt_format="Paragraphs with cohesive follow-up, segmented by viewpoints.",
        processing=ProcessingConfig(post_limit=2, data_type="replies", max_replies=400, min_replies=5),
    ),
    "fetch_thread_by_id": IntentConfig(
        triggers={"regex": r"(?:帖子\s*ID\s*[:=]?\s*|ID\s*[:=]?\s*)(\w+)", "confidence": 0.95, "reason": "Detected specific thread ID"},
        word_range=(500, 2000),
        prompt_instruction="Summarize specified thread, highlighting core discussions.",
        prompt_format="Paragraphs summarizing thread, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(post_limit=1, data_type="replies", max_replies=400, min_replies=0),
    ),
    "general_query": DEFAULT_INTENT,
    "list_titles": IntentConfig(
        triggers={"keywords": ["列出", "標題", "清單", "所有標題"], "confidence": 0.95, "reason": "Detected list titles query"},
        word_range=(500, 3000),
        prompt_instruction="List up to 15 thread titles, explaining relevance.",
        prompt_format="List with [帖子 ID: {thread_id}], title, and relevance note.",
        processing=ProcessingConfig(post_limit=15, data_type="metadata", max_replies=20, min_replies=5),
    ),
    "find_themed": IntentConfig(
        triggers={"keywords": ["主題", "類似", "相關"], "confidence": 0.85, "reason": "Detected themed query"},
        word_range=(700, 3000),
        prompt_instruction="Find threads matching themes, highlighting thematic links.",
        prompt_format="Paragraphs emphasizing themes, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(post_limit=20),
    ),
    "fetch_dates": IntentConfig(
        triggers={"keywords": ["日期", "時間", "最近更新"], "confidence": 0.85, "reason": "Detected date-focused query"},
        word_range=(500, 2000),
        prompt_instruction="Extract dates for up to 5 threads, integrating into summary.",
        prompt_format="Table: | ID | Title | Date |, followed by brief summary.",
        processing=ProcessingConfig(data_type="metadata", sort="new", min_replies=5),
    ),
    "search_keywords": IntentConfig(
        triggers={"keywords": ["搜索", "查找", "關鍵詞"], "confidence": 0.85, "reason": "Detected keyword search query"},
        word_range=(700, 3000),
        prompt_instruction="Search threads with matching keywords, emphasizing matches.",
        prompt_format="Paragraphs highlighting keyword matches, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(post_limit=20),
    ),
    "recommend_threads": IntentConfig(
        triggers={"keywords": ["推薦", "建議", "熱門"], "confidence": 0.85, "reason": "Detected recommendation query"},
        word_range=(500, 3000),
        prompt_instruction="Recommend 2-5 threads based on replies/likes, justifying choices.",
        prompt_format="List with [帖子 ID: {thread_id}], title, and recommendation reason.",
        processing=ProcessingConfig(data_type="metadata"),
    ),
    "time_sensitive_analysis": IntentConfig(
        triggers={"keywords": ["今晚", "今日", "最近", "今個星期"], "confidence": 0.85, "reason": "Detected time-sensitive query"},
        word_range=(500, 2000),
        prompt_instruction="Summarize threads from last 24 hours, focusing on recent discussions.",
        prompt_format="Paragraphs noting reply times, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(sort="new", min_replies=5),
    ),
    "contextual_analysis": IntentConfig(
        triggers={"keywords": ["Reddit", "LIHKG", "子版", "討論區"], "confidence": 0.85, "reason": "Detected platform-specific query"},
        word_range=(700, 3000),
        prompt_instruction="Summarize 5 threads, identifying themes and sentiment ratios.",
        prompt_format="Paragraphs by theme, ending with table: | Theme | Ratio | Thread |.",
        processing=ProcessingConfig(),
    ),
    "rank_topics": IntentConfig(
        triggers={"keywords": ["熱門", "最多", "關注", "流行"], "confidence": 0.85, "reason": "Detected ranking query"},
        word_range=(500, 2000),
        prompt_instruction="Rank up to 5 threads/topics by engagement, explaining reasons.",
        prompt_format="List with [帖子 ID: {thread_id}], title, and engagement reason.",
        processing=ProcessingConfig(data_type="metadata"),
    ),
    "search_odd_posts": IntentConfig(
        triggers={"keywords": ["離奇", "怪事", "奇聞", "詭異", "不可思議", "怪談", "荒誕"], "confidence": 0.85, "reason": "Detected query for odd or unusual posts"},
        word_range=(700, 3000),
        prompt_instruction="Summarize up to 5 threads with odd or unusual content, highlighting peculiar aspects.",
        prompt_format="Paragraphs summarizing odd discussions, citing [帖子 ID: {thread_id}].",
        processing=ProcessingConfig(),
    ),
    "hypothetical_advice": IntentConfig(
        triggers={"keywords": ["你點睇", "你認為", "你會點", "你會如何", "如果你是", "假設你是", "你的看法", "你建議"], "regex": r"(你|Grok)\s*(點睇|認為|會\s*\w+|如何|是\s*.*\s*會|看法|建議)", "confidence": 0.90, "reason": "Detected hypothetical or advice-seeking query"},
        word_range=(500, 2500),
        prompt_instruction="Answer as Grok, providing insights or advice for the hypothetical scenario or query, incorporating platform discussions if relevant.",
        prompt_format="Paragraphs with Grok's perspective, optionally citing [帖子 ID: {thread_id}] for context.",
        processing=ProcessingConfig(min_replies=5),
    ),
    "risk_warning": IntentConfig(
        triggers={"keywords": ["投資", "股票", "基金", "債券", "樓市", "房地產", "加密貨幣", "比特幣", "風險", "市場"], "confidence": 0.85, "reason": "Detected financial query requiring risk analysis"},
        word_range=(700, 3000),
        prompt_instruction="Identify and analyze risk factors in financial discussions, assessing their potential impact on the specified investment.",
        prompt_format="Table: | Risk Factor | Description | Potential Impact | Thread Reference |, followed by a summary paragraph.",
        processing=ProcessingConfig(sort="hot", sort_override={"reddit": "controversial"}),
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
    log_func = getattr(logger, level, logger.info)
    context = f"[{function_name}] {message}" + (f" | {kwargs}" if kwargs else "")
    log_func(context)

async def parse_api_response(api_response: Optional[Dict], default_output: Dict, function_name: str) -> Dict:
    log_event("debug", f"Parsing API response: {api_response if api_response else 'None'}", function_name)
    if api_response is None:
        log_event("warning", "API response is None", function_name)
        return default_output
    if not isinstance(api_response, dict):
        log_event("warning", f"Invalid API response type: {type(api_response)}", function_name)
        return default_output
    if not api_response.get("choices") or not isinstance(api_response["choices"], list) or not api_response["choices"]:
        log_event("warning", f"No valid choices in API response: {api_response}", function_name)
        return default_output
    try:
        content = api_response["choices"][0]["message"]["content"]
        log_event("debug", f"API response content: {content[:500]}...", function_name)
        parsed = json.loads(content)
        if not parsed.get("keywords") or not parsed.get("intents"):
            log_event("warning", "Parsed JSON missing required fields", function_name)
            return default_output
        return parsed
    except (KeyError, json.JSONDecodeError) as e:
        log_event("error", f"JSON parse error: {str(e)}, response content: {content[:500] if 'content' in locals() else 'None'}", function_name)
        if isinstance(content, str) and (content.endswith("...") or "}" not in content):
            try:
                fixed_content = content.rstrip("...") + "}"
                parsed = json.loads(fixed_content)
                if not parsed.get("keywords") or not parsed.get("intents"):
                    log_event("warning", "Fixed JSON missing required fields", function_name)
                    return default_output
                log_event("info", "Successfully fixed truncated JSON", function_name)
                return parsed
            except json.JSONDecodeError:
                log_event("warning", "Failed to fix truncated JSON", function_name)
        return default_output

@retry(stop=stop_after_attempt(CONFIG["max_parse_retries"]), wait=wait_fixed(2))
async def call_grok3_api(payload: Dict, timeout: int = CONFIG["parse_timeout"], function_name: str = "unknown") -> Optional[Dict]:
    log_event("debug", f"Calling API with payload: {json.dumps(payload)[:500]}...", function_name)
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.secrets['grok3key']}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    log_event("warning", f"API call failed: status={response.status}, response={await response.text()[:500]}", function_name)
                    raise ValueError(f"API call failed with status {response.status}")
                data = await response.json()
                if not data or not isinstance(data, dict):
                    log_event("warning", f"Invalid API response: {data}", function_name)
                    raise ValueError("Invalid API response")
                if not data.get("choices") or not isinstance(data["choices"], list) or not data["choices"]:
                    log_event("warning", f"No choices in API response: {data}", function_name)
                    raise ValueError("No choices in API response")
                log_event("info", "API call succeeded", function_name, prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0), completion_tokens=data.get("usage", {}).get("completion_tokens", 0))
                return data
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
        log_event("error", f"API call error: {str(e)}", function_name)
        raise
    except Exception as e:
        log_event("error", f"Unexpected API call error: {str(e)}", function_name)
        raise

async def parse_query(query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg") -> Dict:
    log_event("debug", f"Parsing query: query={query}, source_type={source_type}, conversation_context_length={len(conversation_context)}", "parse_query")
    conversation_context = conversation_context or []
    default_output = {
        "keywords": [],
        "related_terms": [],
        "time_sensitive": False,
        "intents": [{"intent": "summarize_posts", "confidence": 0.7, "reason": "Default fallback"}],
        "reason": "Default output",
        "thread_ids": [],
        "confidence": 0.7
    }
    
    if not isinstance(query, str):
        log_event("warning", f"Invalid query type: {type(query)}", "parse_query")
        return {**default_output, "reason": "Invalid query type", "intents": [{"intent": "summarize_posts", "confidence": 0.5, "reason": "Invalid query type"}]}

    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何", "係咩"]
    platform_terms = {"lihkg": ["吹水", "高登", "連登"], "reddit": ["YOLO", "DD", "subreddit"]}
    prompt = f"""
Extract {CONFIG['min_keywords']}-{CONFIG['max_keywords']} keywords (exclude: {generic_terms}).
Generate up to 8 related terms (synonyms or platform terms: {platform_terms[source_type]}).
Set time_sensitive to true if query contains time-sensitive words (e.g., 今晚, 今日, 最近).
Classify up to 4 intents from {json.dumps(list(INTENT_CONFIG.keys()), ensure_ascii=False)}.
Query: {query}
Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}
Source: {source_type}
Output Format: {{"keywords": [], "related_terms": [], "time_sensitive": true/false, "intents": [{{"intent": "intent", "confidence": 0.0-1.0, "reason": "reason"}}, ...], "reason": "overall reason"}}
"""
    payload = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": "Semantic analysis assistant."}, *conversation_context, {"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.5,
    }

    try:
        api_response = await call_grok3_api(payload, function_name="parse_query")
        if api_response is None:
            log_event("error", "API response is None", "parse_query")
            return default_output
        result = await parse_api_response(api_response, default_output, "parse_query")
        if not isinstance(result, dict) or not result.get("intents"):
            log_event("warning", f"Invalid parse_api_response result: {result}", "parse_query")
            return default_output
    except Exception as e:
        log_event("error", f"Failed to parse query: {str(e)}", "parse_query")
        return default_output

    intents, thread_ids = [], []
    query_lower = query.lower()
    for intent, config in INTENT_CONFIG.items():
        triggers = config.triggers
        if "regex" in triggers and (match := re.search(triggers["regex"], query, re.IGNORECASE)):
            if intent == "fetch_thread_by_id":
                thread_ids.append(match.group(1))
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": triggers["reason"]})
        elif "keywords" in triggers and any(kw.lower() in query_lower for kw in triggers["keywords"]):
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": triggers["reason"]})

    is_vague = len(result.get("keywords", [])) < 2 and not any(kw in query_lower for kw in ["分析", "總結", "討論", "主題", "時事", "推薦", "熱門", "最多", "關注"])
    intents = intents or result.get("intents", default_output["intents"])
    if is_vague and not any(ind in query_lower for ind in ["並且", "同時", "總結並", "列出並", "分析並"]):
        intents = [max(intents, key=lambda x: x["confidence"])]

    output = {
        "intents": intents[:4],
        "keywords": [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms][:CONFIG["max_keywords"]],
        "related_terms": result.get("related_terms", [])[:8],
        "time_range": "recent" if result.get("time_sensitive", False) else "all",
        "thread_ids": thread_ids,
        "reason": result.get("reason", "Semantic matching"),
        "confidence": max(i["confidence"] for i in intents) if intents else 0.7
    }

    log_event("info", f"Parsed query: intents={[i['intent'] for i in intents]}, keywords={output['keywords']}, thread_ids={thread_ids}", "parse_query")
    return output

async def extract_relevant_thread(conversation_context: List[Dict], query: str, grok3_api_key: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    log_event("debug", f"Extracting relevant thread: query={query}, conversation_context_length={len(conversation_context)}", "extract_relevant_thread")
    if not conversation_context or len(conversation_context) < 2:
        log_event("info", "No conversation history available", "extract_relevant_thread")
        return None, None, None, "No conversation history"

    query_result = await parse_query(query, conversation_context, grok3_api_key)
    if not query_result:
        log_event("warning", "Failed to parse query result", "extract_relevant_thread")
        return None, None, None, "Failed to parse query"

    query_keywords = query_result["keywords"] + query_result["related_terms"]
    log_event("debug", f"Query keywords: {query_keywords}", "extract_relevant_thread")

    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: ([a-zA-Z0-9]+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_result = await parse_query(title, conversation_context, grok3_api_key)
                if not title_result:
                    log_event("warning", f"Failed to parse title: {title}", "extract_relevant_thread")
                    continue
                title_keywords = title_result["keywords"] + title_result["related_terms"]
                common_keywords = set(query_keywords).intersection(set(title_keywords))
                content_contains_keywords = any(kw.lower() in message["content"].lower() for kw in query_keywords)
                if common_keywords or content_contains_keywords:
                    log_event("info", f"Found relevant thread: thread_id={thread_id}, title={title}", "extract_relevant_thread")
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
        "messages": [{"role": "system", "content": "Semantic analysis assistant."}, {"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    default_output = {}
    result = await parse_api_response(await call_grok3_api(payload, function_name="extract_relevant_thread"), default_output, "extract_relevant_thread")
    thread_id = result.get("thread_id")
    log_event("info", f"Extract relevant thread result: thread_id={thread_id}, reason={result.get('reason', 'No matching thread')}", "extract_relevant_thread")
    return thread_id, result.get("title"), None, result.get("reason", "No matching thread") if thread_id else "No matching thread"

async def truncate_data(thread_data: List[Dict], max_length: int) -> List[Dict]:
    log_event("debug", f"Truncating thread data: input_length={len(thread_data)}, max_length={max_length}", "truncate_data")
    if not thread_data:
        log_event("warning", "Empty thread data provided", "truncate_data")
        return []
    if sum(len(json.dumps(d, ensure_ascii=False)) for d in thread_data) <= max_length:
        return thread_data
    truncated = [{"thread_id": d["thread_id"], "title": d["title"], "replies": d.get("replies", [])[:10], "total_fetched_replies": min(len(d.get("replies", [])), 10)} for d in thread_data[:3]]
    log_event("info", f"Truncated thread data: output_length={len(truncated)}", "truncate_data")
    return truncated

async def build_dynamic_prompt(query: str, conversation_context: List[Dict], metadata: List[Dict], thread_data: List[Dict], filters: Dict, intent: str, selected_source: Dict, grok3_api_key: str) -> str:
    log_event("debug", f"Building dynamic prompt: query={query[:50]}..., intent={intent}, metadata_length={len(metadata if metadata else [])}, thread_data_length={len(thread_data if thread_data else [])}", "build_dynamic_prompt")
    
    if not query or not isinstance(query, str):
        log_event("warning", f"Invalid query: {query}", "build_dynamic_prompt")
        return "Invalid query provided."
    if metadata is None:
        log_event("warning", "Metadata is None, using empty list", "build_dynamic_prompt")
        metadata = []
    if thread_data is None:
        log_event("warning", "Thread data is None, using empty list", "build_dynamic_prompt")
        thread_data = []
    if filters is None:
        log_event("warning", "Filters is None, using empty dict", "build_dynamic_prompt")
        filters = {}

    selected_source = {"source_name": selected_source, "source_type": "reddit" if "reddit" in str(selected_source).lower() else "lihkg"} if isinstance(selected_source, str) else selected_source or {"source_name": "Unknown", "source_type": "lihkg"}
    source_name, source_type = selected_source.get("source_name", "Unknown"), selected_source.get("source_type", "lihkg")
    log_event("debug", f"Source: name={source_name}, type={source_type}", "build_dynamic_prompt")

    intent_config = INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"])
    word_min, word_max = intent_config.word_range
    prompt_length = len(query) + len(json.dumps(conversation_context, ensure_ascii=False)) + len(json.dumps(thread_data, ensure_ascii=False))
    length_factor = min(prompt_length / (CONFIG["max_prompt_length"] * 0.8), 1.0)
    word_min, word_max = int(word_min * (1 + length_factor * 0.7)), int(word_max * (1 + length_factor * 0.7))
    log_event("debug", f"Prompt parameters: word_min={word_min}, word_max={word_max}, prompt_length={prompt_length}", "build_dynamic_prompt")

    # Inline metadata extraction with validation
    validated_metadata = []
    for item in metadata:
        if not isinstance(item, dict) or "thread_id" not in item or "title" not in item:
            log_event("warning", f"Invalid metadata item: {item}", "build_dynamic_prompt")
            continue
        validated_metadata.append({
            "thread_id": item["thread_id"],
            "title": item["title"],
            "no_of_reply": item.get("no_of_reply", 0),
            "like_count": item.get("like_count", 0),
            "last_reply_time": item.get("last_reply_time", 0)
        })
    log_event("debug", f"Validated metadata: length={len(validated_metadata)}", "build_dynamic_prompt")

    validated_thread_data = await truncate_data(thread_data, CONFIG["max_prompt_length"] - 2000)
    log_event("debug", f"Validated thread data: length={len(validated_thread_data)}", "build_dynamic_prompt")

    prompt = PROMPT_TEMPLATE.render(
        source_name=source_name,
        source_type=source_type,
        query=query,
        history=conversation_context,
        metadata=validated_metadata,
        thread_data=validated_thread_data,
        filters=filters,
        instruction=intent_config.prompt_instruction,
        format=intent_config.prompt_format,
        word_min=word_min,
        word_max=word_max
    )
    log_event("info", f"Built prompt: query={query[:50]}..., length={len(prompt)} chars, intent={intent}", "build_dynamic_prompt")
    return prompt

def get_intent_processing_params(intent: str, source_type: str = "lihkg") -> Dict:
    log_event("debug", f"Getting intent processing params: intent={intent}, source_type={source_type}", "get_intent_processing_params")
    params = INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"]).processing.dict()
    params["sort"] = params.get("sort_override", {}).get(source_type.lower(), params["sort"])
    params["sort"] = "hot" if source_type.lower() == "lihkg" and params["sort"] == "confidence" else params["sort"]
    log_event("debug", f"Intent processing params: {params}", "get_intent_processing_params")
    return params
