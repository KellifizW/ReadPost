# dynamic_prompt_utils.py
import aiohttp
import asyncio
import json
import logging
import re
from logging_config import configure_logger
import streamlit as st
from typing import Dict, List, Optional, Tuple

logger = configure_logger(__name__, "dynamic_prompt_utils.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"

# Default intent configuration
DEFAULT_INTENT_CONFIG = {
    "word_range": (500, 3000),
    "processing": {
        "post_limit": 5,
        "data_type": "both",
        "max_replies": 150,
        "sort": "hot",
        "min_replies": 15,
        "sort_override": {"reddit": "top"}
    }
}

# INTENT_CONFIG with enhanced triggers and focused instructions
INTENT_CONFIG = {
    "summarize_posts": {
        "triggers": {
            "keywords": ["總結", "摘要", "概覽", "整理"],
            "confidence": 0.7,
            "reason": "Summarization query"
        },
        "prompt_instruction": "Directly summarize up to 5 threads, focusing on key points and keywords relevant to the query, avoiding unnecessary greetings.",
        "prompt_format": "Paragraphs summarizing discussions, citing [帖子 ID: {thread_id}].",
        **DEFAULT_INTENT_CONFIG
    },
    "analyze_sentiment": {
        "triggers": {
            "keywords": ["情緒", "正負面", "氣氛", "指數", "mood"],
            "confidence": 0.9,
            "reason": "Sentiment analysis"
        },
        "prompt_instruction": "Analyze sentiment (positive/neutral/negative) of up to 5 threads, focusing on ratios and query relevance, without greetings.",
        "prompt_format": "Paragraphs with table: | ID | Sentiment | Ratio |",
        "processing": {"max_replies": 300, "min_replies": 15},
        **DEFAULT_INTENT_CONFIG
    },
    "follow_up": {
        "triggers": {
            "keywords": ["詳情", "更多", "為什麼", "點解", "進一步"],
            "confidence": 0.9,
            "reason": "Follow-up query"
        },
        "prompt_instruction": "Deep dive into referenced threads, supplementing with detailed context and viewpoints, directly addressing the query without pleasantries.",
        "prompt_format": "Paragraphs segmented by viewpoints, citing [帖子 ID: {thread_id}].",
        "word_range": (500, 4000),
        "processing": {
            "post_limit": 2,
            "data_type": "replies",
            "max_replies": 400,
            "sort_override": {"reddit": "new"}
        },
        **DEFAULT_INTENT_CONFIG
    },
    "fetch_thread_by_id": {
        "triggers": {
            "regex": r"(?:帖子\s*ID\s*[:=]?\s*|ID\s*[:=]?\s*)(\w+)",
            "confidence": 0.95,
            "reason": "Specific thread ID"
        },
        "prompt_instruction": "Summarize specified thread, focusing on core discussions and query relevance, avoiding greetings.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 1,
            "data_type": "replies",
            "max_replies": 400,
            "min_replies": 0
        },
        **DEFAULT_INTENT_CONFIG
    },
    "general_query": {
        "triggers": {
            "keywords": [],
            "confidence": 0.5,
            "reason": "Default query"
        },
        "prompt_instruction": "Provide a concise summary based on metadata and keywords, directly addressing the query without pleasantries.",
        "prompt_format": "Paragraphs citing relevant threads.",
        **DEFAULT_INTENT_CONFIG
    },
    "list_titles": {
        "triggers": {
            "keywords": ["列出", "標題", "清單", "所有標題"],
            "confidence": 0.95,
            "reason": "List titles query"
        },
        "prompt_instruction": "List up to 15 thread titles, explaining relevance to the query, without greetings.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, and relevance note.",
        "processing": {
            "post_limit": 15,
            "data_type": "metadata",
            "max_replies": 20,
            "min_replies": 5
        },
        **DEFAULT_INTENT_CONFIG
    },
    "find_themed": {
        "triggers": {
            "keywords": ["主題", "相關", "類似"],
            "confidence": 0.85,
            "reason": "Themed query"
        },
        "prompt_instruction": "Find threads matching themes, emphasizing thematic links and query relevance, avoiding pleasantries.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 20,
            "max_replies": 150,
            "min_replies": 15
        },
        **DEFAULT_INTENT_CONFIG
    },
    "fetch_dates": {
        "triggers": {
            "keywords": ["日期", "時間", "最近更新"],
            "confidence": 0.85,
            "reason": "Date-focused query"
        },
        "prompt_instruction": "Extract dates for up to 5 threads, integrating into a summary, directly addressing the query.",
        "prompt_format": "Table: | ID | Title | Date |, followed by brief summary.",
        "processing": {
            "post_limit": 5,
            "data_type": "metadata",
            "sort": "new",
            "min_replies": 5
        },
        **DEFAULT_INTENT_CONFIG
    },
    "search_keywords": {
        "triggers": {
            "keywords": ["搜索", "關鍵詞", "查找"],
            "confidence": 0.85,
            "reason": "Keyword search"
        },
        "prompt_instruction": "Search threads with matching keywords, emphasizing matches and query context, without greetings.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 20,
            "max_replies": 150,
            "min_replies": 15
        },
        **DEFAULT_INTENT_CONFIG
    },
    "recommend_threads": {
        "triggers": {
            "keywords": ["推薦", "熱門", "建議"],
            "confidence": 0.85,
            "reason": "Recommendation query"
        },
        "prompt_instruction": "Recommend 2-5 threads based on engagement, justifying choices, directly addressing the query.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, and recommendation reason.",
        **DEFAULT_INTENT_CONFIG
    },
    "time_sensitive_analysis": {
        "triggers": {
            "keywords": ["今晚", "今日", "最近", "今個星期"],
            "confidence": 0.85,
            "reason": "Time-sensitive query"
        },
        "prompt_instruction": "Summarize threads from last 24 hours, focusing on recent discussions and query relevance, without pleasantries.",
        "prompt_format": "Paragraphs noting reply times, citing [帖子 ID: {thread_id}].",
        "processing": {"sort": "new"},
        **DEFAULT_INTENT_CONFIG
    },
    "contextual_analysis": {
        "triggers": {
            "keywords": ["Reddit", "LIHKG", "子版", "討論區"],
            "confidence": 0.85,
            "reason": "Platform-specific query"
        },
        "prompt_instruction": "Summarize 5 threads, identifying themes and sentiment ratios, directly addressing the query.",
        "prompt_format": "Paragraphs by theme, with table: | Theme | Ratio | Thread |",
        "word_range": (500, 4000),
        **DEFAULT_INTENT_CONFIG
    },
    "rank_topics": {
        "triggers": {
            "keywords": ["熱門", "最多", "關注", "流行"],
            "confidence": 0.85,
            "reason": "Ranking query"
        },
        "prompt_instruction": "Rank up to 5 threads by engagement, explaining reasons, focusing on query relevance.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, and engagement reason.",
        **DEFAULT_INTENT_CONFIG
    },
    "search_odd_posts": {
        "triggers": {
            "keywords": ["離奇", "怪事", "奇聞", "詭異"],
            "confidence": 0.85,
            "reason": "Odd posts query"
        },
        "prompt_instruction": "Summarize up to 5 threads with odd or unusual content, highlighting peculiar aspects, without greetings.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        **DEFAULT_INTENT_CONFIG
    },
    "hypothetical_advice": {
        "triggers": {
            "keywords": ["你認為", "你建議", "你點睇", "你的看法"],
            "regex": r"(你|Grok)\s*(認為|建議|點睇|看法)",
            "confidence": 0.9,
            "reason": "Advice-seeking query"
        },
        "prompt_instruction": "Answer as Grok, providing insights or advice directly addressing the query, incorporating platform discussions if relevant, without pleasantries.",
        "prompt_format": "Paragraphs with Grok's perspective, optionally citing [帖子 ID: {thread_id}].",
        **DEFAULT_INTENT_CONFIG
    },
    "risk_warning": {
        "triggers": {
            "keywords": ["投資", "股票", "風險", "市場", "加密貨幣"],
            "confidence": 0.85,
            "reason": "Financial risk query"
        },
        "prompt_instruction": "Analyze risk factors (e.g., market volatility, policy changes) in financial discussions, assessing impact, directly addressing the query.",
        "prompt_format": "Table: | Risk Factor | Description | Impact | Thread |, with summary.",
        "word_range": (500, 4000),
        "processing": {"sort_override": {"reddit": "controversial"}},
        **DEFAULT_INTENT_CONFIG
    }
}

CONFIG = {
    "max_prompt_length": 300000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75
}

async def call_ai_api(api_type: str, api_key: str, messages: List[dict], base_url: str = None, model: str = None, max_tokens: int = 3000, temperature: float = 0.7, stream: bool = False) -> Optional[Dict]:
    """Unified API call handler for Grok and other APIs, supporting streaming and non-streaming responses with fallback."""
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    except KeyError:
        logger.error(f"API call failed: Missing API key for {api_type}")
        return None
    
    # Use different endpoints based on api_type
    endpoint = "/chat/completions" if api_type == "grok" else "/completions"
    base_url = base_url or GROK3_API_URL
    url = f"{base_url.rstrip('/')}{endpoint}"  # 修正 URL 拼接
    
    # Set model based on api_type with fallback for ChatAnywhere
    models_to_try = ["gpt-3.5-turbo", "deepseek-r1", "deepseek-v3"] if api_type == "chatanywhere" else ["grok-3"]
    model = model or models_to_try[0]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    
    logger.info(f"Calling API: type={api_type}, url={url}, model={model}, stream={stream}, payload_size={len(json.dumps(payload, ensure_ascii=False))} chars...")
    
    async with aiohttp.ClientSession() as session:
        for model in models_to_try:
            payload["model"] = model
            try:
                async with session.post(url, headers=headers, json=payload, timeout=CONFIG["parse_timeout"]) as response:
                    if response.status != 200:
                        error_message = await response.text()
                        logger.warning(f"API call failed: status={response.status}, message={error_message}, url={url}, model={model}")
                        # Fallback to non-streaming if streaming fails
                        if stream and response.status in (403, 429):
                            logger.info(f"Streaming failed, retrying with stream=False: status={response.status}, url={url}, model={model}")
                            payload["stream"] = False
                            async with session.post(url, headers=headers, json=payload, timeout=CONFIG["parse_timeout"]) as retry_response:
                                if retry_response.status != 200:
                                    retry_error = await retry_response.text()
                                    logger.warning(f"Non-streaming retry failed: status={retry_response.status}, message={retry_error}, url={url}, model={model}")
                                    continue
                                data = await retry_response.json()
                                logger.info(f"Non-streaming retry successful: status={retry_response.status}, response={json.dumps(data, ensure_ascii=False)[:200]}...")
                                return data
                        continue
                    
                    if stream:
                        if response.content_type != "text/event-stream":
                            logger.error(f"Unexpected content type: {response.content_type}, expected text/event-stream, url={url}, model={model}")
                            # Fallback to non-streaming
                            logger.info(f"Streaming content type invalid, retrying with stream=False: content_type={response.content_type}, url={url}, model={model}")
                            payload["stream"] = False
                            async with session.post(url, headers=headers, json=payload, timeout=CONFIG["parse_timeout"]) as retry_response:
                                if retry_response.status != 200:
                                    retry_error = await retry_response.text()
                                    logger.warning(f"Non-streaming retry failed: status={retry_response.status}, message={retry_error}, url={url}, model={model}")
                                    continue
                                data = await retry_response.json()
                                logger.info(f"Non-streaming retry successful: status={retry_response.status}, response={json.dumps(data, ensure_ascii=False)[:200]}...")
                                return data
                        async def stream_response():
                            try:
                                async for line in response.content:
                                    if line and not line.isspace():
                                        line_str = line.decode("utf-8").strip()
                                        if line_str == "data: [DONE]":
                                            break
                                        if line_str.startswith("data:"):
                                            try:
                                                chunk = json.loads(line_str[5:])
                                                yield chunk
                                            except json.JSONDecodeError as e:
                                                logger.warning(f"Failed to parse stream chunk: {line_str}, error={str(e)}")
                                                continue
                                        yield line_str
                            finally:
                                await response.release()
                        logger.info(f"Streaming API call successful: status={response.status}, url={url}, model={model}")
                        return stream_response()
                    else:
                        data = await response.json()
                        logger.info(f"Non-streaming API call successful: status={response.status}, response={json.dumps(data, ensure_ascii=False)[:200]}..., model={model}")
                        return data
            except aiohttp.ClientError as e:
                logger.error(f"API call error: {str(e)}, url={url}, model={model}, payload_size={len(json.dumps(payload, ensure_ascii=False))} chars...")
                # Fallback to non-streaming for network errors
                if stream:
                    logger.info(f"Streaming network error, retrying with stream=False: error={str(e)}, url={url}, model={model}")
                    payload["stream"] = False
                    try:
                        async with session.post(url, headers=headers, json=payload, timeout=CONFIG["parse_timeout"]) as retry_response:
                            if retry_response.status != 200:
                                retry_error = await retry_response.text()
                                logger.warning(f"Non-streaming retry failed: status={retry_response.status}, message={retry_error}, url={url}, model={model}")
                                continue
                            data = await retry_response.json()
                            logger.info(f"Non-streaming retry successful: status={retry_response.status}, response={json.dumps(data, ensure_ascii=False)[:200]}..., model={model}")
                            return data
                    except aiohttp.ClientError as retry_e:
                        logger.error(f"Non-streaming retry error: {str(retry_e)}, url={url}, model={model}")
                        continue
                continue
        logger.error(f"All model attempts failed for {api_type}: models={models_to_try}, url={url}")
        return None

async def call_grok3_api(payload: Dict, function_name: str = "unknown", retries: int = CONFIG["max_parse_retries"], timeout: int = CONFIG["parse_timeout"]) -> Optional[Dict]:
    """Unified Grok 3 API call handler with detailed logging and JSON fix."""
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error(f"API call failed in {function_name}: Missing Grok 3 API key")
        return None
    messages = payload.get("messages", [])
    return await call_ai_api(
        api_type="grok",
        api_key=api_key,
        messages=messages,
        base_url=GROK3_API_URL,
        model=payload.get("model", "grok-3"),
        max_tokens=payload.get("max_tokens", 3000),
        temperature=payload.get("temperature", 0.7),
        stream=payload.get("stream", False)
    )

async def determine_post_limit(query: str, keywords: List[str], intents: List[Dict], source_type: str, grok3_api_key: str) -> int:
    """Determine dynamic post limit (3-15) based on query specificity."""
    prompt = f"""
Analyze query for optimal post count (3-15).
- Broad queries (e.g., '市場趨勢') require more posts (10-15).
- Specific queries (e.g., '某帖子詳情') require fewer posts (3-5).
Query: {query}
Keywords: {keywords}
Intents: {intents}
Source: {source_type}
Output: {{"post_limit": number, "reason": "reason (max 70 chars)"}}
"""
    payload = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": "Post limit analyzer"}, {"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    if api_response := await call_grok3_api(payload, "determine_post_limit"):
        response_content = api_response["choices"][0]["message"]["content"]
        try:
            result = json.loads(response_content)
            return max(3, min(15, result.get("post_limit", 5)))
        except json.JSONDecodeError:
            if response_content.strip().endswith("...") or response_content.strip().endswith('"'):
                try:
                    fixed_content = response_content.rsplit(",", 1)[0] + "}"
                    result = json.loads(fixed_content)
                    return max(3, min(15, result.get("post_limit", 5)))
                except json.JSONDecodeError:
                    logger.warning("Failed to fix truncated JSON in determine_post_limit")
            keyword_count = len(keywords)
            return 3 if keyword_count <= 2 else 5 if keyword_count <= 4 else 10
    logger.info(f"Fallback post_limit: 5, reason=API call failed")
    return 5

async def parse_query(query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg") -> Dict:
    """Parse query to extract intents, keywords, thread IDs, post_limit, and context summary."""
    if not isinstance(query, str):
        logger.error(f"Invalid query type: {type(query)}")
        return {
            "intents": [{"intent": "summarize_posts", "confidence": 0.5, "reason": "Invalid query type"}],
            "keywords": [],
            "related_terms": [],
            "time_range": "all",
            "thread_ids": [],
            "reason": "Invalid query type",
            "confidence": 0.5,
            "post_limit": 5,
            "context_summary": ""
        }

    # Generate context summary from conversation history
    context_summary = ""
    if conversation_context:
        prompt = f"""
Summarize conversation history to identify user's query focus (e.g., topics, intents, preferences).
History: {json.dumps(conversation_context, ensure_ascii=False)}
Output: {{"summary": "brief summary (max 100 chars)", "reason": "summarization logic (max 70 chars)"}}
"""
        payload = {
            "model": "grok-3",
            "messages": [{"role": "system", "content": "Conversation summarizer"}, {"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.5
        }
        if api_response := await call_grok3_api(payload, "generate_context_summary"):
            response_content = api_response["choices"][0]["message"]["content"]
            try:
                result = json.loads(response_content)
                context_summary = result.get("summary", "")[:100]
                logger.info(f"Context summary generated: {context_summary}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse context summary: {response_content}")
                context_summary = "Unable to summarize conversation history"

    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key, source_type)
    keywords, time_sensitive = keyword_result.get("keywords", []), keyword_result.get("time_sensitive", False)
    intents, thread_ids, query_lower, reason = [], [], query.lower(), "Initial intent detection"

    for intent, config in INTENT_CONFIG.items():
        triggers = config.get("triggers", {})
        if "regex" in triggers and (match := re.search(triggers["regex"], query, re.IGNORECASE)):
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": triggers["reason"]})
            if intent == "fetch_thread_by_id":
                thread_ids.append(match.group(1))
        elif "keywords" in triggers and any(kw.lower() in query_lower for kw in triggers["keywords"]):
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": triggers["reason"]})
            if intent == "time_sensitive_analysis":
                time_sensitive = True

    is_vague = len(keywords) < 2 and not any(kw in query_lower for kw in ["分析", "總結", "討論", "主題"])
    if not intents or is_vague:
        prompt = f"""
Analyze query for intents, considering history and keywords.
Query: {query}
History: {json.dumps(conversation_context, ensure_ascii=False)}
Source: {source_type}
Output: {{"intents": [{{"intent": "intent", "confidence": 0.0-1.0, "reason": "reason"}}, ...], "reason": "overall reason"}}
"""
        payload = {
            "model": "grok-3",
            "messages": [{"role": "system", "content": "Semantic analysis assistant"}, *conversation_context, {"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.5
        }
        if api_response := await call_grok3_api(payload, "parse_query"):
            response_content = api_response["choices"][0]["message"]["content"]
            try:
                result = json.loads(response_content)
                intents = [i for i in result.get("intents", []) if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                reason = result.get("reason", "Semantic matching")
            except json.JSONDecodeError:
                if response_content.strip().endswith("..."):
                    try:
                        fixed_content = response_content.rsplit(",", 1)[0] + "]}"
                        result = json.loads(fixed_content)
                        intents = [i for i in result.get("intents", []) if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                        reason = result.get("reason", "Fixed truncated JSON")
                    except json.JSONDecodeError:
                        logger.warning("Failed to fix truncated JSON in parse_query")
                intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "JSON parsing failed"}]
        else:
            intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "API call failed"}]

    if not intents:
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "Default to summarization"}]

    post_limit = await determine_post_limit(query, keywords, intents, source_type, grok3_api_key)
    logger.info(f"Parsed query: intents={[i['intent'] for i in intents]}, keywords={keywords}, post_limit={post_limit}, context_summary={context_summary}")
    return {
        "intents": intents[:4],
        "keywords": keywords,
        "related_terms": keyword_result.get("related_terms", []),
        "time_range": "recent" if time_sensitive else "all",
        "thread_ids": thread_ids,
        "reason": reason,
        "confidence": max(i["confidence"] for i in intents),
        "post_limit": post_limit,
        "context_summary": context_summary
    }

async def extract_keywords(query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg") -> Dict:
    """Extract semantic keywords and related terms."""
    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何", "係咩"]
    platform_terms = {"lihkg": ["吹水", "連登", "高登"], "reddit": ["subreddit", "YOLO", "DD"]}
    prompt = f"""
Extract {CONFIG['min_keywords']}-{CONFIG['max_keywords']} keywords (exclude: {generic_terms}).
Generate up to 8 related terms (e.g., {platform_terms[source_type]}).
Check for time-sensitive words (e.g., 今日, 最近, 今晚).
Query: {query}
History: {json.dumps(conversation_context, ensure_ascii=False)}
Source: {source_type}
Output: {{"keywords": [], "related_terms": [], "reason": "logic (max 70 chars)", "time_sensitive": true/false}}
"""
    payload = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": f"Keyword analyzer for {source_type}"}, *conversation_context, {"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3
    }
    if api_response := await call_grok3_api(payload, "extract_keywords"):
        try:
            result = json.loads(api_response["choices"][0]["message"]["content"])
            keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
            return {
                "keywords": keywords[:CONFIG["max_keywords"]],
                "related_terms": result.get("related_terms", [])[:8],
                "reason": result.get("reason", "Extracted keywords")[:70],
                "time_sensitive": result.get("time_sensitive", False)
            }
        except json.JSONDecodeError:
            logger.error(f"JSON decode error in extract_keywords: response={api_response['choices'][0]['message']['content']}")
    return {"keywords": [], "related_terms": [], "reason": "API call failed", "time_sensitive": False}

async def extract_relevant_thread(conversation_context: List[Dict], query: str, grok3_api_key: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Extract relevant thread ID from history."""
    if len(conversation_context) < 2:
        return None, None, None, "No conversation history"

    query_keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = query_keyword_result["keywords"] + query_keyword_result["related_terms"]

    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: ([a-zA-Z0-9]+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keyword_result = await extract_keywords(title, conversation_context, grok3_api_key)
                title_keywords = title_keyword_result["keywords"] + title_keyword_result["related_terms"]
                common_keywords = set(query_keywords).intersection(title_keywords)
                if common_keywords or any(kw.lower() in message["content"].lower() for kw in query_keywords):
                    return thread_id, title, message["content"], f"Keyword match: {common_keywords or query_keywords}"

    prompt = f"""
Find relevant thread ID from history.
Query: {query}
History: {json.dumps(conversation_context, ensure_ascii=False)}
Keywords: {query_keywords}
Output: {{"thread_id": "id", "title": "title", "reason": "reason"}}
"""
    payload = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": "Thread matching assistant"}, {"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    if api_response := await call_grok3_api(payload, "extract_relevant_thread"):
        try:
            result = json.loads(api_response["choices"][0]["message"]["content"])
            thread_id = result.get("thread_id")
            return thread_id, result.get("title"), None, result.get("reason", "No reason") if thread_id else "No match"
        except json.JSONDecodeError:
            logger.error(f"JSON decode error in extract_relevant_thread: response={api_response['choices'][0]['message']['content']}")
    return None, None, None, "API call failed"

def extract_thread_metadata(metadata: List[Dict]) -> List[Dict]:
    """Extract key fields from thread metadata."""
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
        logger.error(f"Error extracting metadata: {str(e)}")
        return []

async def build_dynamic_prompt(query: str, conversation_context: List[Dict], metadata: List[Dict], thread_data: List[Dict], filters: Dict, intent: str, selected_source: Dict, grok3_api_key: str) -> str:
    """Build dynamic prompt based on intent and data, incorporating context summary."""
    if isinstance(selected_source, str):
        source_name = selected_source
        source_type = "reddit" if "reddit" in source_name.lower() else "lihkg"
        selected_source = {"source_name": source_name, "source_type": source_type}
    elif not isinstance(selected_source, dict):
        logger.warning(f"Invalid selected_source type: {type(selected_source)}, using default")
        selected_source = {"source_name": "Unknown", "source_type": "lihkg"}

    source_name = selected_source.get("source_name", "Unknown")
    source_type = selected_source.get("source_type", "lihkg")

    system = (
        f"You are a data assistant for {source_name} ({source_type}), responding in Traditional Chinese. "
        f"Provide direct, focused answers to the query, avoiding greetings or pleasantries. "
        f"Cite threads using [帖子 ID: {{thread_id}}] format. Choose format (e.g., paragraphs, lists, tables) based on intent."
    )
    context_summary = filters.get("context_summary", "") or ""
    context = (
        f"Query: {query}\n"
        f"Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}\n"
        f"Context Summary: {context_summary}\n"
        f"Platform: {source_name}"
    )
    metadata = extract_thread_metadata(metadata)
    data = (
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"Thread Data: {json.dumps(thread_data, ensure_ascii=False)}\n"
        f"Filters: {json.dumps(filters, ensure_ascii=False)}"
    )

    intent_config = INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"])
    word_min, word_max = intent_config["word_range"]
    prompt_length = len(context) + len(data) + len(system) + 500
    length_factor = min(prompt_length / (CONFIG["max_prompt_length"] * 0.8), 1.0)
    word_min = int(word_min + (word_max - word_min) * length_factor * 0.8)
    word_max = int(word_min + (word_max - word_min) * (1 + length_factor * 0.8))

    instruction_parts = [intent_config["prompt_instruction"]]
    format_instructions = [intent_config["prompt_format"]]

    platform_instruction = (
        f"For {source_type}, incorporate platform-specific context (e.g., Reddit subreddits, LIHKG trends)."
    )
    combined_instruction = (
        f"Generate a cohesive response ({word_min}-{word_max} words): {'; '.join(instruction_parts)} "
        f"Use format: Intro (1 sentence) + thematic paragraphs + {'; '.join(format_instructions)} "
        f"Focus on query, ensure fluency, cite [帖子 ID: {{thread_id}}] {{title}}. {platform_instruction}"
    )

    prompt = f"[System]\n{system}\n[Context]\n{context}\n[Data]\n{data}\n[Instructions]\n{combined_instruction}"
    if len(prompt) > CONFIG["max_prompt_length"]:
        logger.warning(f"Prompt exceeds {CONFIG['max_prompt_length']}, truncating thread data")
        thread_data = [
            {
                **data,
                "replies": data.get("replies", [])[:10],
                "total_fetched_replies": min(len(data.get("replies", [])), 10)
            }
            for data in thread_data[:3]
        ]
        data = (
            f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Thread Data: {json.dumps(thread_data, ensure_ascii=False)}\n"
            f"Filters: {json.dumps(filters, ensure_ascii=False)}"
        )
        prompt = f"[System]\n{system}\n[Context]\n{context}\n[Data]\n{data}\n[Instructions]\n{combined_instruction}"
        logger.info(f"Truncated prompt: length={len(prompt)}, threads={len(thread_data)}")

    logger.info(f"Built prompt: query={query}, length={len(prompt)}, intent={intent}, context_summary={context_summary}")
    return prompt
