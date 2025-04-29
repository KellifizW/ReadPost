import aiohttp
import asyncio
import json
import re
import datetime
import time
import logging
import streamlit as st
import os
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from logging_config import configure_logger

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "grok_processing.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90

class PromptBuilder:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "prompts.json")
        
        logger.info(f"Attempting to load prompts.json from: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"prompts.json not found at: {config_path}")
            raise FileNotFoundError(f"prompts.json not found at: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.loads(f.read())
            logger.info("Loaded prompts.json successfully")
        except Exception as e:
            logger.error(f"Failed to load prompts.json: {str(e)}")
            raise

    def build_analyze(self, query, cat_name, cat_id, conversation_context=None):
        config = self.config["analyze"]
        context = config["context"].format(
            query=query, cat_name=cat_name, cat_id=cat_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        return f"{config['system']}\n{context}\n{config['instructions']}"

    def build_prioritize(self, query, cat_name, cat_id, threads):
        config = self.config.get("prioritize")
        if not config:
            logger.error("Prompt configuration for 'prioritize' not found")
            raise ValueError("Prompt configuration for 'prioritize' not found")
        context = config["context"].format(query=query, cat_name=cat_name, cat_id=cat_id)
        data = config["data"].format(threads=json.dumps(threads, ensure_ascii=False))
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def build_response(self, intent, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        config = self.config["response"].get(intent, self.config["response"]["general"])
        context = config["context"].format(
            query=query, selected_cat=selected_cat,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        data = config["data"].format(
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or [], ensure_ascii=False),
            filters=json.dumps(filters, ensure_ascii=False)
        )
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def get_system_prompt(self, mode):
        return self.config["system"].get(mode, "")

def clean_html(text):
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return "[表情符號]" if "hkgmoji" in text else "[圖片]" if any(ext in text.lower() for ext in ['.webp', '.jpg', '.png']) else "[無內容]"
        return text
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}")
        return text

def clean_response(response):
    if isinstance(response, str):
        cleaned = re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)
        if cleaned != response:
            logger.info("Cleaned response: removed post_id strings")
        return cleaned
    return response

async def extract_keywords_with_grok(query, conversation_context=None):
    conversation_context = conversation_context or []
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"keywords": [], "reason": "Missing API key"}

    prompt = f"""
你是一個專業的語義分析助手，專注於從用戶查詢中提取關鍵詞，以繁體中文回答。請分析以下查詢，提取 1-3 個最相關的核心關鍵詞（僅保留名詞或核心動詞，過濾掉無意義的停用詞如「的」「是」「個」等）。關鍵詞應反映查詢的主題或意圖，適合用於 LIHKG 論壇的帖子搜索或匹配。特別注意處理粵語俚語（如「講D咩」「點解」），將其視為無意義詞語並過濾。請以 JSON 格式返回結果，並提供提取邏輯的簡要解釋（70字以內）。

查詢："{query}"
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}

返回格式：
{{
  "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
  "reason": "提取邏輯說明（70字以內）"
}}
"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": "你是 LIHKG 論壇的語義分析助手，以繁體中文回答，專注於理解用戶意圖並提取關鍵詞。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Keyword extraction failed: status={response.status}, attempt={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Keyword extraction failed: missing choices, attempt={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    keywords = result.get("keywords", [])[:3]
                    reason = result.get("reason", "No reason provided")[:70]
                    logger.info(f"Keywords extracted: {keywords}, reason: {reason}")
                    return {"keywords": keywords, "reason": reason}
        except Exception as e:
            logger.warning(f"Keyword extraction error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            logger.error(f"Keyword extraction failed after {max_retries} attempts")
            return {"keywords": [], "reason": f"Extraction failed: {str(e)}"[:70]}

async def summarize_context(conversation_context):
    if not conversation_context:
        return {"theme": "general", "keywords": []}
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"theme": "general", "keywords": []}
    
    prompt = f"""
    你是對話摘要助手，請分析以下對話歷史，提煉主要主題和關鍵詞（最多3個）。
    特別注意用戶問題中的意圖（例如「熱門」「總結」「追問」）和回應中的帖子標題。
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    輸出格式：{{"theme": "主要主題", "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"]}}
    """
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    logger.warning(f"Context summarization failed: status={response.status}")
                    return {"theme": "general", "keywords": []}
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                logger.info(f"Context summarized: theme={result['theme']}, keywords={result['keywords']}")
                return result
    except Exception as e:
        logger.warning(f"Context summarization error: {str(e)}")
        return {"theme": "general", "keywords": []}

async def extract_relevant_thread(conversation_context, query):
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None
    
    query_keyword_result = await extract_keywords_with_grok(query, conversation_context)
    query_keywords = set(query_keyword_result["keywords"])
    
    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: (\d+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keyword_result = await extract_keywords_with_grok(title, conversation_context)
                title_keywords = set(title_keyword_result["keywords"])
                common_keywords = query_keywords.intersection(title_keywords)
                if common_keywords or any(kw.lower() in title.lower() for kw in query_keywords):
                    logger.info(f"Follow-up matched: thread_id={thread_id}, title={title}, keywords={common_keywords}")
                    return thread_id, title, message["content"]
    logger.info("No relevant thread matched for follow-up query")
    return None, None, None

async def analyze_and_screen(user_query, cat_name, cat_id, conversation_context=None):
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
    id_match = re.search(r'(?:ID|帖子)\s*(\d+)', user_query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        logger.info(f"Detected explicit thread ID: {thread_id}")
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": "fetch_thread_by_id",
            "theme": "特定帖子查詢",
            "category_ids": [cat_id],
            "data_type": "replies",
            "post_limit": 1,
            "reply_limit": 200,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": []},
            "processing": {"intent": "fetch_thread_by_id", "top_thread_ids": [thread_id]},
            "candidate_thread_ids": [thread_id],
            "top_thread_ids": [thread_id],
            "needs_advanced_analysis": False,
            "reason": f"Detected explicit thread ID {thread_id} in query",
            "theme_keywords": []
        }
    
    keyword_result = await extract_keywords_with_grok(user_query, conversation_context)
    query_keywords = keyword_result["keywords"]
    logger.info(f"Extracted keywords with Grok: {query_keywords}, reason: {keyword_result['reason']}")
    
    thread_id, thread_title, last_response = await extract_relevant_thread(conversation_context, user_query)
    if thread_id:
        logger.info(f"Follow-up intent confirmed: thread_id={thread_id}, title={thread_title}")
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": "follow_up",
            "theme": thread_title or "追問相關主題",
            "category_ids": [cat_id],
            "data_type": "replies",
            "post_limit": 1,
            "reply_limit": 200,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": query_keywords},
            "processing": {"intent": "follow_up", "top_thread_ids": [thread_id]},
            "candidate_thread_ids": [thread_id],
            "top_thread_ids": [thread_id],
            "needs_advanced_analysis": False,
            "reason": f"Matched follow-up query to thread_id={thread_id}, title={thread_title}, keywords: {query_keywords}",
            "theme_keywords": query_keywords
        }
    
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "general")
    historical_keywords = context_summary.get("keywords", [])
    
    is_vague = len(query_keywords) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"])
    
    is_follow_up = False
    referenced_thread_ids = []
    referenced_titles = []
    if conversation_context and len(conversation_context) >= 2:
        last_user_query = conversation_context[-2].get("content", "")
        last_response = conversation_context[-1].get("content", "")
        
        matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        referenced_thread_ids = matches
        
        last_query_keywords = (await extract_keywords_with_grok(last_user_query, conversation_context))["keywords"]
        common_words = set(query_keywords).intersection(set(last_query_keywords))
        explicit_follow_up = any(keyword in user_query for keyword in ["詳情", "更多", "進一步", "點解", "為什麼", "原因"])
        
        if len(common_words) >= 1 or explicit_follow_up:
            is_follow_up = True
            logger.info(f"Follow-up intent detected, referenced thread IDs: {referenced_thread_ids}, common_words: {common_words}")
    
    if is_follow_up and not thread_id:
        intent = "search_keywords"
        reason = "檢測到追問意圖，但無歷史帖子 ID匹配，回退到關鍵詞搜索"
        theme = query_keywords[0] if query_keywords else historical_theme
        theme_keywords = query_keywords or historical_keywords
        logger.info(f"Follow-up intent fallback to search_keywords, extracted keywords: {theme_keywords}")
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": intent,
            "theme": theme,
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 2,
            "reply_limit": 200,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": theme_keywords},
            "processing": {"intent": intent, "top_thread_ids": referenced_thread_ids[:2]},
            "candidate_thread_ids": [],
            "top_thread_ids": referenced_thread_ids[:2],
            "needs_advanced_analysis": False,
            "reason": reason,
            "theme_keywords": theme_keywords
        }
    
    semantic_prompt = f"""
    你是語義分析助手，請比較用戶問題與以下意圖描述，選擇最匹配的意圖。
    若問題模糊，優先延續對話歷史的意圖（歷史主題：{historical_theme}）。
    若問題涉及對之前回應的追問（包含「詳情」「更多」「進一步」「點解」「為什麼」「原因」等詞或與前問題/回應的帖子標題有重疊），選擇「follow_up」意圖。
    用戶問題：{user_query}
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    歷史主題：{historical_theme}
    歷史關鍵詞：{json.dumps(historical_keywords, ensure_ascii=False)}
    意圖描述：
    {json.dumps({
        "list_titles": "列出帖子標題或清單",
        "summarize_posts": "總結帖子內容或討論",
        "analyze_sentiment": "分析帖子或回覆的情緒",
        "general_query": "與LIHKG無關或模糊的問題",
        "find_themed": "尋找特定主題的帖子",
        "fetch_dates": "提取帖子或回覆的日期資料",
        "search_keywords": "根據關鍵詞搜索帖子",
        "recommend_threads": "推薦相關或熱門帖子",
        "follow_up": "追問之前回應中提到的帖子內容",
        "fetch_thread_by_id": "根據明確的帖子 ID 抓取內容"
    }, ensure_ascii=False, indent=2)}
    輸出格式：{{"intent": "最匹配的意圖", "confidence": 0.0-1.0, "reason": "匹配原因"}}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": historical_theme,
            "category_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "reply_limit": 0,
            "filters": {},
            "processing": {"intent": "general"},
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "Missing API key",
            "theme_keywords": historical_keywords
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": prompt_builder.get_system_prompt("analyze")},
        *conversation_context,
        {"role": "user", "content": semantic_prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    logger.info(f"Starting semantic intent analysis for query: {user_query}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Semantic intent analysis failed: status={response.status}, attempt={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Semantic intent analysis failed: missing choices, attempt={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intent = result.get("intent", "summarize_posts")
                    confidence = result.get("confidence", 0.7)
                    reason = result.get("reason", "語義匹配")
                    
                    if is_vague and historical_theme != "general":
                        intent = "summarize_posts"
                        reason = f"問題模糊，延續歷史主題：{historical_theme}"
                    elif is_vague:
                        intent = "summarize_posts"
                        reason = "問題模糊，默認總結帖子"
                    
                    theme = historical_theme if is_vague else "general"
                    theme_keywords = historical_keywords if is_vague else query_keywords
                    post_limit = 10
                    reply_limit = 0
                    data_type = "both"
                    processing = {"intent": intent, "top_thread_ids": []}
                    min_likes = 0 if cat_id in ["5", "15"] else 5
                    if intent in ["search_keywords", "find_themed"]:
                        theme = query_keywords[0] if query_keywords else historical_theme
                        theme_keywords = query_keywords or historical_keywords
                    elif intent == "follow_up":
                        theme = historical_theme
                        reply_limit = 200
                        data_type = "replies"
                        post_limit = min(len(referenced_thread_ids), 2) or 2
                        processing["top_thread_ids"] = referenced_thread_ids[:2]
                    elif intent in ["general_query", "introduce"]:
                        reply_limit = 0
                        data_type = "none"
                    
                    return {
                        "direct_response": intent in ["general_query", "introduce"],
                        "intent": intent,
                        "theme": theme,
                        "category_ids": [cat_id],
                        "data_type": data_type,
                        "post_limit": post_limit,
                        "reply_limit": reply_limit,
                        "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": theme_keywords},
                        "processing": processing,
                        "candidate_thread_ids": [],
                        "top_thread_ids": referenced_thread_ids[:2],
                        "needs_advanced_analysis": confidence < 0.7,
                        "reason": reason,
                        "theme_keywords": theme_keywords
                    }
        except Exception as e:
            logger.warning(f"Semantic intent analysis error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            min_likes = 0 if cat_id in ["5", "15"] else 5
            return {
                "direct_response": False,
                "intent": "summarize_posts",
                "theme": historical_theme,
                "category_ids": [cat_id],
                "data_type": "both",
                "post_limit": 5,
                "reply_limit": 0,
                "filters": {"min_replies": 0, "min_likes": min_likes, "keywords": historical_keywords},
                "processing": {"intent": "summarize"},
                "candidate_thread_ids": [],
                "top_thread_ids": [],
                "needs_advanced_analysis": False,
                "reason": f"Semantic analysis failed, defaulting to historical theme: {historical_theme}",
                "theme_keywords": historical_keywords
            }

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent="summarize_posts"):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"top_thread_ids": [], "reason": "Missing API key"}

    if intent == "follow_up":
        referenced_thread_ids = []
        context = st.session_state.get("conversation_context", [])
        if context:
            last_response = context[-1].get("content", "")
            matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
            referenced_thread_ids = [int(tid) for tid in matches if any(t["thread_id"] == int(tid) for t in threads)]
        if referenced_thread_ids:
            logger.info(f"Follow-up intent, using referenced thread IDs: {referenced_thread_ids}")
            return {"top_thread_ids": referenced_thread_ids[:2], "reason": "Using referenced thread IDs for follow_up"}

    prompt_builder = PromptBuilder()
    try:
        prompt = prompt_builder.build_prioritize(
            query=user_query,
            cat_name=cat_name,
            cat_id=cat_id,
            threads=[{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads]
        )
    except Exception as e:
        logger.error(f"Failed to build prioritize prompt: {str(e)}")
        return {"top_thread_ids": [], "reason": f"Prompt building failed: {str(e)}"}
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Thread prioritization failed: status={response.status}, attempt={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Thread prioritization failed: missing choices, attempt={attempt + 1}")
                        continue
                    content = data["choices"][0]["message"]["content"]
                    try:
                        result = json.loads(content)
                        logger.info(f"Thread prioritization succeeded: {result}")
                        return result
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse prioritization result: {content}")
                        return {"top_thread_ids": [], "reason": "Failed to parse API response"}
        except Exception as e:
            logger.warning(f"Thread prioritization error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            sorted_threads = sorted(
                threads,
                key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                reverse=True
            )
            return {
                "top_thread_ids": [t["thread_id"] for t in sorted_threads[:5]],
                "reason": f"Prioritization failed, fallback to popularity sorting"
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, cat_id=None):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5}
    prompt_builder = PromptBuilder()
    
    if not isinstance(processing, dict):
        logger.error(f"Invalid processing format: expected dict, got {type(processing)}")
        yield f"錯誤：無效的處理數據格式（{type(processing)}）。請聯繫支持。"
        return
    intent = processing.get('intent', 'summarize')
    logger.info(f"Processing intent: {intent}, processing: {processing}")

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    # 定義意圖對應的字數範圍（從 prompts.json 提取，字數已提高40%）
    intent_word_ranges = {
        "list": (140, 280),
        "summarize": (420, 700),
        "sentiment": (420, 700),
        "compare": (560, 840),
        "introduce": (70, 140),
        "general": (280, 560),
        "themed": (420, 700),
        "fetch_dates": (280, 560),
        "search_keywords": (420, 700),
        "recommend_threads": (280, 560),
        "monitor_events": (420, 700),
        "classify_opinions": (420, 700),
        "follow_up": (700, 2100),
        "fetch_thread_by_id": (420, 700)
    }
    
    # 獲取當前意圖的字數範圍，默認 summarize
    word_min, word_max = intent_word_ranges.get(intent, (420, 700))
    # 轉換為 token 範圍（1 token ≈ 0.8 字）
    min_tokens = int(word_min / 0.8)
    max_tokens = int(word_max / 0.8)
    # 設置目標 token 為字數範圍中位數，考慮回覆數影響
    target_tokens = int((min_tokens + max_tokens) / 2)
    total_replies_count = sum(len(data.get("replies", [])) for data in (thread_data if isinstance(thread_data, list) else thread_data.values()))
    if total_replies_count:
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens) * 0.9
    target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)
    logger.info(f"Dynamic target_tokens: {target_tokens}, min_tokens={min_tokens}, max_tokens={max_tokens}, total_replies_count={total_replies_count}")

    max_tokens_limit = 4600
    max_tokens = min(target_tokens + 300, max_tokens_limit)
    logger.info(f"Final target_tokens: {target_tokens}, max_tokens_limit={max_tokens_limit}")

    # 動態決定回覆數量
    reply_count_prompt = f"""
    你是資料抓取助手，請根據問題和意圖決定每個帖子應下載的回覆數量（0、25、50、100、200、250、500 條）。
    僅以 JSON 格式回應，禁止生成自然語言或其他格式的內容。
    問題：{user_query}
    意圖：{intent}
    若問題需要深入分析（如情緒分析、意見分類、追問、特定帖子ID），建議較多回覆（200-500）。
    若問題簡單（如標題列出、日期提取），建議較少回覆（25-50）。
    若意圖為「general_query」或「introduce」，無需討論區數據，建議 0 條。
    默認：100 條。
    輸出格式：{{"replies_per_thread": 100, "reason": "決定原因"}}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": reply_count_prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    max_replies_per_thread = 100
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    max_replies_per_thread = min(result.get("replies_per_thread", 100), 500)
                    logger.info(f"Grok selected replies_per_thread: {max_replies_per_thread}, reason: {result.get('reason', 'Default')}")
                else:
                    logger.warning("Failed to determine replies_per_thread, using default 100")
    except Exception as e:
        logger.warning(f"Replies per thread selection failed: {str(e)}, using default 100")
    
    thread_data_dict = {}
    if isinstance(thread_data, list):
        thread_data_dict = {str(data["thread_id"]): data for data in thread_data if isinstance(data, dict) and "thread_id" in data}
    elif isinstance(thread_data, dict):
        thread_data_dict = thread_data
    else:
        logger.error(f"Invalid thread_data format: expected list or dict, got {type(thread_data)}")
        yield f"錯誤：無效的 thread_data 格式（{type(thread_data)}）。請聯繫支持。"
        return
    
    if intent in ["follow_up", "fetch_thread_by_id"]:
        referenced_thread_ids = []
        if intent == "follow_up":
            thread_id, thread_title, last_response = await extract_relevant_thread(conversation_context, user_query)
            if thread_id:
                referenced_thread_ids = [thread_id]
            else:
                last_response = conversation_context[-1].get("content", "") if conversation_context else ""
                matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
                referenced_thread_ids = [tid for tid in matches if any(str(t["thread_id"]) == tid for t in metadata)]
        else:
            top_thread_ids = processing.get("top_thread_ids", [])
            referenced_thread_ids = [tid for tid in top_thread_ids if str(tid) in thread_data_dict]
            logger.info(f"fetch_thread_by_id: top_thread_ids={top_thread_ids}, referenced_thread_ids={referenced_thread_ids}")
        
        if not referenced_thread_ids and intent == "fetch_thread_by_id":
            referenced_thread_ids = processing.get("top_thread_ids", [])
        
        prioritized_thread_data = {tid: thread_data_dict[tid] for tid in map(str, referenced_thread_ids) if tid in thread_data_dict}
        supplemental_thread_data = {tid: data for tid, data in thread_data_dict.items() if tid not in map(str, referenced_thread_ids)}
        thread_data_dict = {**prioritized_thread_data, **supplemental_thread_data}
        logger.info(f"Updated thread_data_dict: prioritized={list(prioritized_thread_data.keys())}, supplemental={list(supplemental_thread_data.keys())}")

    filtered_thread_data = {}
    total_replies_count = 0
    
    for tid, data in thread_data_dict.items():
        try:
            replies = data.get("replies", [])
            if not isinstance(replies, list):
                logger.warning(f"Invalid replies format for thread_id={tid}: expected list, got {type(replies)}")
                replies = []
            filtered_replies = [
                r for r in replies
                if isinstance(r, dict) and r.get("msg") and len(r["msg"].strip()) > 7 and r["msg"].strip() not in ["[圖片]", "[無內容]"]
            ]
            sorted_replies = sorted(
                filtered_replies,
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:max_replies_per_thread]
            
            total_replies_count += len(sorted_replies)
            filtered_thread_data[tid] = {
                "thread_id": data.get("thread_id", tid),
                "title": data.get("title", ""),
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": sorted_replies,
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(sorted_replies)
            }
            logger.info(f"Filtered replies for thread_id={tid}: {len(sorted_replies)}/{len(replies)}")
        except Exception as e:
            logger.error(f"Error processing thread_id={tid}: {str(e)}")
            yield f"錯誤：處理帖子（tid={tid}）失敗（{str(e)}）。請聯繫支持。"
            return
    
    if total_replies_count < max_replies_per_thread and intent in ["follow_up", "fetch_thread_by_id"]:
        logger.info(f"Replies insufficient: {total_replies_count}/{max_replies_per_thread}, fetching more pages")
        for tid, data in filtered_thread_data.items():
            if data["total_fetched_replies"] < max_replies_per_thread:
                additional_pages = [page + 1 for page in data["fetched_pages"]][-2:]
                content_result = await get_lihkg_thread_content(
                    thread_id=tid,
                    cat_id=cat_id,
                    max_replies=max_replies_per_thread - data["total_fetched_replies"],
                    fetch_last_pages=2,
                    specific_pages=additional_pages,
                    start_page=max(data["fetched_pages"], default=0) + 1
                )
                if content_result.get("replies"):
                    cleaned_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                        }
                        for reply in content_result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                    ]
                    filtered_additional_replies = [
                        r for r in cleaned_replies
                        if len(r["msg"].strip()) > 7 and r["msg"].strip() not in ["[圖片]", "[無內容]"]
                    ]
                    updated_data = {
                        "thread_id": data.get("thread_id", tid),
                        "title": data.get("title", ""),
                        "no_of_reply": data.get("no_of_reply", 0),
                        "last_reply_time": data.get("last_reply_time", ""),
                        "like_count": data.get("like_count", 0),
                        "dislike_count": data.get("dislike_count", 0),
                        "replies": data.get("replies", []) + filtered_additional_replies,
                        "fetched_pages": list(set(data.get("fetched_pages", []) + content_result.get("fetched_pages", []))),
                        "total_fetched_replies": len(data.get("replies", []) + filtered_additional_replies)
                    }
                    filtered_thread_data[tid] = updated_data
                    total_replies_count += len(filtered_additional_replies)
                    logger.info(f"Updated filtered_thread_data for tid={tid}: new_replies={len(filtered_additional_replies)}, total_fetched_replies={updated_data['total_fetched_replies']}")
                    st.session_state.thread_cache[tid] = {
                        "data": updated_data,
                        "timestamp": time.time()
                    }

    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata:
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": [],
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": 0
            } for tid, data in filtered_thread_data.items()
        }
        total_replies_count = 0
    
    thread_id_prompt = "\n請在回應中明確包含相關帖子 ID，格式為 [帖子 ID: xxx]。禁止包含 [post_id: ...] 格式。"
    prompt = prompt_builder.build_response(
        intent=intent,
        query=user_query,
        selected_cat=selected_cat,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=list(filtered_thread_data.values()),
        filters=filters
    ) + thread_id_prompt
    
    prompt_length = len(prompt)
    if prompt_length > GROK3_TOKEN_LIMIT:
        max_replies_per_thread = max_replies_per_thread // 2
        total_replies_count = 0
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": data["replies"][:max_replies_per_thread],
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(data["replies"][:max_replies_per_thread])
            } for tid, data in filtered_thread_data.items()
        }
        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
        prompt = prompt_builder.build_response(
            intent=intent,
            query=user_query,
            selected_cat=selected_cat,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=list(filtered_thread_data.values()),
            filters=filters
        ) + thread_id_prompt
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens) * 0.9
        target_tokens = min(max(int(target_tokens), min_tokens), max_tokens_limit)
        logger.info(f"Truncated prompt: original_length={prompt_length}, new_length={len(prompt)}, new_target_tokens: {target_tokens}")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": prompt_builder.get_system_prompt("response")},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": target_tokens,
        "temperature": 0.7,
        "stream": True
    }
    
    logger.info(f"Starting response generation for query: {user_query}")
    
    response_content = ""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    logger.error(f"Response generation failed: status={response.status}")
                    yield f"錯誤：生成回應失敗（狀態碼 {response.status}）。請稍後重試。"
                    return
                
                async for line in response.content:
                    if line and not line.isspace():
                        line_str = line.decode('utf-8').strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data:"):
                            try:
                                chunk = json.loads(line_str[6:])
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    if "###" in content and ("Content Moderation" in content or "Blocked" in content):
                                        logger.warning(f"Content moderation detected: {content}")
                                        raise ValueError("Content moderation detected")
                                    cleaned_content = clean_response(content)
                                    response_content += cleaned_content
                                    yield cleaned_content
                            except json.JSONDecodeError:
                                logger.warning(f"JSON decode error in stream chunk")
                                continue
                
                logger.info(f"Response completed: length={len(response_content)}, target={target_tokens}")
                if len(response_content) < target_tokens * 0.9:
                    logger.warning(f"Response too short: length={len(response_content)}, target={target_tokens}")
                    yield f"\n提示：回應長度（{len(response_content)}）未達預期（{target_tokens}）。請嘗試更具體的查詢或稍後重試。"
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
        finally:
            await session.close()

def clean_cache(max_age=3600):
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]

def unix_to_readable(timestamp):
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert timestamp {timestamp}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    configure_logger("lihkg_api", "lihkg_api.log")

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, conversation_context=None, progress_callback=None):
    configure_lihkg_api_logger()
    try:
        logger.info(f"Processing user question: {user_query}, category: {selected_cat}")
        
        clean_cache()
        
        if rate_limit_until > time.time():
            logger.warning(f"Rate limit active until {rate_limit_until}")
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        post_limit = min(analysis.get("post_limit", 5), 20)
        reply_limit = analysis.get("reply_limit", 0)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 0)
        min_likes = filters.get("min_likes", 0 if cat_id in ["5", "15"] else 5)
        top_thread_ids = analysis.get("top_thread_ids", [])
        intent = analysis.get("intent", "summarize_posts")
        
        if intent in ["fetch_thread_by_id", "follow_up"] and top_thread_ids:
            logger.info(f"{intent} intent with top_thread_ids: {top_thread_ids}")
            thread_data = []
            rate_limit_info = []
            
            candidate_threads = [{"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0} for tid in top_thread_ids]
            
            tasks = []
            for thread_id in top_thread_ids:
                thread_id_str = str(thread_id)
                if thread_id_str in st.session_state.thread_cache and st.session_state.thread_cache[thread_id_str]["data"].get("replies"):
                    thread_data.append(st.session_state.thread_cache[thread_id_str]["data"])
                    continue
                tasks.append(get_lihkg_thread_content(
                    thread_id=thread_id_str,
                    cat_id=cat_id,
                    max_replies=reply_limit,
                    fetch_last_pages=0,
                    specific_pages=[],
                    start_page=1
                ))
            
            if tasks:
                content_results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(content_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Failed to fetch content for thread {candidate_threads[idx]['thread_id']}: {str(result)}")
                        continue
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    
                    thread_id = str(candidate_threads[idx]["thread_id"])
                    if result.get("title"):
                        cleaned_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0),
                                "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                            }
                            for reply in result.get("replies", [])
                            if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                        ]
                        thread_info = {
                            "thread_id": thread_id,
                            "title": result.get("title"),
                            "no_of_reply": result.get("total_replies", 0),
                            "last_reply_time": unix_to_readable(result.get("last_reply_time", "0")),
                            "like_count": result.get("like_count", 0),
                            "dislike_count": result.get("dislike_count", 0),
                            "replies": cleaned_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(cleaned_replies)
                        }
                        thread_data.append(thread_info)
                        st.session_state.thread_cache[thread_id] = {
                            "data": thread_info,
                            "timestamp": time.time()
                        }
                        logger.info(f"Fetched thread_id={thread_id}, pages={result.get('fetched_pages', [])}, replies={len(cleaned_replies)}, expected_replies={reply_limit}")
            
            if len(thread_data) == 1 and intent == "follow_up":
                logger.info("Single thread matched, searching for supplemental threads")
                keyword_result = await extract_keywords_with_grok(user_query, conversation_context)
                theme_keywords = keyword_result["keywords"]
                logger.info(f"Supplemental search keywords: {theme_keywords}, reason: {keyword_result['reason']}")
                
                supplemental_result = await get_lihkg_topic_list(
                    cat_id=cat_id,
                    start_page=1,
                    max_pages=2
                )
                supplemental_threads = supplemental_result.get("items", [])
                filtered_supplemental = [
                    item for item in supplemental_threads
                    if str(item["thread_id"]) not in top_thread_ids
                    and any(kw.lower() in item["title"].lower() for kw in theme_keywords)
                ][:1]
                request_counter = supplemental_result.get("request_counter", request_counter)
                last_reset = supplemental_result.get("last_reset", last_reset)
                rate_limit_until = supplemental_result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(supplemental_result.get("rate_limit_info", []))
                
                supplemental_tasks = []
                for item in filtered_supplemental:
                    thread_id = str(item["thread_id"])
                    supplemental_tasks.append(get_lihkg_thread_content(
                        thread_id=thread_id,
                        cat_id=cat_id,
                        max_replies=reply_limit,
                        fetch_last_pages=0,
                        specific_pages=[],
                        start_page=1
                    ))
                
                if supplemental_tasks:
                    supplemental_results = await asyncio.gather(*supplemental_tasks, return_exceptions=True)
                    for idx, result in enumerate(supplemental_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Failed to fetch supplemental thread {filtered_supplemental[idx]['thread_id']}: {str(result)}")
                            continue
                        request_counter = result.get("request_counter", request_counter)
                        last_reset = result.get("last_reset", last_reset)
                        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                        rate_limit_info.extend(result.get("rate_limit_info", []))
                        
                        thread_id = str(filtered_supplemental[idx]["thread_id"])
                        if result.get("title"):
                            cleaned_replies = [
                                {
                                    "reply_id": reply.get("reply_id"),
                                    "msg": clean_html(reply.get("msg", "[無內容]")),
                                    "like_count": reply.get("like_count", 0),
                                    "dislike_count": reply.get("dislike_count", 0),
                                    "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                                }
                                for reply in result.get("replies", [])
                                if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                            ]
                            thread_info = {
                                "thread_id": thread_id,
                                "title": result.get("title"),
                                "no_of_reply": result.get("total_replies", 0),
                                "last_reply_time": unix_to_readable(result.get("last_reply_time", "0")),
                                "like_count": filtered_supplemental[idx].get("like_count", 0),
                                "dislike_count": filtered_supplemental[idx].get("dislike_count", 0),
                                "replies": cleaned_replies,
                                "fetched_pages": result.get("fetched_pages", []),
                                "total_fetched_replies": len(cleaned_replies)
                            }
                            thread_data.append(thread_info)
                            st.session_state.thread_cache[thread_id] = {
                                "data": thread_info,
                                "timestamp": time.time()
                            }
                            logger.info(f"Supplemental thread fetched: thread_id={thread_id}, replies={len(cleaned_replies)}")
            
            return {
                "selected_cat": selected_cat,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        if reply_limit == 0:
            logger.info(f"Skipping reply fetch due to reply_limit=0, intent: {intent}")
            thread_data = []
            initial_threads = []
            for page in range(1, 6):
                result = await get_lihkg_topic_list(
                    cat_id=cat_id,
                    start_page=page,
                    max_pages=1
                )
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info = result.get("rate_limit_info", [])
                items = result.get("items", [])
                for item in items:
                    item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
                initial_threads.extend(items)
                if not items:
                    logger.warning(f"No threads fetched for cat_id={cat_id}, page={page}")
                if len(initial_threads) >= 150:
                    initial_threads = initial_threads[:150]
                    break
                if progress_callback:
                    progress_callback(f"已抓取第 {page}/5 頁帖子", 0.1 + 0.2 * (page / 5))
            
            filtered_items = [
                item for item in initial_threads
                if item.get("no_of_reply", 0) >= min_replies and (cat_id in ["5", "15"] or int(item.get("like_count", 0)) >= min_likes)
            ]
            
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                if thread_id not in st.session_state.thread_cache:
                    st.session_state.thread_cache[thread_id] = {
                        "data": {
                            "thread_id": thread_id,
                            "title": item["title"],
                            "no_of_reply": item.get("no_of_reply", 0),
                            "last_reply_time": item["last_reply_time"],
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0),
                            "replies": [],
                            "fetched_pages": []
                        },
                        "timestamp": time.time()
                    }
            
            if intent == "fetch_dates":
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                    reverse=True
                )
                top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
            else:
                if not top_thread_ids and filtered_items:
                    prioritization = await prioritize_threads_with_grok(user_query, filtered_items, selected_cat, cat_id, intent)
                    top_thread_ids = prioritization.get("top_thread_ids", [])
                    if not top_thread_ids:
                        sorted_items = sorted(
                            filtered_items,
                            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                            reverse=True
                        )
                        top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
            
            for thread_id in top_thread_ids:
                thread_id_str = str(thread_id)
                if thread_id_str in st.session_state.thread_cache:
                    thread_data.append(st.session_state.thread_cache[thread_id_str]["data"])
                else:
                    for item in filtered_items:
                        if str(item["thread_id"]) == thread_id_str:
                            thread_info = {
                                "thread_id": thread_id_str,
                                "title": item["title"],
                                "no_of_reply": item.get("no_of_reply", 0),
                                "last_reply_time": item["last_reply_time"],
                                "like_count": item.get("like_count", 0),
                                "dislike_count": item.get("dislike_count", 0),
                                "replies": [],
                                "fetched_pages": []
                            }
                            thread_data.append(thread_info)
                            st.session_state.thread_cache[thread_id_str] = {
                                "data": thread_info,
                                "timestamp": time.time()
                            }
                            break
            
            return {
                "selected_cat": selected_cat,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        thread_data = []
        rate_limit_info = []
        candidate_threads = []
        
        if top_thread_ids:
            logger.info(f"Using top_thread_ids from analysis: {top_thread_ids}")
            candidate_threads = [
                {"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0}
                for tid in top_thread_ids
            ]
        else:
            initial_threads = []
            for page in range(1, 6):
                result = await get_lihkg_topic_list(
                    cat_id=cat_id,
                    start_page=page,
                    max_pages=1
                )
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                items = result.get("items", [])
                for item in items:
                    item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
                initial_threads.extend(items)
                if not items:
                    logger.warning(f"No threads fetched for cat_id={cat_id}, page={page}")
                if len(initial_threads) >= 150:
                    initial_threads = initial_threads[:150]
                    break
                if progress_callback:
                    progress_callback(f"已抓取第 {page}/5 頁帖子", 0.1 + 0.2 * (page / 5))
            
            filtered_items = [
                item for item in initial_threads
                if item.get("no_of_reply", 0) >= min_replies and (cat_id in ["5", "15"] or int(item.get("like_count", 0)) >= min_likes)
            ]
            
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                if thread_id not in st.session_state.thread_cache:
                    st.session_state.thread_cache[thread_id] = {
                        "data": {
                            "thread_id": thread_id,
                            "title": item["title"],
                            "no_of_reply": item.get("no_of_reply", 0),
                            "last_reply_time": item["last_reply_time"],
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0),
                            "replies": [],
                            "fetched_pages": []
                        },
                        "timestamp": time.time()
                    }
            
            if intent == "fetch_dates":
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                    reverse=True
                )
                candidate_threads = sorted_items[:post_limit]
            else:
                if filtered_items:
                    prioritization = await prioritize_threads_with_grok(user_query, filtered_items, selected_cat, cat_id, intent)
                    top_thread_ids = prioritization.get("top_thread_ids", [])
                    if not top_thread_ids:
                        sorted_items = sorted(
                            filtered_items,
                            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                            reverse=True
                        )
                        candidate_threads = sorted_items[:post_limit]
                    else:
                        candidate_threads = [
                            item for item in filtered_items
                            if str(item["thread_id"]) in map(str, top_thread_ids)
                        ][:post_limit]
        
        if progress_callback:
            progress_callback("正在抓取帖子內容", 0.3)
        
        tasks = []
        for item in candidate_threads:
            thread_id = str(item["thread_id"])
            if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
                thread_data.append(st.session_state.thread_cache[thread_id]["data"])
                continue
            tasks.append(get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                max_replies=reply_limit,
                fetch_last_pages=0,
                specific_pages=[],
                start_page=1
            ))
        
        if tasks:
            content_results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(content_results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch content for thread {candidate_threads[idx]['thread_id']}: {str(result)}")
                    continue
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                
                thread_id = str(candidate_threads[idx]["thread_id"])
                if result.get("title"):
                    cleaned_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"))
                        }
                        for reply in result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) != "[無內容]"
                    ]
                    thread_info = {
                        "thread_id": thread_id,
                        "title": result.get("title"),
                        "no_of_reply": result.get("total_replies", 0),
                        "last_reply_time": unix_to_readable(result.get("last_reply_time", "0")),
                        "like_count": candidate_threads[idx].get("like_count", 0),
                        "dislike_count": candidate_threads[idx].get("dislike_count", 0),
                        "replies": cleaned_replies,
                        "fetched_pages": result.get("fetched_pages", []),
                        "total_fetched_replies": len(cleaned_replies)
                    }
                    thread_data.append(thread_info)
                    st.session_state.thread_cache[thread_id] = {
                        "data": thread_info,
                        "timestamp": time.time()
                    }
                    logger.info(f"Fetched thread_id={thread_id}, pages={result.get('fetched_pages', [])}, replies={len(cleaned_replies)}, expected_replies={reply_limit}")
        
        if progress_callback:
            progress_callback("正在準備數據", 0.5)
        
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
        logger.error(f"Error in process_user_question: {str(e)}")
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": [{"message": f"Processing error: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
