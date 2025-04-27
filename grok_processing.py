# grok_processing.py

import aiohttp
import asyncio
import json
import re
import random
import datetime
import time
import logging
import streamlit as st
import os
import hashlib
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content, get_lihkg_thread_content_batch
from logging_config import configure_logger
from cachetools import LRUCache

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = configure_logger(__name__, "grok_processing.log")

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90  # 秒

# 初始化 LRU 緩存，限制大小為 500MB
thread_cache = LRUCache(maxsize=500)

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
                content = f.read()
                self.config = json.loads(content)
                logger.info(f"Loaded prompts.json successfully")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in prompts.json: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load prompts.json: {str(e)}")
            raise

    def build_analyze(self, query, cat_name, cat_id, conversation_context=None, thread_titles=None, metadata=None, thread_data=None):
        config = self.config["analyze"]
        limited_context = (conversation_context or [])[-3:]
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id,
            conversation_context=json.dumps(limited_context, ensure_ascii=False)
        )
        data = config["data"].format(
            thread_titles=json.dumps(thread_titles or [], ensure_ascii=False),
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False)
        )
        prompt = f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        return prompt

    def build_prioritize(self, query, cat_name, cat_id, threads):
        config = self.config.get("prioritize", None)
        if not config:
            logger.error("Prompt configuration for 'prioritize' not found in prompts.json")
            raise ValueError("Prompt configuration for 'prioritize' not found")
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id
        )
        data = config["data"].format(
            threads=json.dumps(threads, ensure_ascii=False)
        )
        prompt = f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        return prompt

    def build_response(self, intent, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        config = self.config["response"].get(intent, self.config["response"]["general"])
        limited_context = (conversation_context or [])[-3:]
        context = config["context"].format(
            query=query,
            selected_cat=selected_cat,
            conversation_context=json.dumps(limited_context, ensure_ascii=False)
        )
        data = config["data"].format(
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False),
            filters=json.dumps(filters, ensure_ascii=False)
        )
        prompt = f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        return prompt

    def get_system_prompt(self, mode):
        return self.config["system"].get(mode, "")

def clean_html(text):
    if not isinstance(text, str):
        text = str(text)
    try:
        original_text = text
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            if "hkgmoji" in original_text:
                text = "[表情符號]"
                logger.info(f"HTML cleaning: replaced with [表情符號], original: {original_text}")
            elif any(ext in original_text.lower() for ext in ['.webp', '.jpg', '.png']):
                text = "[圖片]"
                logger.info(f"HTML cleaning: filtered image, original: {original_text}")
            else:
                logger.info(f"HTML cleaning: empty after cleaning, original: {original_text}")
                text = "[無內容]"
        return text
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}, original: {original_text}")
        return original_text

def clean_response(response):
    if not isinstance(response, str):
        return response
    cleaned = re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)
    if cleaned != response:
        logger.info(f"Cleaned response: removed post_id strings")
    return cleaned

def extract_keywords(query):
    stop_words = {"的", "是", "在", "有", "什麼", "嗎", "請問"}
    words = re.findall(r'\w+', query)
    return [word for word in words if word not in stop_words][:3]

async def summarize_context(conversation_context, current_query="", is_follow_up=False):
    """
    提煉對話歷史主題，僅在 follow_up 意圖下匹配關鍵詞。
    """
    ### 修改：僅在 follow_up 意圖下啟用關鍵詞匹配
    if not conversation_context or not current_query:
        return {"theme": "general", "keywords": [], "referenced_thread_id": None, "referenced_title": None}
    
    limited_context = conversation_context[-3:]
    query_keywords = extract_keywords(current_query) if is_follow_up else []
    
    # 提取歷史回應中的帖子 ID 和標題
    referenced_thread_id = None
    referenced_title = None
    last_response = next((msg.get("content", "") for msg in limited_context[::-1] if msg.get("role") == "assistant"), "")
    thread_id_matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
    
    # 僅在 follow_up 意圖下檢查標題關鍵詞匹配
    if is_follow_up and thread_id_matches:
        for thread_id in thread_id_matches:
            if thread_id in thread_cache:
                title = thread_cache[thread_id]["data"].get("title", "")
                if any(kw in title.lower() for kw in [q.lower() for q in query_keywords]):
                    referenced_thread_id = thread_id
                    referenced_title = title
                    logger.info(f"Follow-up: Matched thread_id={thread_id}, title='{title}' with query keywords={query_keywords}")
                    break
    
    # 若無匹配或非 follow_up，選擇第一個帖子 ID
    if not referenced_thread_id and thread_id_matches:
        referenced_thread_id = thread_id_matches[0]
        if referenced_thread_id in thread_cache:
            referenced_title = thread_cache[referenced_thread_id]["data"].get("title", "")
            logger.info(f"Using fallback thread_id={referenced_thread_id}, title='{referenced_title}'")
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"theme": "general", "keywords": [], "referenced_thread_id": None, "referenced_title": None}
    
    prompt = f"""
    你是對話摘要助手，請分析以下對話歷史，提煉主要主題和關鍵詞（最多3個）。
    當前用戶查詢：{current_query}
    {'若為追問（follow_up），優先選擇標題包含以下關鍵詞的帖子 ID：' + json.dumps(query_keywords, ensure_ascii=False) if is_follow_up else '不需匹配關鍵詞，僅提取歷史主題和帖子 ID。'}
    對話歷史：{json.dumps(limited_context, ensure_ascii=False)}
    輸出格式：{{
        "theme": "主要主題",
        "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
        "referenced_thread_id": "帖子ID或null",
        "referenced_title": "帖子標題或null"
    }}
    """
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    logger.warning(f"Context summarization failed: status={response.status}")
                    return {
                        "theme": "general",
                        "keywords": query_keywords if is_follow_up else [],
                        "referenced_thread_id": referenced_thread_id,
                        "referenced_title": referenced_title
                    }
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                if is_follow_up and result.get("referenced_thread_id") and result.get("referenced_thread_id") != referenced_thread_id:
                    if result["referenced_thread_id"] in thread_cache:
                        title = thread_cache[result["referenced_thread_id"]]["data"].get("title", "")
                        if not any(kw in title.lower() for kw in [q.lower() for q in query_keywords]):
                            logger.info(f"Follow-up: Overriding API thread_id={result['referenced_thread_id']} with matched thread_id={referenced_thread_id}")
                            result["referenced_thread_id"] = referenced_thread_id
                            result["referenced_title"] = referenced_title
                else:
                    result["referenced_thread_id"] = referenced_thread_id or result.get("referenced_thread_id")
                    result["referenced_title"] = referenced_title or result.get("referenced_title")
                logger.info(f"Context summarized: theme={result['theme']}, keywords={result['keywords']}, thread_id={result['referenced_thread_id']}, title={result['referenced_title']}")
                return result
    except Exception as e:
        logger.warning(f"Context summarization error: {str(e)}")
        return {
            "theme": "general",
            "keywords": query_keywords if is_follow_up else [],
            "referenced_thread_id": referenced_thread_id,
            "referenced_title": referenced_title
        }
    ### 修改結束

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，僅在 follow_up 意圖下驗證關鍵詞匹配。
    """
    ### 修改：確保 follow_up 使用上下文匹配的帖子 ID
    conversation_context = (conversation_context or [])[-3:]
    prompt_builder = PromptBuilder()
    
    # 初步檢查是否可能為 follow_up
    is_potential_follow_up = conversation_context and len(conversation_context) >= 2
    context_summary = await summarize_context(conversation_context, user_query, is_follow_up=is_potential_follow_up)
    historical_theme = context_summary.get("theme", "general")
    historical_keywords = context_summary.get("keywords", [])
    referenced_thread_id = context_summary.get("referenced_thread_id")
    referenced_title = context_summary.get("referenced_title")
    
    query_words = set(extract_keywords(user_query))
    is_vague = len(query_words) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"])
    
    # 追問檢測
    is_follow_up = False
    referenced_thread_ids = []
    if is_potential_follow_up:
        last_response = conversation_context[-1].get("content", "")
        matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        thread_id_matches = list(set(matches))
        
        # 優先使用 summarize_context 的 thread_id
        if referenced_thread_id and referenced_thread_id in thread_id_matches:
            referenced_thread_ids = [referenced_thread_id]
            is_follow_up = True
            logger.info(f"Follow-up detected: using summarize_context thread_id={referenced_thread_id}, title='{referenced_title}'")
        else:
            # 驗證歷史帖子 ID 的標題
            if thread_id_matches and metadata:
                for thread_id in thread_id_matches:
                    for thread in metadata:
                        if str(thread.get("thread_id")) == thread_id:
                            title = thread.get("title", "")
                            if any(kw in title.lower() for kw in [q.lower() for q in query_words]):
                                referenced_thread_ids = [thread_id]
                                is_follow_up = True
                                logger.info(f"Follow-up detected: query keywords {query_words} matched title '{title}' for thread_id={thread_id}")
                                break
                    if is_follow_up:
                        break
        
        # 語義關聯檢查
        if not is_follow_up:
            last_user_query = conversation_context[-2].get("content", "")
            common_words = query_words.intersection(set(extract_keywords(last_user_query + " " + last_response)))
            explicit_follow_up = any(keyword in user_query for keyword in ["詳情", "更多", "進一步", "點解", "為什麼", "原因"])
            if len(common_words) >= 1 or explicit_follow_up:
                is_follow_up = True
                referenced_thread_ids = thread_id_matches[:1] if thread_id_matches else []
                logger.info(f"Follow-up detected: common_words={common_words}, explicit_follow_up={explicit_follow_up}, thread_ids={referenced_thread_ids}")
    
    if is_follow_up and not referenced_thread_ids:
        intent = "search_keywords"
        reason = "追問意圖無匹配帖子 ID，回退到關鍵詞搜索"
        theme = query_words.pop() if query_words else historical_theme
        theme_keywords = list(query_words) or historical_keywords
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
            "processing": intent,
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": reason,
            "theme_keywords": theme_keywords
        }
    
    semantic_prompt = f"""
    你是語義分析助手，請比較用戶問題與以下意圖描述，選擇最匹配的意圖。
    若問題模糊，優先延續對話歷史的主題（歷史主題：{historical_theme}）。
    若問題涉及對之前回應的追問，選擇「follow_up」意圖。
    用戶問題：{user_query}
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    歷史主題：{historical_theme}
    歷史關鍵詞：{json.dumps(historical_keywords, ensure_ascii=False)}
    歷史帖子 ID：{json.dumps(referenced_thread_ids, ensure_ascii=False)}
    歷史帖子標題：{referenced_title or '無'}
    意圖描述：
    {json.dumps({
        "list_titles": "列出帖子標題或清單",
        "summarize_posts": "總結帖子內容或討論",
        "analyze_sentiment": "分析帖子或回覆的情緒",
        "compare_categories": "比較不同討論區的話題",
        "general_query": "與LIHKG無關或模糊的問題",
        "find_themed": "尋找特定主題的帖子",
        "fetch_dates": "提取帖子或回覆的日期資料",
        "search_keywords": "根據關鍵詞搜索帖子",
        "recommend_threads": "推薦相關或熱門帖子",
        "monitor_events": "追蹤特定事件或話題的討論",
        "classify_opinions": "將回覆按意見立場分類",
        "follow_up": "追問之前回應中提到的帖子內容"
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
            "processing": "general",
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
                    
                    if is_follow_up:
                        intent = "follow_up"
                        reason = f"檢測到追問，優先使用上下文匹配的帖子 ID={referenced_thread_ids}"
                    
                    theme = historical_theme if is_vague else referenced_title or "general"
                    theme_keywords = historical_keywords if is_vague else list(query_words) or historical_keywords
                    post_limit = 10
                    reply_limit = 0
                    data_type = "both"
                    processing = intent
                    min_likes = 0 if cat_id in ["5", "15"] else 5
                    if intent in ["search_keywords", "find_themed"]:
                        theme = list(query_words)[0] if query_words else historical_theme
                        theme_keywords = list(query_words) or historical_keywords
                    elif intent == "monitor_events":
                        theme = "事件追蹤"
                    elif intent == "classify_opinions":
                        theme = "意見分類"
                        data_type = "replies"
                    elif intent == "recommend_threads":
                        theme = "帖子推薦"
                        post_limit = 5
                    elif intent == "fetch_dates":
                        theme = "日期相關資料"
                        post_limit = 5
                    elif intent == "follow_up":
                        theme = referenced_title or historical_theme
                        reply_limit = 200
                        data_type = "replies"
                        post_limit = 1
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
      	   		        "processing": intent,
     		 	        "candidate_thread_ids": [],
      			        "top_thread_ids": referenced_thread_ids,
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
                "processing": "summarize",
                "candidate_thread_ids": [],
                "top_thread_ids": [],
                "needs_advanced_analysis": False,
                "reason": f"Semantic analysis failed, defaulting to historical theme: {historical_theme}",
                "theme_keywords": historical_keywords
            }
    ### 修改結束

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
            return {"top_thread_ids": referenced_thread_ids[:1], "reason": "Using referenced thread IDs for follow_up"}
        else:
            logger.info(f"No referenced thread IDs for follow_up, proceeding with prioritization")

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
                    logger.info(f"Raw API response for prioritization: {content}")
                    try:
                        result = json.loads(content)
                        if not isinstance(result, dict) or "top_thread_ids" not in result or "reason" not in result:
                            logger.warning(f"Invalid prioritization result format: {content}")
                            return {"top_thread_ids": [], "reason": "Invalid result format: missing required keys"}
                        logger.info(f"Thread prioritization succeeded: {result}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse prioritization result as JSON: {content}, error: {str(e)}")
                        return {"top_thread_ids": [], "reason": f"Failed to parse API response as JSON: {str(e)}"}
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
            top_thread_ids = [t["thread_id"] for t in sorted_threads[:5]]
            return {
                "top_thread_ids": top_thread_ids,
                "reason": f"Prioritization failed after {max_retries} attempts, fallback to popularity sorting"
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, cat_id=None):
    """
    生成流式回應，僅在 follow_up 意圖下使用關鍵詞過濾。
    """
    ### 修改：僅 follow_up 使用關鍵詞過濾，記錄標題
    conversation_context = (conversation_context or [])[-3:]
    filters = filters or {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5}
    prompt_builder = PromptBuilder()
    
    context_summary = await summarize_context(conversation_context, user_query, is_follow_up=processing.get('intent', 'summarize') == "follow_up")
    historical_theme = context_summary.get("theme", "general")
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    reply_count_prompt = f"""
    你是資料抓取助手，請根據問題和意圖決定每個帖子應下載的回覆數量（0、25、50、100、200、250、500 條）。
    僅以 JSON 格式回應，禁止生成自然語言或其他格式的內容。
    問題：{user_query}
    意圖：{processing}
    若問題需要深入分析（如情緒分析、意見分類、追問），建議較多回覆（200-500）。
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
                    if max_replies_per_thread == 0:
                        logger.info(f"Skipping reply download due to replies_per_thread=0 for intent: {processing}")
                else:
                    logger.warning("Failed to determine replies_per_thread, using default 100")
    except Exception as e:
        logger.warning(f"Replies per thread selection failed: {str(e)}, using default 100")
    
    intent = processing.get('intent', 'summarize') if isinstance(processing, dict) else processing
    if intent == "follow_up":
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\d+)\]", conversation_context[-1].get("content", "") if conversation_context else "")
        if not referenced_thread_ids:
            prioritization = await prioritize_threads_with_grok(user_query, metadata, selected_cat, cat_id, intent)
            referenced_thread_ids = prioritization.get("top_thread_ids", [])[:1]
            logger.info(f"No referenced IDs in context, using prioritized ID: {referenced_thread_ids}")
        prioritized_thread_data = {tid: data for tid, data in thread_data.items() if str(tid) in map(str, referenced_thread_ids)}
        supplemental_thread_data = {tid: data for tid, data in thread_data.items() if str(tid) not in map(str, referenced_thread_ids)}
        thread_data = {**prioritized_thread_data, **supplemental_thread_data}
        for tid, data in prioritized_thread_data.items():
            logger.info(f"Processing thread_id={tid}, title='{data.get('title', '未知')}' for follow_up")
    
    filtered_thread_data = {}
    total_replies_count = 0
    for tid, data in thread_data.items():
        replies = data.get("replies", [])
        if intent == "follow_up":
            keywords = extract_keywords(user_query)
            sorted_replies = sorted(
                [r for r in replies if r.get("msg") and r.get("msg") != "[無內容]" and any(kw in r.get("msg", "").lower() for kw in [k.lower() for k in keywords])],
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:max_replies_per_thread]
            if not sorted_replies and replies:
                logger.info(f"No keyword-matched replies for thread_id={tid}, using raw replies")
                sorted_replies = sorted(
                    [r for r in replies if r.get("msg") and r.get("msg") != "[無內容]"],
                    key=lambda x: x.get("like_count", 0),
                    reverse=True
                )[:max_replies_per_thread]
        else:
            sorted_replies = sorted(
                [r for r in replies if r.get("msg") and r.get("msg") != "[無內容]"],
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:max_replies_per_thread]
        
        total_replies_count += len(sorted_replies)
        filtered_thread_data[tid] = {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": sorted_replies,
            "fetched_pages": data.get("fetched_pages", [])
        }
    
    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata:
        logger.info(f"No replies in filtered thread data, using metadata for summary due to intent: {intent}")
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": data.get("last_reply_time", 0),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0),
                "replies": [],
                "fetched_pages": data.get("fetched_pages", [])
            } for tid, data in thread_data.items()
        }
        total_replies_count = 0
    
    min_tokens = 1200
    max_tokens = 3600
    if total_replies_count == 0:
        target_tokens = min_tokens
    else:
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)
        target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)
    logger.info(f"Dynamic max_tokens: {target_tokens}, based on total_replies_count: {total_replies_count}")
    
    thread_id_prompt = "\n請在回應中明確包含相關帖子 ID，格式為 [帖子 ID: xxx]。禁止包含 [post_id: ...] 格式。"
    prompt = prompt_builder.build_response(
        intent=intent,
        query=user_query,
        selected_cat=selected_cat,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=filtered_thread_data,
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
                "fetched_pages": data.get("fetched_pages", [])
            } for tid, data in filtered_thread_data.items()
        }
        for data in filtered_thread_data.values():
            total_replies_count += len(data["replies"])
        prompt = prompt_builder.build_response(
            intent=intent,
            query=user_query,
            selected_cat=selected_cat,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=filtered_thread_data,
            filters=filters
        ) + thread_id_prompt
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)
        target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)
        logger.info(f"Truncated prompt: original_length={prompt_length}, new_length={len(prompt)}, new_max_tokens: {target_tokens}")
    
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
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        status_code = response.status
                        if status_code != 200:
                            response_text = await response.text()
                            logger.warning(f"Response generation failed: status={status_code}, attempt={attempt + 1}")
                            if attempt < 2:
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            yield f"錯誤：API 請求失敗（狀態碼 {status_code}）。請稍後重試。"
                            return
                        
                        async for line in response.content:
                            if line and not line.isspace():
                                line_str = line.decode('utf-8').strip()
                                if line_str == "data: [DONE]":
                                    break
                                if line_str.startswith("data: "):
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
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON decode error in stream chunk: {str(e)}")
                                        continue
                        if not response_content:
                            logger.warning(f"No content generated, attempt={attempt + 1}")
                            if attempt < 2:
                                simplified_thread_data = {
                                    tid: {
                                        "thread_id": data["thread_id"],
                                        "title": data["title"],
                                        "no_of_reply": data.get("no_of_reply", 0),
                                        "like_count": data.get("like_count", 0),
                                        "replies": data["replies"][:5]
                                    } for tid, data in filtered_thread_data.items()
                                }
                                prompt = prompt_builder.build_response(
                                    intent=intent,
                                    query=user_query,
                                    selected_cat=selected_cat,
                                    conversation_context=conversation_context,
                                    metadata=metadata,
                                    thread_data=simplified_thread_data,
                                    filters=filters
                                ) + thread_id_prompt
                                payload["messages"][-1]["content"] = prompt
                                payload["max_tokens"] = min_tokens
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            if metadata:
                                fallback_prompt = prompt_builder.build_response(
                                    intent="summarize",
                                    query=user_query,
                                    selected_cat=selected_cat,
                                    conversation_context=conversation_context,
                                    metadata=metadata,
                                    thread_data={},
                                    filters=filters
                                ) + thread_id_prompt
                                payload["messages"][-1]["content"] = fallback_prompt
                                payload["max_tokens"] = min_tokens
                                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as fallback_response:
                                    if fallback_response.status == 200:
                                        data = await fallback_response.json()
                                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                        if content:
                                            response_content = clean_response(content)
                                            yield response_content
                                            return
                            response_content = clean_response(f"以下是 {selected_cat} 的通用概述：討論涵蓋多主題，網民觀點多元。[帖子 ID: {list(thread_data.keys())[0] if thread_data else '無'}]")
                            yield response_content
                            return
                        logger.info(f"Response generation completed: length={len(response_content)}")
                        logger.info(f"Referenced thread IDs: {re.findall(r'\[帖子 ID: (\d+)\]', response_content)}")
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    logger.warning(f"Response generation error: {str(e)}, attempt={attempt + 1}")
                    if attempt < 2:
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
                                "fetched_pages": data.get("fetched_pages", [])
                            } for tid, data in filtered_thread_data.items()
                        }
                        for data in filtered_thread_data.values():
                            total_replies_count += len(data["replies"])
                        prompt = prompt_builder.build_response(
                            intent=intent,
                            query=user_query,
                            selected_cat=selected_cat,
                            conversation_context=conversation_context,
                            metadata=metadata,
                            thread_data=filtered_thread_data,
                            filters=filters
                        ) + thread_id_prompt
                        payload["messages"][-1]["content"] = prompt
                        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)
                        target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)
                        payload["max_tokens"] = target_tokens
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試。"
                    return
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
        finally:
            await session.close()
    ### 修改結束

def clean_cache(max_age=3600):
    logger.info(f"LRU cache stats: size={len(thread_cache)}, maxsize={thread_cache.maxsize}")

def unix_to_readable(timestamp):
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert timestamp {timestamp}: {str(e)}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    lihkg_logger = configure_logger("lihkg_api", "lihkg_api.log")

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    """
    處理用戶問題，確保 follow_up 抓取足夠回覆。
    """
    ### 修改：確保 follow_up 抓取 200 條回覆
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
        
        if progress_callback:
            progress_callback("正在抓取帖子列表", 0.1)
        
        post_limit = min(analysis.get("post_limit", 5), 20)
        reply_limit = analysis.get("reply_limit", 0)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 0)
        min_likes = filters.get("min_likes", 0 if cat_id in ["5", "15"] else 5)
        sort_method = filters.get("sort", "popular")
        time_range = filters.get("time_range", "recent")
        top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
        previous_thread_ids = previous_thread_ids or []
        intent = analysis.get("intent", "summarize_posts")
        
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
                if item.get("no_of_reply", 0) >= min_replies and (cat_id in ["5", "15"] or int(item.get("like_count", 0)) >= min_likes) and str(item["thread_id"]) not in previous_thread_ids
            ]
            
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                if thread_id not in thread_cache:
                    thread_cache[thread_id] = {
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
                if thread_id_str in thread_cache:
                    thread_data.append(thread_cache[thread_id_str]["data"])
                else:
                    for item in filtered_items:
                        if str(item["thread_id"]) == thread_id_str:
                            thread_data.append({
                                "thread_id": thread_id_str,
                                "title": item["title"],
                                "no_of_reply": item.get("no_of_reply", 0),
                                "last_reply_time": item["last_reply_time"],
                                "like_count": item.get("like_count", 0),
                                "dislike_count": item.get("dislike_count", 0),
                                "replies": [],
                                "fetched_pages": []
                            })
                            thread_cache[thread_id_str] = {
                                "data": thread_data[-1],
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
                {"thread_id": tid, "title": thread_cache.get(str(tid), {}).get("data", {}).get("title", ""), "no_of_reply": 0, "like_count": 0}
                for tid in top_thread_ids if str(tid) in thread_cache
            ]
            if not candidate_threads:
                logger.info(f"No cached threads for top_thread_ids={top_thread_ids}, fetching fresh data")
        
        if not candidate_threads:
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
                if item.get("no_of_reply", 0) >= min_replies and (cat_id in ["5", "15"] or int(item.get("like_count", 0)) >= min_likes) and str(item["thread_id"]) not in previous_thread_ids
            ]
            
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                if thread_id not in thread_cache:
                    thread_cache[thread_id] = {
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
        
        thread_ids_to_fetch = []
        if intent == "follow_up" and top_thread_ids:
            thread_ids_to_fetch = top_thread_ids[:1]
            logger.info(f"Follow-up intent: fetching single thread_id={thread_ids_to_fetch}")
        else:
            for item in candidate_threads:
                thread_id = str(item["thread_id"])
                if thread_id in thread_cache and thread_cache[thread_id]["data"].get("replies"):
                    thread_data.append(thread_cache[thread_id]["data"])
                else:
                    thread_ids_to_fetch.append(thread_id)
        
        if thread_ids_to_fetch:
            batch_result = await get_lihkg_thread_content_batch(
                thread_ids=thread_ids_to_fetch,
                cat_id=cat_id,
                max_replies=reply_limit,
                fetch_last_pages=0,
                specific_pages=[],
                start_page=1
            )
            request_counter = batch_result.get("request_counter", request_counter)
            last_reset = batch_result.get("last_reset", last_reset)
            rate_limit_until = batch_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(batch_result.get("rate_limit_info", []))
            
            for result in batch_result.get("results", []):
                thread_id = result["thread_id"]
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
                        "last_reply_time": unix_to_readable(next((item.get("last_reply_time", "0") for item in candidate_threads if str(item["thread_id"]) == thread_id), "0")),
                        "like_count": next((item.get("like_count", 0) for item in candidate_threads if str(item["thread_id"]) == thread_id), 0),
                        "dislike_count": next((item.get("dislike_count", 0) for item in candidate_threads if str(item["thread_id"]) == thread_id), 0),
                        "replies": cleaned_replies,
                        "fetched_pages": result.get("fetched_pages", [])
                    }
                    thread_data.append(thread_info)
                    thread_cache[thread_id] = {
                        "data": thread_info,
                        "timestamp": time.time()
                    }
        
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
    ### 修改結束
