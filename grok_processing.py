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
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from logging_config import configure_logger

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = configure_logger(__name__, "grok_processing.log")

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90  # 秒

class PromptBuilder:
    """
    提示詞生成器，從 prompts.json 載入模板並動態構建提示詞。
    """
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
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
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
        context = config["context"].format(
            query=query,
            selected_cat=selected_cat,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
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
    """
    清理 HTML 標籤，保留短文本和表情符號，記錄圖片過濾為 INFO。
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        original_text = text
        # 移除 HTML 標籤
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        # 規範化空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 若清空後無內容，檢查是否為表情符號或圖片
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
    """
    清理回應，移除 [post_id: ...] 字串，保留其他格式。
    """
    if not isinstance(response, str):
        return response
    # 移除 [post_id: ...] 格式的字串
    cleaned = re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)
    if cleaned != response:
        logger.info(f"Cleaned response: removed post_id strings")
    return cleaned

def extract_keywords(query):
    """
    提取查詢中的關鍵詞，過濾停用詞。
    """
    stop_words = {"的", "是", "在", "有", "什麼", "嗎", "請問"}
    words = re.findall(r'\w+', query)
    return [word for word in words if word not in stop_words][:3]

async def summarize_context(conversation_context):
    """
    使用 Grok 3 提煉對話歷史主題。
    """
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

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，使用語義嵌入識別意圖，放寬語義要求，動態設置篩選條件。
    """
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
    # 提煉對話歷史主題
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "general")
    historical_keywords = context_summary.get("keywords", [])
    
    # 提取關鍵詞
    query_words = set(extract_keywords(user_query))
    is_vague = len(query_words) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"])
    
    # 增強追問檢測
    is_follow_up = False
    referenced_thread_ids = []
    referenced_titles = []
    matched_thread_ids = []
    theme_keywords = []
    
    if conversation_context and len(conversation_context) >= 2:
        last_user_query = conversation_context[-2].get("content", "")
        last_response = conversation_context[-1].get("content", "")
        
        # 提取歷史回應中的帖子 ID 和標題
        matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        referenced_thread_ids = list(set(matches))  # 去重
        for tid in referenced_thread_ids:
            for thread in metadata or []:
                if str(thread.get("thread_id")) == tid:
                    referenced_titles.append(thread.get("title", ""))
        
        # 檢查語義關聯
        common_words = query_words.intersection(set(extract_keywords(last_user_query + " " + last_response)))
        title_overlap = []
        for title in referenced_titles:
            title_words = set(extract_keywords(title))
            overlap = query_words.intersection(title_words)
            if overlap:
                title_overlap.append((title, overlap))
        
        explicit_follow_up = any(keyword in user_query for keyword in ["詳情", "更多", "進一步", "點解", "為什麼", "原因"])
        direct_thread_reference = any(f"[帖子 ID: {tid}]" in user_query for tid in referenced_thread_ids)
        
        if direct_thread_reference or explicit_follow_up or title_overlap:
            is_follow_up = True
            logger.info(f"Follow-up intent detected, referenced thread IDs: {referenced_thread_ids}, title_overlap: {title_overlap}, common_words: {common_words}, direct_reference: {direct_thread_reference}")
            
            # 提取匹配的帖子 ID 和關鍵詞
            if title_overlap:
                for title, overlap in title_overlap:
                    for thread in metadata or []:
                        if thread.get("title") == title and str(thread.get("thread_id")) in referenced_thread_ids:
                            matched_thread_ids.append(str(thread.get("thread_id")))
                    theme_keywords.extend(list(overlap))
                theme_keywords = list(set(theme_keywords))[:3]
                logger.info(f"Matched thread IDs: {matched_thread_ids}, theme_keywords: {theme_keywords}")
            elif direct_thread_reference:
                matched_thread_ids = [tid for tid in referenced_thread_ids if f"[帖子 ID: {tid}]" in user_query]
                theme_keywords = query_words
                logger.info(f"Direct thread reference detected, matched thread IDs: {matched_thread_ids}, theme_keywords: {theme_keywords}")
            else:
                matched_thread_ids = referenced_thread_ids
                theme_keywords = query_words.union(set(historical_keywords))
                logger.info(f"No title overlap, using all referenced thread IDs: {matched_thread_ids}, theme_keywords: {theme_keywords}")
        
        if is_follow_up and not matched_thread_ids:
            logger.info("No matched thread IDs found, falling back to search_keywords")
            is_follow_up = False
            intent = "search_keywords"
            reason = "追問無匹配歷史帖子，回退到關鍵詞搜索"
            theme = extract_keywords(user_query)[0] if extract_keywords(user_query) else historical_theme
            theme_keywords = extract_keywords(user_query) or historical_keywords
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
    
    # 若檢測到追問，設置參數
    if is_follow_up:
        intent = "follow_up"
        reason = f"檢測到追問，匹配帖子 ID: {matched_thread_ids}, 標題語義重疊: {title_overlap}"
        theme = title_overlap[0][0] if title_overlap else historical_theme
        post_limit = max(2, min(len(matched_thread_ids), 5))
        reply_limit = 500
        data_type = "replies"
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": intent,
            "theme": theme,
            "category_ids": [cat_id],
            "data_type": data_type,
            "post_limit": post_limit,
            "reply_limit": reply_limit,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": theme_keywords},
            "processing": intent,
            "candidate_thread_ids": [],
            "top_thread_ids": matched_thread_ids,
            "needs_advanced_analysis": False,
            "reason": reason,
            "theme_keywords": theme_keywords
        }
    
    # 準備語義比較提示詞
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
                    
                    # 若問題模糊，延續歷史意圖或默認 summarize_posts
                    if is_vague and historical_theme != "general":
                        intent = "summarize_posts"
                        reason = f"問題模糊，延續歷史主題：{historical_theme}"
                    elif is_vague:
                        intent = "summarize_posts"
                        reason = "問題模糊，默認總結帖子"
                    
                    # 根據意圖設置參數
                    theme = historical_theme if is_vague else "general"
                    theme_keywords = historical_keywords if is_vague else extract_keywords(user_query)
                    post_limit = 10
                    reply_limit = 0
                    data_type = "both"
                    processing = intent
                    min_likes = 0 if cat_id in ["5", "15"] else 5
                    if intent in ["search_keywords", "find_themed"]:
                        theme = extract_keywords(user_query)[0] if extract_keywords(user_query) else historical_theme
                        theme_keywords = extract_keywords(user_query) or historical_keywords
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
                    elif intent == "general_query" or intent == "introduce":
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
                        "top_thread_ids": [],
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

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent="summarize_posts", analysis=None):
    """
    使用 Grok 3 根據問題語義排序帖子，返回最相關的帖子ID，強化錯誤處理。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"top_thread_ids": [], "reason": "Missing API key"}

    if intent == "follow_up" and analysis:
        referenced_thread_ids = analysis.get("top_thread_ids", [])
        if referenced_thread_ids:
            logger.info(f"Follow-up intent, using matched thread IDs from analysis: {referenced_thread_ids}")
            return {
                "top_thread_ids": referenced_thread_ids[:min(5, len(referenced_thread_ids))],
                "reason": "Using matched thread IDs from analysis for follow_up"
            }
        else:
            logger.info(f"No matched thread IDs for follow_up, falling back to keyword-based prioritization")
            query_keywords = analysis.get("theme_keywords", extract_keywords(user_query))
            filtered_threads = [
                t for t in threads
                if any(kw in t["title"] for kw in query_keywords)
            ]
            if filtered_threads:
                sorted_threads = sorted(
                    filtered_threads,
                    key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                    reverse=True
                )
                top_thread_ids = [t["thread_id"] for t in sorted_threads[:5]]
                return {
                    "top_thread_ids": top_thread_ids,
                    "reason": f"No matched thread IDs, prioritized by keywords: {query_keywords}"
                }
            else:
                return {
                    "top_thread_ids": [],
                    "reason": "No threads matched query keywords"
                }

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
    使用 Grok 3 API 生成流式回應，確保包含帖子 ID，優化 follow_up 意圖。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5}
    prompt_builder = PromptBuilder()
    
    context_summary = await summarize_context(conversation_context)
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
    analysis = processing if isinstance(processing, dict) else {}
    
    if intent == "follow_up":
        referenced_thread_ids = analysis.get("top_thread_ids", [])
        if not referenced_thread_ids:
            logger.info(f"No matched thread IDs in analysis, falling back to keyword search")
            intent = "search_keywords"
            analysis["intent"] = intent
            analysis["processing"] = intent
            analysis["theme_keywords"] = extract_keywords(user_query)
            analysis["reply_limit"] = 200
            analysis["reason"] = "No matched thread IDs for follow_up, falling back to search_keywords"
        
        # 動態選擇回覆頁數
        prioritized_thread_data = {}
        for tid in referenced_thread_ids:
            if str(tid) in thread_data:
                data = thread_data[str(tid)]
                total_pages = data.get("total_pages", 1)
                fetched_pages = data.get("fetched_pages", [])
                # 若回覆數不足，抓取最後3頁
                if len(data.get("replies", [])) < 50 and total_pages > len(fetched_pages):
                    additional_pages = min(3, total_pages - len(fetched_pages))
                    logger.info(f"Fetching additional pages for thread_id={tid}, total_pages={total_pages}, additional_pages={additional_pages}")
                    content_result = await get_lihkg_thread_content(
                        thread_id=tid,
                        cat_id=cat_id,
                        max_replies=500,
                        specific_pages=list(range(total_pages - additional_pages + 1, total_pages + 1)),
                        start_page=1
                    )
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
                    data["replies"].extend(cleaned_replies)
                    data["fetched_pages"].extend(content_result.get("fetched_pages", []))
                    data["total_replies"] = content_result.get("total_replies", data.get("total_replies", 0))
                    data["total_pages"] = content_result.get("total_pages", data.get("total_pages", 1))
                    st.session_state.thread_cache[str(tid)] = {
                        "data": data,
                        "timestamp": time.time()
                    }
                prioritized_thread_data[str(tid)] = data
        
        supplemental_thread_data = {tid: data for tid, data in thread_data.items() if str(tid) not in map(str, referenced_thread_ids)}
        thread_data = {**prioritized_thread_data, **supplemental_thread_data}
        logger.info(f"Filtered thread_data for follow_up: prioritized={list(prioritized_thread_data.keys())}, supplemental={list(supplemental_thread_data.keys())}")

    filtered_thread_data = {}
    total_replies_count = 0
    for tid, data in thread_data.items():
        replies = data.get("replies", [])
        keywords = extract_keywords(user_query)
        sorted_replies = sorted(
            [r for r in replies if r.get("msg") and r.get("msg") != "[無內容]" and any(kw in r.get("msg", "") for kw in keywords)],
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

def clean_cache(max_age=3600):
    """
    清理過期緩存數據，防止記憶體膨脹。
    """
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]

def unix_to_readable(timestamp):
    """
    將 Unix 時間戳轉換為香港時區的普通時間格式（YYYY-MM-DD HH:MM:SS）。
    """
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:MM:SS")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert timestamp {timestamp}: {str(e)}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    """
    配置 lihkg_api 日誌，確保使用香港時區格式器。
    """
    lihkg_logger = configure_logger("lihkg_api", "lihkg_api.log")

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    """
    處理用戶問題，分階段抓取並分析 LIHKG 帖子，支援並行抓取和動態頁數。
    """
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
        min_likes = filters.get("min_likes", 0)
        sort = filters.get("sort", "popular")
        keywords = filters.get("keywords", [])
        
        # 抓取帖子列表
        topic_list_result = await get_lihkg_topic_list(
            cat_id=cat_id,
            page=1,
            count=post_limit,
            order=sort
        )
        request_counter += 1
        threads = topic_list_result.get("items", [])
        
        if progress_callback:
            progress_callback("正在篩選帖子", 0.3)
        
        # 篩選帖子
        filtered_threads = [
            thread for thread in threads
            if thread.get("no_of_reply", 0) >= min_replies and thread.get("like_count", 0) >= min_likes
            and (not keywords or any(kw in thread.get("title", "") for kw in keywords))
        ]
        
        # 優先排序帖子
        prioritization_result = await prioritize_threads_with_grok(
            user_query=user_query,
            threads=filtered_threads,
            cat_name=selected_cat,
            cat_id=cat_id,
            intent=analysis.get("intent", "summarize_posts"),
            analysis=analysis
        )
        top_thread_ids = prioritization_result.get("top_thread_ids", [])
        prioritization_reason = prioritization_result.get("reason", "No prioritization reason provided")
        
        if not top_thread_ids:
            logger.info("No threads matched after prioritization")
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        if progress_callback:
            progress_callback("正在抓取帖子內容", 0.5)
        
        # 初始化緩存
        if "thread_cache" not in st.session_state:
            st.session_state.thread_cache = {}
        
        # 並行抓取帖子內容
        thread_data = {}
        tasks = []
        for thread_id in top_thread_ids:
            if str(thread_id) in st.session_state.thread_cache and time.time() - st.session_state.thread_cache[str(thread_id)]["timestamp"] < 3600:
                thread_data[str(thread_id)] = st.session_state.thread_cache[str(thread_id)]["data"]
            else:
                tasks.append(get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=cat_id,
                    max_replies=reply_limit,
                    specific_pages=None,
                    start_page=1
                ))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for thread_id, result in zip(top_thread_ids[len(thread_data):], results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch thread {thread_id}: {str(result)}")
                    continue
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
                thread_data[str(thread_id)] = {
                    "thread_id": thread_id,
                    "title": result.get("title", ""),
                    "no_of_reply": result.get("total_replies", 0),
                    "last_reply_time": result.get("last_reply_time", 0),
                    "like_count": result.get("like_count", 0),
                    "dislike_count": result.get("dislike_count", 0),
                    "replies": cleaned_replies,
                    "fetched_pages": result.get("fetched_pages", []),
                    "total_pages": result.get("total_pages", 1)
                }
                st.session_state.thread_cache[str(thread_id)] = {
                    "data": thread_data[str(thread_id)],
                    "timestamp": time.time()
                }
                request_counter += 1
        
        if progress_callback:
            progress_callback("正在生成回應", 0.8)
        
        # 生成流式回應
        response_generator = stream_grok3_response(
            user_query=user_query,
            metadata=filtered_threads,
            thread_data=thread_data,
            processing=analysis,
            selected_cat=selected_cat,
            conversation_context=conversation_context,
            needs_advanced_analysis=analysis.get("needs_advanced_analysis", False),
            reason=prioritization_reason,
            filters=filters,
            cat_id=cat_id
        )
        
        response_content = ""
        async for chunk in response_generator:
            response_content += chunk
        
        if progress_callback:
            progress_callback("完成", 1.0)
        
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": [],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis,
            "response": response_content
        }
    
    except Exception as e:
        logger.error(f"Error processing user question: {str(e)}")
        if progress_callback:
            progress_callback("錯誤", 1.0)
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": [{"message": f"Error: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
