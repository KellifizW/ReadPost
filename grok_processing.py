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
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

# 自定義日誌格式器，將時間戳設為香港時區
class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] + " HKT"

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

# 控制台處理器
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 檔案處理器
file_handler = logging.FileHandler("grok_processing.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 檢查系統時區
import tzlocal
logger.info(f"System timezone: {tzlocal.get_localzone()}, using HongKongFormatter (Asia/Hong_Kong)")

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

    def build_prioritize(self, query, cat_name, cat_id, threads, theme_keywords=None):
        config = self.config.get("prioritize", None)
        if not config:
            logger.error("Prompt configuration for 'prioritize' not found in prompts.json")
            raise ValueError("Prompt configuration for 'prioritize' not found")
        context = config["context"].format(
            query=query,
            cat_name=cat_name,
            cat_id=cat_id,
            theme_keywords=json.dumps(theme_keywords or [], ensure_ascii=False)
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
    提取查詢中的關鍵詞，過濾停用詞，動態擴展相關詞。
    """
    stop_words = {"的", "是", "在", "有", "什麼", "嗎", "請問"}
    words = re.findall(r'\w+', query)
    keywords = [word for word in words if word not in stop_words][:3]
    
    # 動態擴展主題相關關鍵詞
    if "美股" in query:
        keywords.extend(["美國股市", "納斯達克", "道瓊斯", "股票", "SP500"])
    return list(set(keywords))[:6]

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
    你是對話摘要助手，請分析以下對話歷史，提煉主要主題和關鍵詞（最多6個）。
    特別注意用戶問題中的意圖（例如「熱門」「總結」「追問」）和回應中的帖子標題。
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    輸出格式：{{"theme": "主要主題", "keywords": ["關鍵詞1", "關鍵詞2", ...]}}
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
    if conversation_context and len(conversation_context) >= 2:
        last_user_query = conversation_context[-2].get("content", "")
        last_response = conversation_context[-1].get("content", "")
        
        # 提取歷史回應中的帖子 ID 和標題
        matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        referenced_thread_ids = matches
        for tid in referenced_thread_ids:
            for thread in metadata or []:
                if str(thread.get("thread_id")) == tid:
                    referenced_titles.append(thread.get("title", ""))
        
        # 檢查語義關聯
        common_words = query_words.intersection(set(extract_keywords(last_user_query + " " + last_response)))
        title_overlap = any(any(kw in title for kw in query_words) for title in referenced_titles)
        explicit_follow_up = any(keyword in user_query for keyword in ["詳情", "更多", "進一步", "點解", "為什麼", "原因"])
        
        if len(common_words) >= 1 or title_overlap or explicit_follow_up:
            is_follow_up = True
            logger.info(f"Follow-up intent detected, referenced thread IDs: {referenced_thread_ids}, title_overlap: {title_overlap}, common_words: {common_words}")
            if not referenced_thread_ids:
                logger.info("No referenced thread IDs found, falling back to search_keywords")
                is_follow_up = False
    
    # 若無歷史 ID 且檢測到追問，改用 search_keywords
    if is_follow_up and not referenced_thread_ids:
        intent = "search_keywords"
        reason = "追問意圖無歷史帖子 ID，回退到關鍵詞搜索"
        theme = extract_keywords(user_query)[0] if extract_keywords(user_query) else historical_theme
        theme_keywords = extract_keywords(user_query) or historical_keywords
        # 時事台和財經台放寬篩選條件
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": False,
            "intent": intent,
            "theme": theme,
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 2,
            "reply_limit": 200,
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "relevance", "keywords": theme_keywords},
            "processing": intent,
            "candidate_thread_ids": [],
            "top_thread_ids": [],
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
                    
                    # 若檢測到追問，強制設置為 follow_up
                    if is_follow_up:
                        intent = "follow_up"
                        reason = "檢測到追問，與前問題或回應的帖子標題有語義重疊"
                    
                    # 根據意圖設置參數
                    theme = historical_theme if is_vague else "general"
                    theme_keywords = extract_keywords(user_query) or historical_keywords
                    post_limit = 10
                    reply_limit = 0
                    data_type = "both"
                    processing = intent
                    # 時事台和財經台放寬篩選條件
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
                    elif intent == "follow_up":
                        theme = historical_theme
                        reply_limit = 500
                        data_type = "replies"
                        post_limit = min(len(referenced_thread_ids), 2) or 2
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
                        "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "relevance", "keywords": theme_keywords},
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
            # 時事台和財經台放寬篩選條件
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

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent="summarize_posts", theme_keywords=None):
    """
    使用 Grok 3 根據問題語義排序帖子，返回最相關的帖子ID，強化語義優先。
    """
    logger.info(f"Sorting threads for query: {user_query}, intent: {intent}, keywords: {theme_keywords}")
    
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
        else:
            logger.info(f"No referenced thread IDs for follow_up, proceeding with prioritization")

    prompt_builder = PromptBuilder()
    try:
        prompt = prompt_builder.build_prioritize(
            query=user_query,
            cat_name=cat_name,
            cat_id=cat_id,
            threads=[{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads],
            theme_keywords=theme_keywords
        )
    except Exception as e:
        logger.error(f"Failed to build prioritize prompt: {str(e)}")
        return {"top_thread_ids": [], "reason": f"Prompt building failed: {str(e)}"}
    
    # 動態權重
    weights = {"relevance": 0.8, "popularity": 0.2}
    if "熱門" in user_query:
        weights = {"relevance": 0.4, "popularity": 0.6}
    elif "最新" in user_query:
        weights = {"relevance": 0.2, "last_reply_time": 0.8}
    
    prompt += f"\n權重：標題相關性{weights.get('relevance', 0.8)*100}%，熱度(no_of_reply+like_count){weights.get('popularity', 0.2)*100}%，若查詢含‘最新’，則last_reply_time權重{weights.get('last_reply_time', 0)*100}%。"
    
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
                        top_thread_ids = result["top_thread_ids"]
                        reason = result["reason"]
                        logger.info(f"Thread prioritization succeeded: {result}")
                        excluded_threads = [t["thread_id"] for t in threads if t["thread_id"] not in top_thread_ids]
                        logger.info(f"Excluded threads: {excluded_threads}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse prioritization result as JSON: {content}, error: {str(e)}")
                        return {"top_thread_ids": [], "reason": f"Failed to parse API response as JSON: {str(e)}"}
        except Exception as e:
            logger.warning(f"Thread prioritization error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            # 回退排序：基於關鍵詞匹配
            keyword_matched_threads = []
            for thread in threads:
                title = thread["title"].lower()
                score = sum(1 for kw in theme_keywords or [] if kw.lower() in title)
                if score > 0:
                    keyword_matched_threads.append((thread, score))
            keyword_matched_threads.sort(key=lambda x: x[1], reverse=True)
            top_thread_ids = [t[0]["thread_id"] for t in keyword_matched_threads[:5]]
            if not top_thread_ids:
                sorted_threads = sorted(
                    threads,
                    key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                    reverse=True
                )
                top_thread_ids = [t["thread_id"] for t in sorted_threads[:5]]
            excluded_threads = [t["thread_id"] for t in threads if t["thread_id"] not in top_thread_ids]
            logger.info(f"Excluded threads: {excluded_threads}")
            return {
                "top_thread_ids": top_thread_ids,
                "reason": f"Prioritization failed after {max_retries} attempts, fallback to keyword matching or popularity sorting"
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
    if intent == "follow_up":
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\d+)\]", conversation_context[-1].get("content", "") if conversation_context else "")
        if not referenced_thread_ids:
            prioritization = await prioritize_threads_with_grok(user_query, metadata, selected_cat, cat_id, intent, filters.get("keywords", []))
            referenced_thread_ids = prioritization.get("top_thread_ids", [])[:2]
            logger.info(f"No referenced IDs in context, using prioritized IDs: {referenced_thread_ids}")
        prioritized_thread_data = {tid: data for tid, data in thread_data.items() if str(tid) in map(str, referenced_thread_ids)}
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
    
    min_tokens = 600
    max_tokens = 3600  # 放寬到 3600（約 2400 字）
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
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert timestamp {timestamp}: {str(e)}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    """
    配置 lihkg_api 日誌，確保使用香港時區格式器。
    """
    lihkg_logger = logging.getLogger('lihkg_api')
    lihkg_logger.setLevel(logging.INFO)
    lihkg_logger.handlers.clear()
    lihkg_handler = logging.StreamHandler()
    lihkg_handler.setFormatter(formatter)
    lihkg_logger.addHandler(lihkg_handler)
    file_handler = logging.FileHandler("lihkg_api.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    lihkg_logger.addHandler(file_handler)

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
        min_likes = 0  # 放寬篩選條件
        sort_method = filters.get("sort", "relevance")
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
                    max_pages=1,
                    request_counter=request_counter,
                    last_reset=last_reset,
                    rate_limit_until=rate_limit_until
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
                if item.get("no_of_reply", 0) >= min_replies and str(item["thread_id"]) not in previous_thread_ids
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
                    prioritization = await prioritize_threads_with_grok(
                        user_query, filtered_items, selected_cat, cat_id, intent, analysis.get("theme_keywords", [])
                    )
                    top_thread_ids = prioritization.get("top_thread_ids", [])
                    if not top_thread_ids:
                        sorted_items = sorted(
                            filtered_items,
                            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                            reverse=True
                        )
                        top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
            
            thread_data = [
                st.session_state.thread_cache[str(tid)]["data"]
                for tid in top_thread_ids
                if str(tid) in st.session_state.thread_cache
            ]
            
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
        initial_threads = []
        
        for page in range(1, 6):
            result = await get_lihkg_topic_list(
                cat_id=cat_id,
                start_page=page,
                max_pages=1,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until
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
        
        if progress_callback:
            progress_callback("正在篩選帖子", 0.3)
        
        filtered_items = []
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            no_of_reply = item.get("no_of_reply", 0)
            
            if no_of_reply >= min_replies and thread_id not in previous_thread_ids:
                filtered_items.append(item)
        
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
            if progress_callback:
                progress_callback("正在處理日期相關資料", 0.4)
            sorted_items = sorted(
                filtered_items,
                key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                reverse=True
            )
            top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
            logger.info(f"Fetch dates mode, selected threads: {top_thread_ids}")
        else:
            if not top_thread_ids and filtered_items:
                if progress_callback:
                    progress_callback("正在重新分析帖子選擇", 0.4)
                prioritization = await prioritize_threads_with_grok(
                    user_query, filtered_items, selected_cat, cat_id, intent, analysis.get("theme_keywords", [])
                )
                if not isinstance(prioritization, dict):
                    logger.error(f"Invalid prioritization result: {prioritization}")
                    prioritization = {"top_thread_ids": [], "reason": "Invalid prioritization result"}
                top_thread_ids = prioritization.get("top_thread_ids", [])
                if not top_thread_ids:
                    logger.warning(f"Prioritization failed: {prioritization.get('reason', 'Unknown reason')}")
                    sorted_items = sorted(
                        filtered_items,
                        key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                        reverse=True
                    )
                    top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
                    logger.info(f"Fallback to popularity sorting: {top_thread_ids}")
                else:
                    logger.info(f"Grok prioritized threads: {top_thread_ids}")
            
            if not top_thread_ids and filtered_items:
                if sort_method == "relevance":
                    sorted_items = sorted(
                        filtered_items,
                        key=lambda x: sum(1 for kw in analysis.get("theme_keywords", []) if kw.lower() in x["title"].lower()),
                        reverse=True
                    )
                else:
                    sorted_items = sorted(
                        filtered_items,
                        key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                        reverse=(time_range == "recent")
                    )
                top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
                logger.info(f"Generated top_thread_ids: {top_thread_ids}")
        
        if progress_callback:
            progress_callback("正在決定抓取頁數", 0.45)
        
        try:
            GROK3_API_KEY = st.secrets["grok3key"]
        except KeyError as e:
            logger.error(f"Grok 3 API key missing: {str(e)}")
            pages_to_fetch = [1, 2]
            page_type = "latest"
        else:
            page_prompt = f"""
            你是資料抓取助手，請根據問題複雜度和意圖決定需要抓取的回覆頁數（1-5頁）以及頁數類型（最舊、中段、最新）。
            僅以 JSON 格式回應，禁止生成自然語言或其他格式的內容。
            問題：{user_query}
            意圖：{intent}
            若問題需要深入分析（如情緒分析、意見分類、追問）或追蹤事件，建議多頁（3-5），偏向最新頁。
            若問題簡單（如標題列出、日期提取），建議少頁（1-2），偏向最舊或中段頁。
            若意圖為「follow_up」，優先抓取未下載的最新頁（3-5頁）。
            若意圖為「general_query」或「introduce」，建議 0 頁。
            默認：首頁+末頁（[1, 總頁數]）。
            輸出格式：{{"pages": [頁數列表], "page_type": "oldest/middle/latest", "reason": "決定原因"}}
            """
            
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
            payload = {
                "model": "grok-3-beta",
                "messages": [{"role": "user", "content": page_prompt}],
                "max_tokens": 100,
                "temperature": 0.5
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        logger.info(f"Dynamic page selection raw response: {content}")
                        try:
                            result = json.loads(content)
                            if not isinstance(result, dict) or "pages" not in result or "page_type" not in result:
                                logger.warning(f"Invalid dynamic page response format: {content}")
                                pages_to_fetch = [1, 2, 3, 4, 5] if intent == "follow_up" else [1, 2]
                                page_type = "latest"
                            else:
                                pages_to_fetch = result.get("pages", [1, 2, 3, 4, 5] if intent == "follow_up" else [1, 2])[:5]
                                page_type = result.get("page_type", "latest")
                                logger.info(f"Dynamic pages selected: {pages_to_fetch}, type: {page_type}, reason: {result.get('reason', 'Default')}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse dynamic page response: {content}, error: {str(e)}")
                            pages_to_fetch = [1, 2, 3, 4, 5] if intent == "follow_up" else [1, 2]
                            page_type = "latest"
                    else:
                        logger.warning(f"Dynamic page selection API failed: status={response.status}")
                        pages_to_fetch = [1, 2, 3, 4, 5] if intent == "follow_up" else [1, 2]
                        page_type = "latest"
        
        if intent in ["general_query", "introduce"]:
            pages_to_fetch = []
            logger.info(f"Skipping page fetch due to intent: {intent}")
        
        if progress_callback:
            progress_callback("正在抓取候選帖子內容", 0.5)
        
        candidate_threads = []
        referenced_thread_ids = top_thread_ids if intent == "follow_up" else []
        if intent == "follow_up" and referenced_thread_ids:
            candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, referenced_thread_ids)]
            logger.info(f"Follow-up intent, prioritized candidate threads: {[item['thread_id'] for item in candidate_threads]}")
        
        if len(candidate_threads) < post_limit:
            other_threads = [item for item in filtered_items if str(item["thread_id"]) not in map(str, referenced_thread_ids)]
            candidate_threads.extend(other_threads[:post_limit - len(candidate_threads)])
            logger.info(f"Supplemented candidate threads: {[item['thread_id'] for item in candidate_threads]}")
        
        excluded_threads = [item["thread_id"] for item in filtered_items if item["thread_id"] not in [c["thread_id"] for c in candidate_threads]]
        logger.info(f"Excluded threads: {excluded_threads}, reason: not in top_thread_ids or insufficient priority")
        
        tasks = []
        for idx, item in enumerate(candidate_threads):
            thread_id = str(item["thread_id"])
            cache_key = thread_id
            cache_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
            
            if cache_data and cache_data.get("replies") and cache_data.get("fetched_pages"):
                thread_data.append(cache_data)
                continue
            
            specific_pages = pages_to_fetch
            if intent == "follow_up" and cache_data.get("fetched_pages"):
                total_pages = cache_data.get("total_pages", 10)
                specific_pages = [p for p in range(1, total_pages + 1) if p not in cache_data["fetched_pages"]][:5]
                if not specific_pages:
                    specific_pages = pages_to_fetch
                logger.info(f"Follow-up intent, fetching new pages for thread_id={thread_id}: {specific_pages}")
            
            tasks.append(
                get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=cat_id,
                    request_counter=request_counter,
                    last_reset=last_reset,
                    rate_limit_until=rate_limit_until,
                    max_replies=reply_limit,
                    fetch_last_pages=0,
                    specific_pages=specific_pages,
                    start_page=1
                )
            )
            if progress_callback:
                progress_callback(f"正在準備抓取帖子 {idx + 1}/{len(candidate_threads)}", 0.5 + 0.3 * ((idx + 1) / len(candidate_threads)))
        
        content_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, content_result in enumerate(content_results):
            if isinstance(content_result, Exception):
                logger.warning(f"Failed to fetch thread_id={candidate_threads[idx]['thread_id']}: {str(content_result)}")
                continue
            
            thread_id = str(candidate_threads[idx]["thread_id"])
            cache_key = thread_id
            
            if content_result.get("replies"):
                for reply in content_result["replies"]:
                    reply["msg"] = clean_html(reply.get("msg", ""))
                    reply["reply_time"] = unix_to_readable(reply.get("reply_time", "0"))
                
                total_pages = content_result.get("total_pages", 1)
                fetched_pages = content_result.get("fetched_pages", [])
                
                cached_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
                existing_pages = cached_data.get("fetched_pages", [])
                existing_replies = cached_data.get("replies", [])
                
                combined_pages = sorted(set(existing_pages + fetched_pages))
                combined_replies = existing_replies + content_result["replies"]
                
                combined_replies = sorted(
                    combined_replies,
                    key=lambda x: x.get("reply_time", "1970-01-01 00:00:00")
                )
                
                unique_replies = []
                seen_post_ids = set()
                for reply in combined_replies:
                    post_id = reply.get("post_id")
                    if post_id not in seen_post_ids:
                        unique_replies.append(reply)
                        seen_post_ids.add(post_id)
                
                unique_replies = unique_replies[:reply_limit]
                
                thread_data.append({
                    "thread_id": thread_id,
                    "title": content_result.get("title", candidate_threads[idx]["title"]),
                    "no_of_reply": content_result.get("total_replies", candidate_threads[idx]["no_of_reply"]),
                    "last_reply_time": candidate_threads[idx]["last_reply_time"],
                    "like_count": candidate_threads[idx].get("like_count", 0),
                    "dislike_count": candidate_threads[idx].get("dislike_count", 0),
                    "replies": unique_replies,
                    "fetched_pages": combined_pages,
                    "total_pages": total_pages
                })
                
                st.session_state.thread_cache[cache_key] = {
                    "data": {
                        "thread_id": thread_id,
                        "title": content_result.get("title", candidate_threads[idx]["title"]),
                        "no_of_reply": content_result.get("total_replies", candidate_threads[idx]["no_of_reply"]),
                        "last_reply_time": candidate_threads[idx]["last_reply_time"],
                        "like_count": candidate_threads[idx].get("like_count", 0),
                        "dislike_count": candidate_threads[idx].get("dislike_count", 0),
                        "replies": unique_replies,
                        "fetched_pages": combined_pages,
                        "total_pages": total_pages
                    },
                    "timestamp": time.time()
                }
                logger.info(f"Cached thread_id={thread_id}, replies={len(unique_replies)}, pages={combined_pages}")
            
            rate_limit_info.extend(content_result.get("rate_limit_info", []))
            request_counter = content_result.get("request_counter", request_counter)
            last_reset = result.get("last_reset", last_reset)
            rate_limit_until = content_result.get("rate_limit_until", rate_limit_until)
        
        if progress_callback:
            progress_callback("正在準備最終數據", 0.8)
        
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
            "rate_limit_info": [{"message": f"Error: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
