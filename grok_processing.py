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
import tzlocal
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 日誌配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S,%f")[:-3] + " HKT"

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler("grok_processing.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f"System timezone: {tzlocal.get_localzone()}, using HongKongFormatter")

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90

class PromptBuilder:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
        logger.info(f"Loading prompts.json from: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"prompts.json not found: {config_path}")
            raise FileNotFoundError(f"prompts.json not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            logger.info("Loaded prompts.json")

    def build_analyze(self, query, cat_name, cat_id, conversation_context=None, thread_titles=None, metadata=None, thread_data=None):
        config = self.config["analyze"]
        context = f"問題：{query}\n分類：{cat_name} (cat_id={cat_id})\n歷史：{json.dumps(conversation_context or [], ensure_ascii=False)}"
        data = f"標題：{json.dumps(thread_titles or [], ensure_ascii=False)}\n元數據：{json.dumps(metadata or [], ensure_ascii=False)}\n數據：{json.dumps(thread_data or {}, ensure_ascii=False)}"
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def build_prioritize(self, query, cat_name, cat_id, threads, theme_keywords=None):
        config = self.config["prioritize"]
        context = f"問題：{query}\n分類：{cat_name} (cat_id={cat_id})\n關鍵詞：{json.dumps(theme_keywords or [], ensure_ascii=False)}"
        data = f"帖子：{json.dumps(threads, ensure_ascii=False)}"
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def build_response(self, intent, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        config = self.config["response"].get(intent, self.config["response"]["general"])
        context = f"問題：{query}\n分類：{selected_cat}\n歷史：{json.dumps(conversation_context or [], ensure_ascii=False)}"
        data = f"元數據：{json.dumps(metadata or [], ensure_ascii=False)}\n數據：{json.dumps(thread_data or {}, ensure_ascii=False)}\n篩選：{json.dumps(filters or {}, ensure_ascii=False)}"
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def get_system_prompt(self, mode):
        return self.config["system"].get(mode, "")

def clean_text(text, is_response=False):
    if not isinstance(text, str):
        text = str(text)
    try:
        # 移除 HTML 標籤
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # 處理特殊內容
        if not text:
            if "hkgmoji" in text:
                text = "[表情符號]"
            elif any(ext in text.lower() for ext in ['.webp', '.jpg', '.png']):
                text = "[圖片]"
            else:
                text = "[無內容]"
            logger.info(f"Cleaned to {text}, original: {text}")
        # 清理回應中的 post_id
        if is_response:
            text = re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', text)
        return text
    except Exception as e:
        logger.error(f"Text clean failed: {str(e)}")
        return text

def extract_keywords(query):
    stop_words = {"的", "是", "在", "有", "什麼", "嗎", "請問"}
    words = re.findall(r'\w+', query)
    keywords = [word for word in words if word not in stop_words][:3]
    if "美股" in query:
        keywords.extend(["美國股市", "納斯達克", "道瓊斯", "股票", "SP500"])
    return list(set(keywords))[:6]

async def summarize_context(conversation_context):
    if not conversation_context:
        return {"theme": "general", "keywords": []}
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {"theme": "general", "keywords": []}
    
    prompt = f"分析對話歷史，提煉主題和關鍵詞(最多6個)。注意意圖(熱門/總結/追問)和標題。\n歷史：{json.dumps(conversation_context, ensure_ascii=False)}\n輸出：{{'theme': '主題', 'keywords': ['關鍵詞1', ...]}}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
                    logger.warning(f"Context summary failed: status={response.status}")
                    return {"theme": "general", "keywords": []}
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                logger.info(f"Context theme: {result['theme']}, keywords: {result['keywords']}")
                return result
    except Exception as e:
        logger.warning(f"Context summary error: {str(e)}")
        return {"theme": "general", "keywords": []}

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "general")
    historical_keywords = context_summary.get("keywords", [])
    
    query_words = set(extract_keywords(user_query))
    is_vague = len(query_words) < 2 and not any(kw in user_query for kw in ["分析", "總結", "討論", "主題", "時事"])
    
    # 追問檢測
    is_follow_up = False
    referenced_thread_ids = []
    referenced_titles = []
    if conversation_context and len(conversation_context) >= 2:
        last_query = conversation_context[-2].get("content", "")
        last_response = conversation_context[-1].get("content", "")
        matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        referenced_thread_ids = matches
        for tid in referenced_thread_ids:
            for thread in metadata or []:
                if str(thread.get("thread_id")) == tid:
                    referenced_titles.append(thread.get("title", ""))
        common_words = query_words.intersection(extract_keywords(last_query + " " + last_response))
        title_overlap = any(any(kw in title for kw in query_words) for title in referenced_titles)
        explicit_follow_up = any(kw in user_query for kw in ["詳情", "更多", "進一步", "點解", "為什麼", "原因"])
        if len(common_words) >= 1 or title_overlap or explicit_follow_up:
            is_follow_up = True
            logger.info(f"Follow-up detected: IDs={referenced_thread_ids}, overlap={title_overlap}, common={common_words}")
    
    if is_follow_up and not referenced_thread_ids:
        intent = "search_keywords"
        reason = "追問無歷史ID，回退關鍵詞搜索"
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
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "relevance", "keywords": theme_keywords},
            "processing": intent,
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": reason,
            "theme_keywords": theme_keywords
        }
    
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        min_likes = 0 if cat_id in ["5", "15"] else 5
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": historical_theme,
            "category_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "reply_limit": 0,
            "filters": {"min_replies": 0, "min_likes": min_likes},
            "processing": "general",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "Missing API key",
            "theme_keywords": historical_keywords
        }
    
    intent_config = {
        "list_titles": "列標題",
        "summarize_posts": "總結內容",
        "analyze_sentiment": "情緒分析",
        "compare_categories": "比較版塊",
        "general_query": "無關問題",
        "find_themed": "主題帖子",
        "fetch_dates": "日期資料",
        "search_keywords": "關鍵詞搜索",
        "recommend_threads": "推薦帖子",
        "monitor_events": "事件追蹤",
        "classify_opinions": "意見分類",
        "follow_up": "追問"
    }
    prompt = f"比較問題與意圖，選最匹配意圖。模糊問題參考歷史主題({historical_theme})，追問(詳情/更多/原因或標題重疊)選follow_up。\n問題：{user_query}\n歷史：{json.dumps(conversation_context, ensure_ascii=False)}\n意圖：{json.dumps(intent_config, ensure_ascii=False)}\n輸出：{{'intent': str, 'confidence': float, 'reason': str}}"
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    messages = [
        {"role": "system", "content": prompt_builder.get_system_prompt("analyze")},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    logger.info(f"Intent analysis for query: {user_query}")
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Intent analysis failed: status={response.status}")
                        continue
                    data = await response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intent = result.get("intent", "summarize_posts")
                    confidence = result.get("confidence", 0.7)
                    reason = result.get("reason", "語義匹配")
                    
                    if is_vague:
                        intent = "summarize_posts"
                        reason = f"模糊問題，{'延續歷史：' + historical_theme if historical_theme != 'general' else '默認總結'}"
                    if is_follow_up:
                        intent = "follow_up"
                        reason = "檢測追問，與歷史標題重疊"
                    
                    theme = historical_theme if is_vague else "general"
                    theme_keywords = extract_keywords(user_query) or historical_keywords
                    post_limit = 10
                    reply_limit = 0
                    data_type = "both"
                    processing = intent
                    min_likes = 0 if cat_id in ["5", "15"] else 5
                    if intent in ["search_keywords", "find_themed"]:
                        theme = extract_keywords(user_query)[0] if extract_keywords(user_query) else historical_theme
                    elif intent == "monitor_events":
                        theme = "事件追蹤"
                    elif intent == "classify_opinions":
                        theme = "意見分類"
                        data_type = "replies"
                    elif intent == "recommend_threads":
                        theme = "帖子推薦"
                        post_limit = 5
                    elif intent == "fetch_dates":
                        theme = "日期資料"
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
            logger.warning(f"Intent analysis error: {str(e)}")
            if attempt < 2:
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
                "reason": f"Analysis failed, default: {historical_theme}",
                "theme_keywords": historical_keywords
            }

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent="summarize_posts", theme_keywords=None):
    logger.info(f"Sorting threads: query={user_query}, intent={intent}")
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {"top_thread_ids": [], "reason": "Missing API key"}
    
    if intent == "follow_up":
        context = st.session_state.get("conversation_context", [])
        if context:
            last_response = context[-1].get("content", "")
            matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
            referenced_thread_ids = [int(tid) for tid in matches if any(t["thread_id"] == int(tid) for t in threads)]
            if referenced_thread_ids:
                logger.info(f"Follow-up: using IDs={referenced_thread_ids}")
                return {"top_thread_ids": referenced_thread_ids[:2], "reason": "Using referenced IDs"}
    
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
        logger.error(f"Prioritize prompt failed: {str(e)}")
        return {"top_thread_ids": [], "reason": f"Prompt failed: {str(e)}"}
    
    weights = {"relevance": 0.8, "popularity": 0.2}
    if "熱門" in user_query:
        weights = {"relevance": 0.4, "popularity": 0.6}
    elif "最新" in user_query:
        weights = {"relevance": 0.2, "last_reply_time": 0.8}
    
    prompt += f"\n權重：相關性{weights.get('relevance', 0.8)*100}%，熱度{weights.get('popularity', 0.2)*100}%，最新{weights.get('last_reply_time', 0)*100}%"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Prioritize failed: status={response.status}")
                        continue
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    result = json.loads(content)
                    if not isinstance(result, dict) or "top_thread_ids" not in result:
                        logger.warning(f"Invalid prioritize result: {content}")
                        return {"top_thread_ids": [], "reason": "Invalid result"}
                    logger.info(f"Prioritized: {result}")
                    return result
        except Exception as e:
            logger.warning(f"Prioritize error: {str(e)}")
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            keyword_matched = [(t, sum(1 for kw in theme_keywords or [] if kw.lower() in t["title"].lower())) for t in threads]
            keyword_matched.sort(key=lambda x: x[1], reverse=True)
            top_thread_ids = [t[0]["thread_id"] for t in keyword_matched[:5]]
            if not top_thread_ids:
                sorted_threads = sorted(threads, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
                top_thread_ids = [t["thread_id"] for t in sorted_threads[:5]]
            return {"top_thread_ids": top_thread_ids, "reason": "Fallback to keyword/popularity"}

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, cat_id=None):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5}
    prompt_builder = PromptBuilder()
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "general")
    
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    intent = processing.get('intent', 'summarize') if isinstance(processing, dict) else processing
    reply_count_prompt = f"決定回覆數(0/25/50/100/200/250/500)。問題：{user_query}\n意圖：{intent}\n深入分析(情緒/意見/追問)用200-500，簡單(標題/日期)用25-50，general/introduce用0，默認100。\n輸出：{{'replies_per_thread': int, 'reason': str}}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
                    logger.info(f"Replies: {max_replies_per_thread}, reason: {result.get('reason')}")
    except Exception as e:
        logger.warning(f"Replies selection failed: {str(e)}")
    
    if intent == "follow_up":
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\d+)\]", conversation_context[-1].get("content", "") if conversation_context else "")
        if not referenced_thread_ids:
            prioritization = await prioritize_threads_with_grok(user_query, metadata, selected_cat, cat_id, intent, filters.get("keywords", []))
            referenced_thread_ids = prioritization.get("top_thread_ids", [])[:2]
        prioritized_thread_data = {tid: data for tid, data in thread_data.items() if str(tid) in map(str, referenced_thread_ids)}
        thread_data = {**prioritized_thread_data, **{tid: data for tid, data in thread_data.items() if str(tid) not in map(str, referenced_thread_ids)}}
    
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
        filtered_thread_data = {
            tid: {k: v for k, v in data.items() if k != "replies"} | {"replies": [], "fetched_pages": data.get("fetched_pages", [])}
            for tid, data in thread_data.items()
        }
        total_replies_count = 0
    
    min_tokens, max_tokens = 600, 3600
    target_tokens = min_tokens if total_replies_count == 0 else min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)
    logger.info(f"Max tokens: {target_tokens}, replies: {total_replies_count}")
    
    thread_id_prompt = "\n回應含帖子ID，格式[帖子 ID: xxx]，禁[post_id: ...]。"
    prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, filtered_thread_data, filters) + thread_id_prompt
    
    if len(prompt) > GROK3_TOKEN_LIMIT:
        max_replies_per_thread //= 2
        filtered_thread_data = {
            tid: {**data, "replies": data["replies"][:max_replies_per_thread]}
            for tid, data in filtered_thread_data.items()
        }
        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
        prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, filtered_thread_data, filters) + thread_id_prompt
        target_tokens = min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)
        logger.info(f"Truncated prompt, new tokens: {target_tokens}")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
    
    logger.info(f"Generating response for query: {user_query}")
    response_content = ""
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"Response failed: status={response.status}")
                        if attempt < 2:
                            await asyncio.sleep(2 + attempt * 2)
                            continue
                        yield f"錯誤：API失敗（狀態碼 {response.status}）"
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
                                            logger.warning(f"Moderation: {content}")
                                            raise ValueError("Moderation detected")
                                        cleaned_content = clean_text(content, is_response=True)
                                        response_content += cleaned_content
                                        yield cleaned_content
                                except json.JSONDecodeError:
                                    logger.warning("Stream chunk JSON decode error")
                                    continue
                    if response_content:
                        logger.info(f"Response length: {len(response_content)}")
                        return
                    logger.warning(f"No content, attempt={attempt + 1}")
                    if attempt < 2:
                        simplified_thread_data = {
                            tid: {k: v for k, v in data.items() if k != "replies"} | {"replies": data["replies"][:5]}
                            for tid, data in filtered_thread_data.items()
                        }
                        prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, simplified_thread_data, filters) + thread_id_prompt
                        payload["messages"][-1]["content"] = prompt
                        payload["max_tokens"] = min_tokens
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    if metadata:
                        fallback_prompt = prompt_builder.build_response("summarize", user_query, selected_cat, conversation_context, metadata, {}, filters) + thread_id_prompt
                        payload["messages"][-1]["content"] = fallback_prompt
                        payload["max_tokens"] = min_tokens
                        async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as fallback_response:
                            if fallback_response.status == 200:
                                data = await fallback_response.json()
                                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                if content:
                                    yield clean_text(content, is_response=True)
                                    return
                    yield clean_text(f"{selected_cat}概述：多主題討論，觀點多元。[帖子 ID: {list(thread_data.keys())[0] if thread_data else '無'}]", is_response=True)
                    return
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                logger.warning(f"Response error: {str(e)}")
                if attempt < 2:
                    max_replies_per_thread //= 2
                    filtered_thread_data = {
                        tid: {**data, "replies": data["replies"][:max_replies_per_thread]}
                        for tid, data in filtered_thread_data.items()
                    }
                    total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
                    prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, filtered_thread_data, filters) + thread_id_prompt
                    payload["messages"][-1]["content"] = prompt
                    target_tokens = min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)
                    payload["max_tokens"] = target_tokens
                    await asyncio.sleep(2 + attempt * 2)
                    continue
                yield f"錯誤：生成失敗（{str(e)}）"
                return
        yield "錯誤：生成失敗，請聯繫支持"

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
        logger.warning(f"Invalid timestamp: {timestamp}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    lihkg_logger = logging.getLogger('lihkg_api')
    lihkg_logger.setLevel(logging.INFO)
    lihkg_logger.handlers.clear()
    lihkg_handler = logging.StreamHandler()
    lihkg_handler.setFormatter(formatter)
    lihkg_logger.addHandler(lihkg_handler)
    file_handler = logging.FileHandler("lihkg_api.log")
    file_handler.setFormatter(formatter)
    lihkg_logger.addHandler(file_handler)

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    configure_lihkg_api_logger()
    logger.info(f"Processing query: {user_query}, cat: {selected_cat}")
    clean_cache()
    
    if rate_limit_until > time.time():
        logger.warning(f"Rate limit until {rate_limit_until}")
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
        progress_callback("抓取帖子列表", 0.1)
    
    post_limit = min(analysis.get("post_limit", 5), 20)
    reply_limit = analysis.get("reply_limit", 0)
    filters = analysis.get("filters", {})
    min_replies = filters.get("min_replies", 0)
    sort_method = filters.get("sort", "relevance")
    time_range = filters.get("time_range", "recent")
    top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
    previous_thread_ids = previous_thread_ids or []
    intent = analysis.get("intent", "summarize_posts")
    
    thread_data = []
    rate_limit_info = []
    initial_threads = []
    
    for page in range(1, 6):
        result = await get_lihkg_topic_list(cat_id, page, 1, request_counter, last_reset, rate_limit_until)
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(result.get("rate_limit_info", []))
        items = result.get("items", [])
        for item in items:
            item["last_reply_time"] = unix_to_readable(item.get("last_reply_time", "0"))
        initial_threads.extend(items)
        if not items:
            logger.warning(f"No threads: cat_id={cat_id}, page={page}")
        if len(initial_threads) >= 150:
            initial_threads = initial_threads[:150]
            break
        if progress_callback:
            progress_callback(f"抓取第{page}/5頁", 0.1 + 0.2 * (page / 5))
    
    if progress_callback:
        progress_callback("篩選帖子", 0.3)
    
    filtered_items = [item for item in initial_threads if item.get("no_of_reply", 0) >= min_replies and str(item["thread_id"]) not in previous_thread_ids]
    
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
            progress_callback("處理日期資料", 0.4)
        sorted_items = sorted(filtered_items, key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"), reverse=True)
        top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
    elif not top_thread_ids and filtered_items:
        if progress_callback:
            progress_callback("重新分析帖子", 0.4)
        prioritization = await prioritize_threads_with_grok(user_query, filtered_items, selected_cat, cat_id, intent, analysis.get("theme_keywords", []))
        top_thread_ids = prioritization.get("top_thread_ids", [])
        if not top_thread_ids:
            sorted_items = sorted(filtered_items, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
            top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
    
    if reply_limit == 0:
        logger.info(f"Skipping replies: intent={intent}")
        thread_data = [st.session_state.thread_cache[str(tid)]["data"] for tid in top_thread_ids if str(tid) in st.session_state.thread_cache]
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    if progress_callback:
        progress_callback("決定抓取頁數", 0.45)
    
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        pages_to_fetch = [1, 2]
        page_type = "latest"
    else:
        page_prompt = f"決定回覆頁數(1-5)和類型(最舊/中段/最新)。問題：{user_query}\n意圖：{intent}\n深入分析/追問/事件用3-5最新頁，簡單用1-2最舊/中段，follow_up抓新頁，general/introduce用0，默認首末頁。\n輸出：{{'pages': [int], 'page_type': str, 'reason': str}}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
                    result = json.loads(data["choices"][0]["message"]["content"])
                    pages_to_fetch = result.get("pages", [1, 2, 3, 4, 5] if intent == "follow_up" else [1, 2])[:5]
                    page_type = result.get("page_type", "latest")
                    logger.info(f"Pages: {pages_to_fetch}, type: {page_type}")
                else:
                    pages_to_fetch = [1, 2, 3, 4, 5] if intent == "follow_up" else [1, 2]
                    page_type = "latest"
    
    if intent in ["general_query", "introduce"]:
        pages_to_fetch = []
    
    if progress_callback:
        progress_callback("抓取帖子內容", 0.5)
    
    candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)] if intent == "follow_up" and top_thread_ids else []
    candidate_threads.extend([item for item in filtered_items if str(item["thread_id"]) not in map(str, top_thread_ids)][:post_limit - len(candidate_threads)])
    
    tasks = []
    for idx, item in enumerate(candidate_threads):
        thread_id = str(item["thread_id"])
        cache_key = thread_id
        cache_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
        if cache_data.get("replies") and cache_data.get("fetched_pages"):
            thread_data.append(cache_data)
            continue
        specific_pages = pages_to_fetch
        if intent == "follow_up" and cache_data.get("fetched_pages"):
            total_pages = cache_data.get("total_pages", 10)
            specific_pages = [p for p in range(1, total_pages + 1) if p not in cache_data["fetched_pages"]][:5] or pages_to_fetch
        tasks.append(get_lihkg_thread_content(thread_id, cat_id, request_counter, last_reset, rate_limit_until, reply_limit, 0, specific_pages, 1))
        if progress_callback:
            progress_callback(f"準備抓取帖子{idx + 1}/{len(candidate_threads)}", 0.5 + 0.3 * ((idx + 1) / len(candidate_threads)))
    
    content_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for idx, content_result in enumerate(content_results):
        if isinstance(content_result, Exception):
            logger.warning(f"Fetch failed: thread_id={candidate_threads[idx]['thread_id']}, error={str(content_result)}")
            continue
        thread_id = str(candidate_threads[idx]["thread_id"])
        cache_key = thread_id
        if content_result.get("replies"):
            for reply in content_result["replies"]:
                reply["msg"] = clean_text(reply.get("msg", ""))
                reply["reply_time"] = unix_to_readable(reply.get("reply_time", "0"))
            total_pages = content_result.get("total_pages", 1)
            fetched_pages = content_result.get("fetched_pages", [])
            cached_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
            combined_pages = sorted(set(cached_data.get("fetched_pages", []) + fetched_pages))
            combined_replies = cached_data.get("replies", []) + content_result["replies"]
            combined_replies = sorted(combined_replies, key=lambda x: x.get("reply_time", "1970-01-01 00:00:00"))
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
            logger.info(f"Cached thread_id={thread_id}, replies={len(unique_replies)}")
        rate_limit_info.extend(content_result.get("rate_limit_info", []))
        request_counter = content_result.get("request_counter", request_counter)
        rate_limit_until = content_result.get("rate_limit_until", rate_limit_until)
    
    if progress_callback:
        progress_callback("準備最終數據", 0.8)
    
    return {
        "selected_cat": selected_cat,
        "thread_data": thread_data,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until,
        "analysis": analysis
    }
