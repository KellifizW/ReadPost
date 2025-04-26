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
import pytz
import tzlocal
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 初始化 session_state
if "thread_cache" not in st.session_state:
    st.session_state.thread_cache = {}

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler("grok_processing.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info(f"系統時區: {tzlocal.get_localzone()}")

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90

async def make_api_call(url, headers, payload, retries=3, timeout=API_TIMEOUT):
    """通用 API 呼叫函數"""
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        logger.warning(f"API 呼叫失敗: status={response.status}, attempt={attempt + 1}")
                        continue
                    try:
                        data = await response.json()
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 解析錯誤: {str(e)}, 回應內容: {await response.text()}")
                        return None
        except Exception as e:
            logger.warning(f"API 呼叫錯誤: {str(e)}, attempt={attempt + 1}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
    return None

class PromptBuilder:
    """簡化提示詞生成器"""
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
        if not os.path.exists(config_path):
            logger.error(f"prompts.json 未找到: {config_path}")
            raise FileNotFoundError(f"prompts.json 未找到")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        logger.info("已載入 prompts.json")

    def _build_prompt(self, mode, intent=None, **kwargs):
        config = self.config[mode] if not intent else self.config[mode].get(intent, self.config[mode]["general"])
        return (f"{config['system']}\n"
                f"{config['context'].format(**kwargs)}\n"
                f"{config['data'].format(**kwargs)}\n"
                f"{config['instructions']}")

    def build_analyze(self, query, cat_name, cat_id, conversation_context=None, thread_titles=None, metadata=None, thread_data=None):
        return self._build_prompt(
            "analyze",
            query=query, cat_name=cat_name, cat_id=cat_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False),
            thread_titles=json.dumps(thread_titles or [], ensure_ascii=False),
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False)
        )

    def build_prioritize(self, query, cat_name, cat_id, threads, theme_keywords=None):
        return self._build_prompt(
            "prioritize",
            query=query, cat_name=cat_name, cat_id=cat_id,
            threads=json.dumps(threads, ensure_ascii=False),
            theme_keywords=json.dumps(theme_keywords or [], ensure_ascii=False)
        )

    def build_response(self, intent, query, selected_cat, conversation_context=None, metadata=None, thread_data=None, filters=None):
        return self._build_prompt(
            "response", intent,
            query=query, selected_cat=selected_cat,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False),
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or {}, ensure_ascii=False),
            filters=json.dumps(filters or {}, ensure_ascii=False)
        )

    def get_system_prompt(self, mode):
        return self.config["system"].get(mode, "")

def clean_html(text):
    """清理 HTML 標籤，保留表情符號和圖片標記"""
    if not text or not isinstance(text, str):
        return "[無內容]"
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text).strip()
    if not text:
        if "hkgmoji" in text:
            return "[表情符號]"
        if any(ext in text.lower() for ext in ['.webp', '.jpg', '.png']):
            return "[圖片]"
        return "[無內容]"
    return re.sub(r'\s+', ' ', text)

def clean_response(response):
    """移除 [post_id: ...] 字串"""
    if isinstance(response, str):
        return re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)
    return response

def extract_keywords(query):
    """提取查詢關鍵詞"""
    stop_words = {"的", "是", "在", "有", "什麼", "嗎", "請問"}
    words = re.findall(r'\w+', query)
    keywords = [word for word in words if word not in stop_words][:3]
    if "美股" in query:
        keywords.extend(["美國股市", "納斯達克", "道瓊斯", "股票", "SP500"])
    return list(set(keywords))[:6]

async def summarize_context(conversation_context):
    """提煉對話歷史主題"""
    if not conversation_context:
        return {"theme": "general", "keywords": []}
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        return {"theme": "general", "keywords": []}

    prompt = f"""
    分析對話歷史，提煉主題和關鍵詞（最多6個）。
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    輸出：{{"theme": "主題", "keywords": []}}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3-beta", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100, "temperature": 0.5}
    
    data = await make_api_call(GROK3_API_URL, headers, payload)
    if data and data.get("choices"):
        return json.loads(data["choices"][0]["message"]["content"])
    logger.warning("對話歷史提煉失敗")
    return {"theme": "general", "keywords": []}

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """分析用戶意圖，設置篩選條件"""
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    context_summary = await summarize_context(conversation_context)
    historical_theme, historical_keywords = context_summary.get("theme", "general"), context_summary.get("keywords", [])

    query_words = set(extract_keywords(user_query))
    is_vague = len(query_words) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"])

    # 檢測功能展示查詢
    if any(keyword in user_query.lower() for keyword in ["示範", "展示", "功能", "可以做", "能力"]):
        return {
            "direct_response": True, "intent": "introduce", "theme": "功能展示", "category_ids": [cat_id], "data_type": "none",
            "post_limit": 0, "reply_limit": 0, "filters": {"min_replies": 0, "min_likes": 0, "sort": "relevance", "keywords": []},
            "processing": "introduce", "candidate_thread_ids": [], "top_thread_ids": [], "needs_advanced_analysis": False, "reason": "功能展示查詢", "theme_keywords": []
        }

    # 追問檢測
    is_follow_up = False
    referenced_thread_ids = []
    if conversation_context and len(conversation_context) >= 2:
        last_user_query = conversation_context[-2].get("content", "")
        last_response = conversation_context[-1].get("content", "")
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
        common_words = query_words.intersection(set(extract_keywords(last_user_query + " " + last_response)))
        title_overlap = any(any(kw in thread.get("title", "") for kw in query_words) for thread in (metadata or []))
        explicit_follow_up = any(keyword in user_query for keyword in ["詳情", "更多", "進一步", "點解", "為什麼", "原因"])
        if len(common_words) >= 1 or title_overlap or explicit_follow_up:
            is_follow_up = True
            logger.info(f"檢測到追問: thread_ids={referenced_thread_ids}, overlap={title_overlap}")

    if is_follow_up and not referenced_thread_ids:
        intent = "search_keywords"
        theme = extract_keywords(user_query)[0] if extract_keywords(user_query) else historical_theme
        theme_keywords = extract_keywords(user_query) or historical_keywords
        return {
            "direct_response": False, "intent": intent, "theme": theme, "category_ids": [cat_id], "data_type": "both",
            "post_limit": 2, "reply_limit": 200, "filters": {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5, "sort": "relevance", "keywords": theme_keywords},
            "processing": intent, "candidate_thread_ids": [], "top_thread_ids": [], "needs_advanced_analysis": False, "reason": "追問無歷史ID，回退關鍵詞搜索", "theme_keywords": theme_keywords
        }

    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        return {
            "direct_response": True, "intent": "general_query", "theme": historical_theme, "category_ids": [], "data_type": "none",
            "post_limit": 5, "reply_limit": 0, "filters": {}, "processing": "general", "candidate_thread_ids": [], "top_thread_ids": [], "needs_advanced_analysis": False, "reason": "缺少 API 密鑰", "theme_keywords": historical_keywords
        }

    prompt = f"""
    比較問題與意圖，選擇最匹配意圖。若模糊，延續歷史主題（{historical_theme}）。
    問題：{user_query}
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    意圖：{json.dumps({"list_titles": "列出標題", "summarize_posts": "總結帖子", "analyze_sentiment": "情緒分析", "compare_categories": "比較版塊", "general_query": "一般問題", "find_themed": "主題搜索", "fetch_dates": "提取日期", "search_keywords": "關鍵詞搜索", "recommend_threads": "推薦帖子", "monitor_events": "事件追蹤", "classify_opinions": "意見分類", "follow_up": "追問"}, ensure_ascii=False)}
    輸出：{{"intent": "意圖", "confidence": 0.0-1.0, "reason": "原因"}}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3-beta", "messages": [{"role": "system", "content": prompt_builder.get_system_prompt("analyze")}, *conversation_context, {"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0.5}

    data = await make_api_call(GROK3_API_URL, headers, payload)
    intent = "summarize_posts"
    reason = "語義匹配"
    confidence = 0.7
    if data and data.get("choices"):
        result = json.loads(data["choices"][0]["message"]["content"])
        intent, confidence, reason = result.get("intent", intent), result.get("confidence", confidence), result.get("reason", reason)

    if is_vague:
        intent = "summarize_posts"
        reason = f"模糊問題，{'延續歷史主題' if historical_theme != 'general' else '默認總結'}"
    if is_follow_up:
        intent = "follow_up"
        reason = "檢測到追問，語義重疊"

    theme = historical_theme if is_vague else "general"
    theme_keywords = extract_keywords(user_query) or historical_keywords
    params = {
        "direct_response": intent in ["general_query", "introduce"], "intent": intent, "theme": theme, "category_ids": [cat_id], "data_type": "both",
        "post_limit": 10, "reply_limit": 0, "filters": {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 5, "sort": "relevance", "keywords": theme_keywords},
        "processing": intent, "candidate_thread_ids": [], "top_thread_ids": referenced_thread_ids, "needs_advanced_analysis": confidence < 0.7, "reason": reason, "theme_keywords": theme_keywords
    }

    if intent in ["search_keywords", "find_themed"]:
        params["theme"] = extract_keywords(user_query)[0] if extract_keywords(user_query) else historical_theme
        params["theme_keywords"] = extract_keywords(user_query) or historical_keywords
    elif intent == "monitor_events":
        params["theme"] = "事件追蹤"
        params["post_limit"] = 15
        params["reply_limit"] = 200
    elif intent == "classify_opinions":
        params["theme"] = "意見分類"
        params["data_type"] = "replies"
        params["post_limit"] = 15
        params["reply_limit"] = 200
    elif intent == "recommend_threads":
        params["theme"] = "帖子推薦"
        params["post_limit"] = 5
    elif intent == "fetch_dates":
        params["theme"] = "日期相關資料"
        params["post_limit"] = 5
    elif intent == "follow_up":
        params["theme"] = historical_theme
        params["reply_limit"] = 500
        params["data_type"] = "replies"
        params["post_limit"] = min(len(referenced_thread_ids), 2) or 2
    elif intent in ["general_query", "introduce"]:
        params["data_type"] = "none"

    return params

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent="summarize_posts", theme_keywords=None):
    """排序帖子，優先語義相關性"""
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        return {"top_thread_ids": [], "reason": "缺少 API 密鑰"}

    if intent == "follow_up":
        context = st.session_state.get("conversation_context", [])
        if context:
            matches = re.findall(r"\[帖子 ID: (\d+)\]", context[-1].get("content", ""))
            referenced_thread_ids = [int(tid) for tid in matches if any(t["thread_id"] == int(tid) for t in threads)]
            if referenced_thread_ids:
                return {"top_thread_ids": referenced_thread_ids[:2], "reason": "使用引用的帖子 ID"}

    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prioritize(
        query=user_query, cat_name=cat_name, cat_id=cat_id,
        threads=[{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads],
        theme_keywords=theme_keywords
    )
    weights = {"relevance": 0.8, "popularity": 0.2} if "熱門" not in user_query else {"relevance": 0.4, "popularity": 0.6}
    if "最新" in user_query:
        weights = {"relevance": 0.2, "last_reply_time": 0.8}
    prompt += f"\n權重：相關性{weights.get('relevance', 0.8)*100}%，熱度{weights.get('popularity', 0.2)*100}%，最新{weights.get('last_reply_time', 0)*100}%"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3-beta", "messages": [{"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.7}

    data = await make_api_call(GROK3_API_URL, headers, payload)
    if data and data.get("choices"):
        try:
            result = json.loads(data["choices"][0]["message"]["content"])
            return result
        except json.JSONDecodeError:
            logger.warning("無法解析排序結果")

    # 回退排序
    keyword_matched_threads = [(t, sum(1 for kw in (theme_keywords or []) if kw.lower() in t["title"].lower())) for t in threads]
    keyword_matched_threads.sort(key=lambda x: x[1], reverse=True)
    top_thread_ids = [t[0]["thread_id"] for t in keyword_matched_threads if t[1] > 0][:5]
    if not top_thread_ids:
        sorted_threads = sorted(threads, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
        top_thread_ids = [t["thread_id"] for t in sorted_threads[:5]]
    return {"top_thread_ids": top_thread_ids, "reason": "回退到關鍵詞或熱度排序"}

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, cat_id=None):
    """生成流式回應，優化追問處理"""
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 0, "min_likes": 0 if cat_id in ["5", "15"] else 0}  # 放寬篩選條件
    prompt_builder = PromptBuilder()
    intent = processing.get('intent', 'summarize') if isinstance(processing, dict) else processing

    if intent == "introduce":
        yield ("作為 LIHKG 論壇數據助手，我可以：\n"
               "- 搜尋並總結熱門帖子\n"
               "- 分析討論趨勢和網民意見\n"
               "- 根據關鍵詞或篩選條件查找帖子\n"
               "- 提供分類概述或事件追蹤\n"
               "- 解答與論壇相關的問題\n"
               f"例如，在「{selected_cat}」，我可以列出最新或最熱門的帖子。請提供具體話題或分類以試試！")
        return

    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        yield "錯誤: 缺少 API 密鑰"
        return

    # 動態決定回覆數
    reply_count_prompt = f"""
    決定帖子回覆數量（0、25、50、100、200、250、500）。
    問題：{user_query}
    意圖：{intent}
    輸出：{{"replies_per_thread": 100, "reason": "原因"}}
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3-beta", "messages": [{"role": "user", "content": reply_count_prompt}], "max_tokens": 100, "temperature": 0.5}
    data = await make_api_call(GROK3_API_URL, headers, payload)
    max_replies_per_thread = 100
    if data and data.get("choices"):
        result = json.loads(data["choices"][0]["message"]["content"])
        max_replies_per_thread = min(result.get("replies_per_thread", 100), 500)

    # 處理追問
    if intent == "follow_up":
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\d+)\]", conversation_context[-1].get("content", "") if conversation_context else "")
        if not referenced_thread_ids:
            prioritization = await prioritize_threads_with_grok(user_query, metadata, selected_cat, cat_id, intent, filters.get("keywords", []))
            referenced_thread_ids = prioritization.get("top_thread_ids", [])[:2]
        thread_data = {tid: data for tid, data in thread_data.items() if str(tid) in map(str, referenced_thread_ids)} | {tid: data for tid, data in thread_data.items() if str(tid) not in map(str, referenced_thread_ids)}

    # 篩選回覆
    filtered_thread_data = {}
    total_replies_count = 0
    for tid, data in thread_data.items():
        replies = sorted(
            [r for r in data.get("replies", []) if r.get("msg") and r.get("msg") != "[無內容]" and any(kw in r.get("msg", "") for kw in extract_keywords(user_query))],
            key=lambda x: x.get("like_count", 0), reverse=True
        )[:max_replies_per_thread] or sorted(
            [r for r in data.get("replies", []) if r.get("msg") and r.get("msg") != "[無內容]"],
            key=lambda x: x.get("like_count", 0), reverse=True
        )[:max_replies_per_thread]
        total_replies_count += len(replies)
        filtered_thread_data[tid] = {**data, "replies": replies}

    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata:
        filtered_thread_data = {tid: {**data, "replies": []} for tid, data in thread_data.items()}
        total_replies_count = 0

    # 動態 token 數
    min_tokens, max_tokens = 600, 3600
    target_tokens = min_tokens if total_replies_count == 0 else min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)

    # 生成提示詞
    prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, filtered_thread_data, filters) + "\n引用帖子使用 [帖子 ID: xxx] 格式，禁止 [post_id: ...]。"
    if len(prompt) > GROK3_TOKEN_LIMIT:
        max_replies_per_thread //= 2
        filtered_thread_data = {tid: {**data, "replies": data["replies"][:max_replies_per_thread]} for tid, data in filtered_thread_data.items()}
        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
        prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, filtered_thread_data, filters) + "\n引用帖子使用 [帖子 ID: xxx] 格式，禁止 [post_id: ...]。"
        target_tokens = min(max(int(min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3-beta", "messages": [{"role": "system", "content": prompt_builder.get_system_prompt("response")}, *conversation_context, {"role": "user", "content": prompt}], "max_tokens": target_tokens, "temperature": 0.7, "stream": True}

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"回應生成失敗: status={response.status}")
                        continue
                    response_content = ""
                    async for line in response.content:
                        if line and not line.isspace() and line.decode('utf-8').strip().startswith("data: "):
                            try:
                                chunk = json.loads(line.decode('utf-8').strip()[6:])
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    if "###" in content and ("Content Moderation" in content or "Blocked" in content):
                                        raise ValueError("檢測到內容審核")
                                    cleaned_content = clean_response(content)
                                    response_content += cleaned_content
                                    yield cleaned_content
                            except json.JSONDecodeError:
                                logger.warning(f"無法解析流式回應塊: {line.decode('utf-8')}")
                                continue
                    if response_content:
                        logger.info(f"回應生成: 長度={len(response_content)}")
                        return
                    logger.warning(f"無內容生成, attempt={attempt + 1}")
                    if attempt < 2:
                        filtered_thread_data = {tid: {**data, "replies": data["replies"][:5]} for tid, data in filtered_thread_data.items()}
                        prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, filtered_thread_data, filters) + "\n引用帖子使用 [帖子 ID: xxx] 格式，禁止 [post_id: ...]。"
                        payload["messages"][-1]["content"] = prompt
                        payload["max_tokens"] = min_tokens
                        await asyncio.sleep(2)
                        continue
                    fallback_prompt = prompt_builder.build_response("summarize", user_query, selected_cat, conversation_context, metadata, {}, filters) + "\n引用帖子使用 [帖子 ID: xxx] 格式，禁止 [post_id: ...]。"
                    payload["messages"][-1]["content"] = fallback_prompt
                    payload["max_tokens"] = min_tokens
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as fallback_response:
                        if fallback_response.status == 200:
                            data = await fallback_response.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if content:
                                yield clean_response(content)
                                return
                    yield clean_response(f"以下是 {selected_cat} 的通用概述：討論涵蓋多主題，網民觀點多元。[帖子 ID: {list(thread_data.keys())[0] if thread_data else '無'}]")
                    return
            except Exception as e:
                logger.warning(f"回應生成錯誤: {str(e)}")
                if attempt < 2:
                    max_replies_per_thread //= 2
                    filtered_thread_data = {tid: {**data, "replies": data["replies"][:max_replies_per_thread]} for tid, data in filtered_thread_data.items()}
                    prompt = prompt_builder.build_response(intent, user_query, selected_cat, conversation_context, metadata, filtered_thread_data, filters) + "\n引用帖子使用 [帖子 ID: xxx] 格式，禁止 [post_id: ...]。"
                    payload["messages"][-1]["content"] = prompt
                    payload["max_tokens"] = min(max(int(min_tokens + (sum(len(data["replies"]) for data in filtered_thread_data.values()) / 500) * (max_tokens - min_tokens)), min_tokens), max_tokens)
                    await asyncio.sleep(2)
                    continue
                yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試。"
                return

def clean_cache(max_age=3600):
    """清理過期緩存"""
    current_time = time.time()
    for key in [k for k, v in st.session_state.thread_cache.items() if current_time - v["timestamp"] > max_age]:
        del st.session_state.thread_cache[key]

def unix_to_readable(timestamp):
    """轉換 Unix 時間戳為香港時區格式"""
    try:
        return datetime.datetime.fromtimestamp(int(timestamp), tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        logger.warning(f"無效時間戳: {timestamp}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    """配置 lihkg_api 日誌"""
    lihkg_logger = logging.getLogger('lihkg_api')
    lihkg_logger.setLevel(logging.INFO)
    if not lihkg_logger.handlers:
        lihkg_logger.handlers.clear()
        lihkg_handler = logging.StreamHandler()
        lihkg_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        lihkg_logger.addHandler(lihkg_handler)
        file_handler = logging.FileHandler("lihkg_api.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        lihkg_logger.addHandler(file_handler)

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    """處理用戶問題，抓取並分析 LIHKG 帖子"""
    configure_lihkg_api_logger()
    clean_cache()
    if rate_limit_until > time.time():
        return {
            "selected_cat": selected_cat, "thread_data": [], "rate_limit_info": [{"message": "速率限制生效", "until": rate_limit_until}],
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until, "analysis": analysis
        }

    if analysis.get("intent") == "introduce":
        return {
            "selected_cat": selected_cat, "thread_data": [], "rate_limit_info": [],
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until, "analysis": analysis
        }

    post_limit = min(analysis.get("post_limit", 5), 20)
    reply_limit = analysis.get("reply_limit", 0)
    filters = analysis.get("filters", {})
    intent = analysis.get("intent", "summarize_posts")
    top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []

    initial_threads = []
    rate_limit_info = []
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
        if len(initial_threads) >= 150:
            initial_threads = initial_threads[:150]
            break
        if progress_callback:
            progress_callback(f"已抓取第 {page}/5 頁帖子", 0.1 + 0.2 * (page / 5))

    filtered_items = [item for item in initial_threads if item.get("no_of_reply", 0) >= filters.get("min_replies", 0) and str(item["thread_id"]) not in (previous_thread_ids or [])]
    for item in initial_threads:
        thread_id = str(item["thread_id"])
        if thread_id not in st.session_state.thread_cache:
            st.session_state.thread_cache[thread_id] = {
                "data": {
                    "thread_id": thread_id, "title": item["title"], "no_of_reply": item.get("no_of_reply", 0),
                    "last_reply_time": item["last_reply_time"], "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0),
                    "replies": [], "fetched_pages": []
                },
                "timestamp": time.time()
            }

    if intent == "fetch_dates":
        sorted_items = sorted(filtered_items, key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"), reverse=True)
        top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
    elif not top_thread_ids and filtered_items:
        prioritization = await prioritize_threads_with_grok(user_query, filtered_items, selected_cat, cat_id, intent, analysis.get("theme_keywords", []))
        top_thread_ids = prioritization.get("top_thread_ids", []) or [item["thread_id"] for item in sorted(filtered_items, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)[:post_limit]]

    if reply_limit == 0:
        thread_data = [st.session_state.thread_cache[str(tid)]["data"] for tid in top_thread_ids if str(tid) in st.session_state.thread_cache]
        return {
            "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until, "analysis": analysis
        }

    candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)] or filtered_items[:post_limit]
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        pages_to_fetch, page_type = [1, 2], "latest"
    else:
        page_prompt = f"""
        決定回覆頁數（1-5）和類型（oldest/middle/latest）。
        問題：{user_query}
        意圖：{intent}
        輸出：{{"pages": [], "page_type": "latest", "reason": "原因"}}
        """
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {"model": "grok-3-beta", "messages": [{"role": "user", "content": page_prompt}], "max_tokens": 100, "temperature": 0.5}
        data = await make_api_call(GROK3_API_URL, headers, payload)
        pages_to_fetch, page_type = ([1, 2, 3, 4, 5] if intent == "follow_up" else [1, 2]), "latest"
        if data and data.get("choices"):
            result = json.loads(data["choices"][0]["message"]["content"])
            pages_to_fetch, page_type = result.get("pages", pages_to_fetch)[:5], result.get("page_type", "latest")

    thread_data = []
    tasks = []
    for idx, item in enumerate(candidate_threads):
        thread_id = str(item["thread_id"])
        cache_data = st.session_state.thread_cache.get(thread_id, {}).get("data", {})
        if cache_data.get("replies") and cache_data.get("fetched_pages"):
            thread_data.append(cache_data)
            continue
        specific_pages = pages_to_fetch
        if intent == "follow_up" and cache_data.get("fetched_pages"):
            specific_pages = [p for p in range(1, cache_data.get("total_pages", 10) + 1) if p not in cache_data["fetched_pages"]][:5] or pages_to_fetch
        tasks.append(get_lihkg_thread_content(thread_id, cat_id, request_counter, last_reset, rate_limit_until, reply_limit, specific_pages=specific_pages))
        if progress_callback:
            progress_callback(f"正在準備抓取帖子 {idx + 1}/{len(candidate_threads)}", 0.5 + 0.3 * ((idx + 1) / len(candidate_threads)))

    content_results = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, result in enumerate(content_results):
        if isinstance(result, Exception):
            logger.warning(f"無法抓取帖子 thread_id={candidate_threads[idx]['thread_id']}: {str(result)}")
            continue
        thread_id = str(candidate_threads[idx]["thread_id"])
        if result.get("replies"):
            for reply in result["replies"]:
                reply["msg"] = clean_html(reply.get("msg", ""))
                reply["reply_time"] = unix_to_readable(reply.get("reply_time", "0"))
            cache_key = thread_id
            cached_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
            combined_replies = sorted(
                list({reply["post_id"]: reply for reply in (cached_data.get("replies", []) + result["replies"])}.values()),
                key=lambda x: x.get("reply_time", "1970-01-01 00:00:00")
            )[:reply_limit]
            thread_data.append({
                "thread_id": thread_id, "title": result.get("title", candidate_threads[idx]["title"]),
                "no_of_reply": result.get("total_replies", candidate_threads[idx]["no_of_reply"]), "last_reply_time": candidate_threads[idx]["last_reply_time"],
                "like_count": candidate_threads[idx].get("like_count", 0), "dislike_count": candidate_threads[idx].get("dislike_count", 0),
                "replies": combined_replies, "fetched_pages": sorted(set(cached_data.get("fetched_pages", []) + result.get("fetched_pages", []))),
                "total_pages": result.get("total_pages", 1)
            })
            st.session_state.thread_cache[cache_key] = {"data": thread_data[-1], "timestamp": time.time()}
        rate_limit_info.extend(result.get("rate_limit_info", []))
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)

    return {
        "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until, "analysis": analysis
    }
