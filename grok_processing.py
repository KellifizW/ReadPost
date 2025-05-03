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
from reddit_api import get_reddit_topic_list, get_reddit_thread_content
from logging_config import configure_logger

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "grok_processing.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90
MAX_CACHE_SIZE = 100  # 最大緩存條目數

# 全局鎖和信號量
cache_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(5)

class PromptBuilder:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "prompts.json")
        
        if not os.path.exists(config_path):
            logger.error(f"未找到 prompts.json：{config_path}")
            raise FileNotFoundError(f"未找到 prompts.json：{config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.loads(f.read())
            # 驗證 prompts.json 結構
            required_keys = ["analyze", "prioritize", "response", "system"]
            missing_keys = [key for key in required_keys if key not in self.config]
            if missing_keys:
                logger.error(f"prompts.json 缺少必要鍵：{missing_keys}")
                raise ValueError(f"prompts.json 缺少必要鍵：{missing_keys}")
        except Exception as e:
            logger.error(f"載入 prompts.json 失敗：{str(e)}")
            raise

    def build_analyze(self, query, source_name, source_id, conversation_context=None):
        config = self.config["analyze"]
        context = config["context"].format(
            query=query, cat_name=source_name, cat_id=source_id,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        return f"{config['system']}\n{context}\n{config['instructions']}"

    def build_prioritize(self, query, source_name, source_id, source_type, threads):
        config = self.config.get("prioritize")
        if not config:
            logger.error("未找到 'prioritize' 的提示配置")
            raise ValueError("未找到 'prioritize' 的提示配置")
        try:
            context = config["context"].format(
                query=query, cat_name=source_name, cat_id=source_id, source_type=source_type or "lihkg"
            )
        except KeyError as e:
            logger.error(f"格式化 prioritize context 失敗，缺少參數：{str(e)}")
            context = config["context"].format(
                query=query, cat_name=source_name, cat_id=source_id, source_type="lihkg"
            )
        data = config["data"].format(threads=json.dumps(threads, ensure_ascii=False))
        return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"

    def build_response(self, intent, query, selected_source, conversation_context=None, metadata=None, thread_data=None, filters=None):
        config = self.config["response"].get(intent, self.config["response"]["general"])
        context = config["context"].format(
            query=query, selected_cat=selected_source,
            conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
        )
        data = config["data"].format(
            metadata=json.dumps(metadata or [], ensure_ascii=False),
            thread_data=json.dumps(thread_data or [], ensure_ascii=False),
            filters=json.dumps(filters or {}, ensure_ascii=False)
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
        logger.error(f"HTML 清理失敗：{str(e)}")
        return text

def clean_response(response):
    if isinstance(response, str):
        cleaned = re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)
        return cleaned
    return response

async def extract_keywords_with_grok(query, conversation_context=None):
    conversation_context = conversation_context or []
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {"keywords": [], "reason": "缺少 API 密鑰", "time_sensitive": False}

    prompt = f"""
請從以下查詢提取 1-3 個核心關鍵詞（僅保留名詞或核心動詞，過濾停用詞如「的」「是」）。關鍵詞應反映主題或意圖。保留「你點睇」作為意圖短語，映射到「分析」。過濾無意義粵語俚語（如「講D咩」）。若查詢包含時間性詞語（如「今晚」「今日」「最近」「呢排」），設置 time_sensitive 為 true。以 JSON 格式返回，附簡要邏輯說明（70字內）。

查詢："{query}"
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}

返回格式：
{{
  "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
  "reason": "提取邏輯說明（70字以內）",
  "time_sensitive": true/false
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
                    status_code = response.status
                    if status_code != 200:
                        logger.debug(f"關鍵詞提取失敗：狀態碼={status_code}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"關鍵詞提取失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    response_content = data["choices"][0]["message"]["content"]
                    logger.info(
                        f"Grok3 API 調用：函數=extract_keywords_with_grok, "
                        f"查詢={query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                        f"輸出 token={completion_tokens}, 回應={response_content}"
                    )
                    result = json.loads(response_content)
                    keywords = result.get("keywords", [])[:3]
                    reason = result.get("reason", "未提供原因")[:70]
                    time_sensitive = result.get("time_sensitive", False)
                    return {"keywords": keywords, "reason": reason, "time_sensitive": time_sensitive}
        except Exception as e:
            logger.debug(f"關鍵詞提取錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            logger.error(f"關鍵詞提取失敗，嘗試 {max_retries} 次後放棄")
            return {"keywords": [], "reason": f"提取失敗：{str(e)}"[:70], "time_sensitive": False}

async def summarize_context(conversation_context):
    if not conversation_context:
        return {"theme": "一般", "keywords": []}
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {"theme": "一般", "keywords": []}
    
    prompt = f"""
你是对話摘要助手，請分析以下對話歷史，提煉主要主題和關鍵詞（最多3個）。
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
                status_code = response.status
                if status_code != 200:
                    logger.warning(f"對話摘要失敗：狀態碼={status_code}")
                    return {"theme": "一般", "keywords": []}
                data = await response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                response_content = data["choices"][0]["message"]["content"]
                logger.info(
                    f"Grok3 API 調用：函數=summarize_context, "
                    f"狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                    f"輸出 token={completion_tokens}, 回應={response_content}"
                )
                result = json.loads(response_content)
                return result
    except Exception as e:
        logger.warning(f"對話摘要錯誤：{str(e)}")
        return {"theme": "一般", "keywords": []}

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
                    return thread_id, title, message["content"]
    return None, None, None

async def analyze_and_screen(user_query, source_name, source_id, source_type="lihkg", conversation_context=None):
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
    id_match = re.search(r'(?:ID|帖子)\s*(\d+)', user_query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        return {
            "direct_response": False,
            "intent": "fetch_thread_by_id",
            "theme": "特定帖子查詢",
            "source_type": source_type,
            "source_ids": [source_id],
            "data_type": "replies",
            "post_limit": 1,
            "filters": {"min_replies": 10, "min_likes": 0, "sort": "popular", "keywords": []},
            "processing": {"intent": "fetch_thread_by_id", "top_thread_ids": [thread_id]},
            "candidate_thread_ids": [thread_id],
            "top_thread_ids": [thread_id],
            "needs_advanced_analysis": False,
            "reason": f"檢測到明確的帖子 ID {thread_id}",
            "theme_keywords": []
        }
    
    keyword_result = await extract_keywords_with_grok(user_query, conversation_context)
    query_keywords = keyword_result["keywords"]
    
    thread_id, thread_title, last_response = await extract_relevant_thread(conversation_context, user_query)
    if thread_id:
        return {
            "direct_response": False,
            "intent": "follow_up",
            "theme": thread_title or "追問相關主題",
            "source_type": source_type,
            "source_ids": [source_id],
            "data_type": "replies",
            "post_limit": 1,
            "filters": {"min_replies": 10, "min_likes": 0, "sort": "popular", "keywords": query_keywords},
            "processing": {"intent": "follow_up", "top_thread_ids": [thread_id]},
            "candidate_thread_ids": [thread_id],
            "top_thread_ids": [thread_id],
            "needs_advanced_analysis": False,
            "reason": f"匹配追問查詢到帖子 ID={thread_id}，標題={thread_title}，關鍵詞：{query_keywords}",
            "theme_keywords": query_keywords
        }
    
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "一般")
    historical_keywords = context_summary.get("keywords", [])
    
    is_vague = len(query_keywords) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"])
    
    is_follow_up = False
    referenced_thread_ids = []
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
    
    if is_follow_up and not thread_id:
        intent = "search_keywords"
        reason = "檢測到追問意圖，但無歷史帖子 ID 匹配，回退到關鍵詞搜索"
        theme = query_keywords[0] if query_keywords else historical_theme
        theme_keywords = query_keywords or historical_keywords
        return {
            "direct_response": False,
            "intent": intent,
            "theme": theme,
            "source_type": source_type,
            "source_ids": [source_id],
            "data_type": "both",
            "post_limit": 2,
            "filters": {"min_replies": 10, "min_likes": 0, "sort": "popular", "keywords": theme_keywords},
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
數據來源：{source_type} ({source_name})
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
歷史主題：{historical_theme}
歷史關鍵詞：{json.dumps(historical_keywords, ensure_ascii=False)}
意圖描述：
{json.dumps({
    "list_titles": "列出帖子標題或清單",
    "summarize_posts": "總結帖子內容或討論",
    "analyze_sentiment": "分析帖子或回覆的情緒",
    "general_query": "與LIHKG或Reddit無關或模糊的問題",
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
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": historical_theme,
            "source_type": source_type,
            "source_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "filters": {},
            "processing": {"intent": "general"},
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "缺少 API 密鑰",
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
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code != 200:
                        logger.debug(f"語義意圖分析失敗：狀態碼={status_code}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"語義意圖分析失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    response_content = data["choices"][0]["message"]["content"]
                    logger.info(
                        f"Grok3 API 調用：函數=analyze_and_screen, "
                        f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                        f"輸出 token={completion_tokens}, 回應={response_content}"
                    )
                    result = json.loads(response_content)
                    intent = result.get("intent", "summarize_posts")
                    confidence = result.get("confidence", 0.7)
                    reason = result.get("reason", "語義匹配")
                    
                    if is_vague and historical_theme != "一般":
                        intent = "summarize_posts"
                        reason = f"問題模糊，延續歷史主題：{historical_theme}"
                    elif is_vague:
                        intent = "summarize_posts"
                        reason = "問題模糊，默認總結帖子"
                    
                    theme = historical_theme if is_vague else "一般"
                    theme_keywords = historical_keywords if is_vague else query_keywords
                    post_limit = 20
                    data_type = "both"
                    processing = {"intent": intent, "top_thread_ids": []}
                    if intent in ["search_keywords", "find_themed"]:
                        theme = query_keywords[0] if query_keywords else historical_theme
                        theme_keywords = query_keywords or historical_keywords
                    elif intent == "follow_up":
                        theme = historical_theme
                        post_limit = min(len(referenced_thread_ids), 2) or 2
                        data_type = "replies"
                        processing["top_thread_ids"] = referenced_thread_ids[:2]
                    elif intent in ["general_query", "introduce"]:
                        data_type = "none"
                    
                    return {
                        "direct_response": intent in ["general_query", "introduce"],
                        "intent": intent,
                        "theme": theme,
                        "source_type": source_type,
                        "source_ids": [source_id],
                        "data_type": "both",
                        "post_limit": post_limit,
                        "filters": {"min_replies": 10, "min_likes": 0, "sort": "popular", "keywords": theme_keywords},
                        "processing": processing,
                        "candidate_thread_ids": [],
                        "top_thread_ids": referenced_thread_ids[:2],
                        "needs_advanced_analysis": confidence < 0.7,
                        "reason": reason,
                        "theme_keywords": theme_keywords
                    }
        except Exception as e:
            logger.debug(f"語義意圖分析錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            logger.error(f"語義意圖分析失敗，嘗試 {max_retries} 次後放棄")
            return {
                "direct_response": False,
                "intent": "summarize_posts",
                "theme": historical_theme,
                "source_type": source_type,
                "source_ids": [source_id],
                "data_type": "both",
                "post_limit": 5,
                "filters": {"min_replies": 10, "min_likes": 0, "keywords": historical_keywords},
                "processing": {"intent": "summarize"},
                "candidate_thread_ids": [],
                "top_thread_ids": [],
                "needs_advanced_analysis": False,
                "reason": f"語義分析失敗，默認使用歷史主題：{historical_theme}",
                "theme_keywords": historical_keywords
            }

async def prioritize_threads_with_grok(user_query, threads, source_name, source_id, source_type="lihkg", intent="summarize_posts"):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {"top_thread_ids": [], "reason": "缺少 API 密鑰"}

    if intent == "follow_up":
        referenced_thread_ids = []
        context = st.session_state.get("conversation_context", [])
        if context:
            last_response = context[-1].get("content", "")
            matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
            referenced_thread_ids = [tid for tid in matches if any(t["thread_id"] == tid for t in threads)]
        if referenced_thread_ids:
            return {"top_thread_ids": referenced_thread_ids[:2], "reason": "使用追問的參考帖子 ID"}

    prompt_builder = PromptBuilder()
    try:
        prompt = prompt_builder.build_prioritize(
            query=user_query,
            source_name=source_name,
            source_id=source_id,
            source_type=source_type,
            threads=[{"thread_id": t["thread_id"], "title": clean_html(t["title"]), "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads]
        )
    except Exception as e:
        logger.error(f"構建優先級提示失敗：{str(e)}")
        return {"top_thread_ids": [], "reason": f"提示構建失敗：{str(e)}"}
    
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
                    status_code = response.status
                    if status_code != 200:
                        logger.debug(f"帖子優先級排序失敗：狀態碼={status_code}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"帖子優先級排序失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    response_content = data["choices"][0]["message"]["content"]
                    logger.info(
                        f"Grok3 API 調用：函數=prioritize_threads_with_grok, "
                        f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                        f"輸出 token={completion_tokens}, 回應={response_content}"
                    )
                    try:
                        result = json.loads(response_content)
                        return result
                    except json.JSONDecodeError:
                        logger.warning(f"無法解析優先級排序結果：{response_content}")
                        return {"top_thread_ids": [], "reason": "無法解析 API 回應"}
        except Exception as e:
            logger.debug(f"帖子優先級排序錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            sorted_threads = sorted(
                threads,
                key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                reverse=True
            )
            return {
                "top_thread_ids": [t["thread_id"] for t in sorted_threads[:20]],
                "reason": "優先級排序失敗，回退到熱門度排序"
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_source, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, source_id=None, source_type="lihkg"):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 10, "min_likes": 0}
    prompt_builder = PromptBuilder()
    
    if not isinstance(processing, dict):
        logger.error(f"無效的處理數據格式：預期 dict，得到 {type(processing)}")
        yield f"錯誤：無效的處理數據格式（{type(processing)}）。請聯繫支持。"
        return
    intent = processing.get('intent', 'summarize')

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        yield "錯誤：缺少 API 密鑰"
        return
    
    intent_word_ranges = {
        "list": (140, 400),
        "summarize": (420, 1000),
        "sentiment": (420, 1000),
        "compare": (560, 1200),
        "introduce": (70, 200),
        "general": (280, 800),
        "themed": (420, 1000),
        "fetch_dates": (280, 800),
        "search_keywords": (420, 1000),
        "recommend_threads": (280, 800),
        "monitor_events": (420, 1000),
        "classify_opinions": (420, 1000),
        "follow_up": (700, 4000),
        "fetch_thread_by_id": (420, 1500)
    }
    
    word_min, word_max = intent_word_ranges.get(intent, (420, 1000))
    min_tokens = int(word_min / 0.8)
    max_tokens = int(word_max / 0.8)
    target_tokens = int((min_tokens + max_tokens) / 2)
    
    total_replies_count = sum(len(data.get("replies", [])) for data in (thread_data if isinstance(thread_data, list) else thread_data.values()))
    
    if total_replies_count:
        complexity_factor = 1.5 if intent in ["follow_up", "fetch_thread_by_id", "summarize", "sentiment", "classify_opinions"] else 1.0
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens) * 0.9 * complexity_factor
    target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)

    max_tokens_limit = 8000
    max_tokens = min(target_tokens + 500, max_tokens_limit)

    max_replies_per_thread = 100
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    
    if intent in ["follow_up", "fetch_thread_by_id"]:
        reply_count_prompt = f"""
你是資料抓取助手，請根據問題和意圖決定每個帖子應下載的回覆數量（100、200、250、500 條）。
僅以 JSON 格式回應，禁止生成自然語言或其他格式的內容。
問題：{user_query}
意圖：{intent}
若問題需要深入分析（如情緒分析、意見分類、追問、特定帖子ID），建議較多回覆（200-500）。
默認：100 條。
輸出格式：{{"replies_per_thread": 100, "reason": "決定原因"}}
"""
        payload = {
            "model": "grok-3-beta",
            "messages": [{"role": "user", "content": reply_count_prompt}],
            "max_tokens": 100,
            "temperature": 0.5
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code == 200:
                        data = await response.json()
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        response_content = data["choices"][0]["message"]["content"]
                        logger.info(
                            f"Grok3 API 調用：函數=stream_grok3_response (reply_count), "
                            f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                            f"輸出 token={completion_tokens}, 回應={response_content}"
                        )
                        result = json.loads(response_content)
                        max_replies_per_thread = min(result.get("replies_per_thread", 100), 500)
                    else:
                        logger.warning(f"無法確定每帖子回覆數，狀態碼={status_code}")
        except Exception as e:
            logger.warning(f"每帖子回覆數選擇失敗：{str(e)}")

    thread_data_dict = {}
    if isinstance(thread_data, list):
        thread_data_dict = {str(data["thread_id"]): data for data in thread_data if isinstance(data, dict) and "thread_id" in data}
    elif isinstance(thread_data, dict):
        thread_data_dict = thread_data
    else:
        logger.error(f"無效的 thread_data 格式：預期 list 或 dict，得到 {type(thread_data)}")
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
        
        if not referenced_thread_ids and intent == "fetch_thread_by_id":
            referenced_thread_ids = processing.get("top_thread_ids", [])
        
        prioritized_thread_data = {tid: thread_data_dict[tid] for tid in map(str, referenced_thread_ids) if tid in thread_data_dict}
        supplemental_thread_data = {tid: data for tid, data in thread_data_dict.items() if tid not in map(str, referenced_thread_ids)}
        thread_data_dict = {**prioritized_thread_data, **supplemental_thread_data}

    filtered_thread_data = {}
    total_replies_count = 0
    
    for tid, data in thread_data_dict.items():
        try:
            replies = data.get("replies", [])
            if not isinstance(replies, list):
                logger.warning(f"無效的回覆格式，帖子 ID={tid}：預期 list，得到 {type(replies)}")
                replies = []
            filtered_replies = []
            for r in replies:
                if not isinstance(r, dict) or not r.get("msg"):
                    continue
                cleaned_msg = clean_html(r["msg"])
                if len(cleaned_msg.strip()) <= 7 or cleaned_msg in ["[圖片]", "[無內容]", "[表情符號]"]:
                    continue
                filtered_replies.append(r)
            
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
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": sorted_replies,
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(sorted_replies)
            }
        except Exception as e:
            logger.error(f"處理帖子 ID={tid} 失敗：{str(e)}")
            yield f"錯誤：處理帖子（ID={tid}）失敗（{str(e)}）。請聯繫支持。"
            return
    
    if total_replies_count < max_replies_per_thread and intent in ["follow_up", "fetch_thread_by_id"]:
        for tid, data in filtered_thread_data.items():
            if data["total_fetched_replies"] < max_replies_per_thread:
                async with request_semaphore:
                    if source_type == "lihkg":
                        content_result = await get_lihkg_thread_content(
                            thread_id=tid,
                            cat_id=source_id,
                            max_replies=max_replies_per_thread - data["total_fetched_replies"],
                            fetch_last_pages=1,
                            start_page=max(data["fetched_pages"], default=0) + 1
                        )
                    else:  # reddit
                        content_result = await get_reddit_thread_content(
                            post_id=tid,
                            subreddit=source_id,
                            max_comments=max_replies_per_thread - data["total_fetched_replies"]
                        )
                if content_result.get("replies"):
                    total_replies = content_result.get("total_replies", data["no_of_reply"])
                    cleaned_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "reply_time": reply.get("reply_time", "0")
                        }
                        for reply in content_result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                    ]
                    filtered_additional_replies = [
                        r for r in cleaned_replies
                        if len(r["msg"].strip()) > 7
                    ]
                    updated_data = {
                        "thread_id": data.get("thread_id", tid),
                        "title": data.get("title", ""),
                        "no_of_reply": total_replies,
                        "last_reply_time": content_result.get("last_reply_time", data["last_reply_time"]),
                        "like_count": data.get("like_count", 0),
                        "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                        "replies": data.get("replies", []) + filtered_additional_replies,
                        "fetched_pages": list(set(data.get("fetched_pages", []) + content_result.get("fetched_pages", []))),
                        "total_fetched_replies": len(data.get("replies", []) + filtered_additional_replies)
                    }
                    filtered_thread_data[tid] = updated_data
                    total_replies_count += len(filtered_additional_replies)
                    async with cache_lock:
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
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
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
        selected_source=selected_source,
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
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": data["replies"][:max_replies_per_thread],
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(data["replies"][:max_replies_per_thread])
            } for tid, data in filtered_thread_data.items()
        }
        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
        prompt = prompt_builder.build_response(
            intent=intent,
            query=user_query,
            selected_source=selected_source,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=list(filtered_thread_data.values()),
            filters=filters
        ) + thread_id_prompt
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens) * 0.9
        target_tokens = min(max(int(target_tokens), min_tokens), max_tokens_limit)
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": prompt_builder.get_system_prompt("response")},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }
    
    response_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                status_code = response.status
                if status_code != 200:
                    logger.error(f"回應生成失敗：狀態碼={status_code}")
                    yield f"錯誤：生成回應失敗（狀態碼 {status_code}）。請稍後重試。"
                    return
                
                async for line in response.content:
                    if line and not line.isspace():
                        line_str = line.decode('utf-8').strip()
                        if line_str == "data: [DONE]":
                            logger.info(
                                f"Grok3 API 調用：函數=stream_grok3_response, "
                                f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                                f"輸出 token={completion_tokens}, 回應長度={len(response_content)}"
                            )
                            break
                        if line_str.startswith("data:"):
                            try:
                                chunk = json.loads(line_str[6:])
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    if "###" in content and ("Content Moderation" in content or "Blocked" in content):
                                        logger.warning(f"檢測到內容審核：{content}")
                                        raise ValueError("檢測到內容審核")
                                    cleaned_content = clean_response(content)
                                    response_content += cleaned_content
                                    yield cleaned_content
                                usage = chunk.get("usage", {})
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                            except json.JSONDecodeError:
                                logger.warning(f"流式數據 JSON 解碼錯誤")
                                continue
        except Exception as e:
            logger.error(f"回應生成失敗：{str(e)}")
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
        finally:
            await session.close()

def clean_cache(max_age=3600):
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]
    # 限制緩存大小
    if len(st.session_state.thread_cache) > MAX_CACHE_SIZE:
        sorted_keys = sorted(
            st.session_state.thread_cache.items(),
            key=lambda x: x[1]["timestamp"]
        )
        for key, _ in sorted_keys[:len(st.session_state.thread_cache) - MAX_CACHE_SIZE]:
            del st.session_state.thread_cache[key]
        logger.info(f"清理緩存，移除 {len(sorted_keys[:len(st.session_state.thread_cache) - MAX_CACHE_SIZE])} 個過舊條目，當前緩存大小：{len(st.session_state.thread_cache)}")

def unix_to_readable(timestamp):
    try:
        timestamp = int(timestamp)
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        logger.warning(f"無法轉換時間戳 {timestamp}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    configure_logger("lihkg_api", "lihkg_api.log")

def configure_reddit_api_logger():
    configure_logger("reddit_api", "reddit_api.log")

async def process_user_question(user_query, selected_source, source_id, source_type="lihkg", analysis=None, request_counter=0, last_reset=0, rate_limit_until=0, conversation_context=None, progress_callback=None):
    if source_type == "lihkg":
        configure_lihkg_api_logger()
    else:
        configure_reddit_api_logger()
    try:
        clean_cache()
        
        if rate_limit_until > time.time():
            logger.warning(f"速率限制生效，直到 {datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
            return {
                "selected_source": selected_source,
                "thread_data": [],
                "rate_limit_info": [{"message": "速率限制生效", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        post_limit = min(analysis.get("post_limit", 5), 20)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 10)
        min_likes = 0
        top_thread_ids = list(set(analysis.get("top_thread_ids", [])))
        intent = analysis.get("intent", "summarize_posts")
        
        logger.info(f"處理用戶問題：intent={intent}, source_type={source_type}, source_id={source_id}, post_limit={post_limit}, top_thread_ids={top_thread_ids}")
        
        keyword_result = await extract_keywords_with_grok(user_query, conversation_context)
        fetch_last_pages = 1 if keyword_result.get("time_sensitive", False) else 0
        
        if intent in ["fetch_thread_by_id", "follow_up"] and top_thread_ids:
            thread_data = []
            rate_limit_info = []
            
            candidate_threads = [{"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0} for tid in top_thread_ids]
            
            tasks = []
            for idx, thread_id in enumerate(top_thread_ids):
                thread_id_str = str(thread_id)
                async with cache_lock:
                    if thread_id_str in st.session_state.thread_cache and st.session_state.thread_cache[thread_id_str]["data"].get("replies"):
                        cached_data = st.session_state.thread_cache[thread_id_str]["data"]
                        thread_data.append(cached_data)
                        logger.debug(f"緩存命中：thread_id={thread_id_str}")
                        continue
                if source_type == "lihkg":
                    tasks.append(get_lihkg_thread_content(
                        thread_id=thread_id_str,
                        cat_id=source_id,
                        max_replies=100,
                        fetch_last_pages=fetch_last_pages,
                        specific_pages=[],
                        start_page=1
                    ))
                else:  # reddit
                    tasks.append(get_reddit_thread_content(
                        post_id=thread_id_str,
                        subreddit=source_id,
                        max_comments=100
                    ))
            
            if tasks:
                content_results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(content_results):
                    if isinstance(result, Exception):
                        logger.warning(f"無法抓取帖子 {candidate_threads[idx]['thread_id']}：{str(result)}")
                        continue
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    
                    thread_id = str(candidate_threads[idx]["thread_id"])
                    if result.get("title"):
                        total_replies = result.get("total_replies", candidate_threads[idx]["no_of_reply"])
                        if total_replies == 0:
                            total_replies = candidate_threads[idx]["no_of_reply"]
                        filtered_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "reply_time": reply.get("reply_time", "0")
                            }
                            for reply in result.get("replies", [])
                            if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                            and len(clean_html(reply.get("msg")).strip()) > 7
                        ]
                        thread_info = {
                            "thread_id": thread_id,
                            "title": result.get("title"),
                            "no_of_reply": total_replies,
                            "last_reply_time": result.get("last_reply_time", "0"),
                            "like_count": result.get("like_count", 0),
                            "dislike_count": result.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": filtered_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(filtered_replies)
                        }
                        thread_data.append(thread_info)
                        async with cache_lock:
                            st.session_state.thread_cache[thread_id] = {
                                "data": thread_info,
                                "timestamp": time.time()
                            }
            
            if len(thread_data) == 1 and intent == "follow_up":
                keyword_result = await extract_keywords_with_grok(user_query, conversation_context)
                theme_keywords = keyword_result["keywords"]
                
                async with request_semaphore:
                    if source_type == "lihkg":
                        supplemental_result = await get_lihkg_topic_list(
                            cat_id=source_id,
                            start_page=1,
                            max_pages=2
                        )
                    else:  # reddit
                        supplemental_result = await get_reddit_topic_list(
                            subreddit=source_id,
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
                    if source_type == "lihkg":
                        supplemental_tasks.append(get_lihkg_thread_content(
                            thread_id=thread_id,
                            cat_id=source_id,
                            max_replies=100,
                            fetch_last_pages=fetch_last_pages,
                            specific_pages=[],
                            start_page=1
                        ))
                    else:  # reddit
                        supplemental_tasks.append(get_reddit_thread_content(
                            post_id=thread_id,
                            subreddit=source_id,
                            max_comments=100
                        ))
                
                if supplemental_tasks:
                    supplemental_results = await asyncio.gather(*supplemental_tasks, return_exceptions=True)
                    for idx, result in enumerate(supplemental_results):
                        if isinstance(result, Exception):
                            logger.warning(f"無法抓取補充帖子 {filtered_supplemental[idx]['thread_id']}：{str(result)}")
                            continue
                        request_counter = result.get("request_counter", request_counter)
                        last_reset = result.get("last_reset", last_reset)
                        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                        rate_limit_info.extend(result.get("rate_limit_info", []))
                        
                        thread_id = str(filtered_supplemental[idx]["thread_id"])
                        if result.get("title"):
                            total_replies = result.get("total_replies", filtered_supplemental[idx].get("no_of_reply", 0))
                            if total_replies == 0:
                                total_replies = filtered_supplemental[idx].get("no_of_reply", 0)
                            filtered_replies = [
                                {
                                    "reply_id": reply.get("reply_id"),
                                    "msg": clean_html(reply.get("msg", "[無內容]")),
                                    "like_count": reply.get("like_count", 0),
                                    "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                    "reply_time": reply.get("reply_time", "0")
                                }
                                for reply in result.get("replies", [])
                                if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                                and len(clean_html(reply.get("msg")).strip()) > 7
                            ]
                            thread_info = {
                                "thread_id": thread_id,
                                "title": result.get("title"),
                                "no_of_reply": total_replies,
                                "last_reply_time": result.get("last_reply_time", "0"),
                                "like_count": filtered_supplemental[idx].get("like_count", 0),
                                "dislike_count": filtered_supplemental[idx].get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "replies": filtered_replies,
                                "fetched_pages": result.get("fetched_pages", []),
                                "total_fetched_replies": len(filtered_replies)
                            }
                            thread_data.append(thread_info)
                            async with cache_lock:
                                st.session_state.thread_cache[thread_id] = {
                                    "data": thread_info,
                                    "timestamp": time.time()
                                }
            
            logger.info(
                f"最終 thread_data：{[{'thread_id': data['thread_id'], 'replies_count': len(data['replies'])} for data in thread_data]}"
            )
            
            return {
                "selected_source": selected_source,
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
            candidate_threads = [
                {"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0}
                for tid in top_thread_ids
            ]
        else:
            initial_threads = []
            for page in range(1, 4):  # 減少抓取頁數至 3
                async with request_semaphore:
                    if source_type == "lihkg":
                        result = await get_lihkg_topic_list(
                            cat_id=source_id,
                            start_page=page,
                            max_pages=1
                        )
                    else:  # reddit
                        result = await get_reddit_topic_list(
                            subreddit=source_id,
                            start_page=page,
                            max_pages=1
                        )
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                items = result.get("items", [])
                initial_threads.extend(items)
                if not items:
                    logger.warning(f"未抓取到分類 ID={source_id}，頁面={page} 的帖子")
                if len(initial_threads) >= 150:
                    initial_threads = initial_threads[:150]
                    break
                if progress_callback:
                    progress_callback(f"已抓取第 {page}/3 頁帖子", 0.1 + 0.2 * (page / 3))
            
            filtered_items = [
                item for item in initial_threads
                if item.get("no_of_reply", 0) >= min_replies
            ]
            
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                async with cache_lock:
                    if thread_id not in st.session_state.thread_cache:
                        cache_data = {
                            "thread_id": thread_id,
                            "title": item["title"],
                            "no_of_reply": item.get("no_of_reply", 0),
                            "last_reply_time": item.get("last_reply_time", "0"),
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": [],
                            "fetched_pages": []
                        }
                        st.session_state.thread_cache[thread_id] = {
                            "data": cache_data,
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
                    prioritization = await prioritize_threads_with_grok(
                        user_query, filtered_items, selected_source, source_id, source_type, intent
                    )
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
        for idx, item in enumerate(candidate_threads):
            thread_id = str(item["thread_id"])
            async with cache_lock:
                if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
                    cached_data = st.session_state.thread_cache[thread_id]["data"]
                    thread_data.append(cached_data)
                    logger.debug(f"緩存命中：thread_id={thread_id}")
                    continue
            if source_type == "lihkg":
                tasks.append((idx, get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=source_id,
                    max_replies=100,
                    fetch_last_pages=fetch_last_pages,
                    specific_pages=[],
                    start_page=1
                )))
            else:  # reddit
                tasks.append((idx, get_reddit_thread_content(
                    post_id=thread_id,
                    subreddit=source_id,
                    max_comments=100
                )))
        
        if tasks:
            content_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            for task_idx, result in enumerate(content_results):
                idx = tasks[task_idx][0]
                if isinstance(result, Exception):
                    logger.warning(f"無法抓取帖子 {candidate_threads[idx]['thread_id']} 內容：{str(result)}")
                    continue
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                
                thread_id = str(candidate_threads[idx]["thread_id"])
                if result.get("title"):
                    total_replies = result.get("total_replies", candidate_threads[idx]["no_of_reply"])
                    if total_replies == 0:
                        total_replies = candidate_threads[idx]["no_of_reply"]
                    filtered_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "reply_time": reply.get("reply_time", "0")
                        }
                        for reply in result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                        and len(clean_html(reply.get("msg")).strip()) > 7
                    ]
                    thread_info = {
                        "thread_id": thread_id,
                        "title": result.get("title"),
                        "no_of_reply": total_replies,
                        "last_reply_time": result.get("last_reply_time", "0"),
                        "like_count": candidate_threads[idx].get("like_count", 0),
                        "dislike_count": candidate_threads[idx].get("dislike_count", 0) if source_type == "lihkg" else 0,
                        "replies": filtered_replies,
                        "fetched_pages": result.get("fetched_pages", []),
                        "total_fetched_replies": len(filtered_replies)
                    }
                    thread_data.append(thread_info)
                    async with cache_lock:
                        st.session_state.thread_cache[thread_id] = {
                            "data": thread_info,
                            "timestamp": time.time()
                        }
        
        if progress_callback:
            progress_callback("正在準備數據", 0.5)
        
        logger.info(
            f"最終 thread_data：{[{'thread_id': data['thread_id'], 'replies_count': len(data['replies'])} for data in thread_data]}"
        )
        
        return {
            "selected_source": selected_source,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(f"處理用戶問題失敗：{str(e)}")
        return {
            "selected_source": selected_source,
            "thread_data": [],
            "rate_limit_info": [{"message": f"處理錯誤：{str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
