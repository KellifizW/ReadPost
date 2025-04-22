```python
"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
修復輸入驗證過嚴問題，確保廣泛查詢（如「分析吹水台時事主題」）進入分析流程。
主要函數：
- analyze_and_screen：分析問題，識別意圖，放寬語義要求，動態設置篩選條件。
- stream_grok3_response：生成流式回應，動態選擇模板。
- process_user_question：處理用戶問題，抓取帖子並生成總結。
- clean_html：清理 HTML 標籤。
"""

import aiohttp
import asyncio
import json
import re
import random
import time
import logging
import streamlit as st
import os
import hashlib
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

# 檔案處理器：寫入 app.log
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台處理器：輸出到 stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 90  # 秒

class PromptBuilder:
    """
    提示詞生成器，從 prompts.json 載入模板並動態構建提示詞。
    """
    def __init__(self, config_path="/app/prompts.json"):
        if not os.path.exists(config_path):
            logger.error(f"prompts.json not found at: {os.path.abspath(config_path)}")
            self.config = self.get_default_config()
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.config = json.loads(content)
                file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
                logger.info(f"Loaded {config_path} with MD5 hash: {file_hash} from: {os.path.abspath(config_path)}")
                if self.config.get("version") != "f726b665":
                    logger.warning(f"prompts.json version mismatch: expected f726b665, got {self.config.get('version')}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {config_path}: line {e.lineno}, column {e.colno}, message: {e.msg}")
            self.config = self.get_default_config()
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {str(e)}")
            self.config = self.get_default_config()

    def get_default_config(self):
        """
        提供內置模板，支援所有意圖。
        """
        return {
            "version": "f726b665",
            "analyze": {
                "system": "你是 LIHKG 論壇助手，分析用戶問題，識別意圖和主題，設置篩選條件，輸出 JSON。",
                "context": "問題：{query}\n分類：{cat_name}（cat_id={cat_id})\n對話上下文：{conversation_context}",
                "data": "帖子標題：{thread_titles}\n元數據：{metadata}\n帖子數據：{thread_data}",
                "instructions": """
                分析用戶問題，輸出 JSON：
                {
                    "direct_response": bool,
                    "intent": str ("summarize_posts", "list_posts", "sentiment_analysis", "compare_categories", "general_query", "introduce", "find_themed"),
                    "theme": str,
                    "category_ids": array,
                    "data_type": str ("titles", "replies", "both", "none"),
                    "post_limit": int,
                    "reply_limit": int,
                    "filters": object,
                    "processing": str,
                    "candidate_thread_ids": array,
                    "top_thread_ids": array,
                    "needs_advanced_analysis": bool,
                    "reason": str,
                    "theme_keywords": array
                }
                - 若問題要求自我介紹，設置 intent: "introduce"。
                - 若問題提及版塊或主題，設置 intent: "summarize_posts" 或 "find_themed"。
                - 若問題模糊，設置 intent: "summarize_posts"，theme: "general"。
                """
            },
            "response": {
                "list": {
                    "system": "你是 LIHKG 論壇的數據助手，以繁體中文回答，模擬論壇用戶的語氣。",
                    "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
                    "data": "帖子元數據：{metadata}\n篩選條件：{filters}",
                    "instructions": """
                    列出帖子標題，格式為：
                    - 帖子 ID: [thread_id] 標題: [title]
                    若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子。」
                    """
                },
                "summarize": {
                    "system": "你是 LIHKG 論壇的集體意見代表，以繁體中文回答，模擬論壇用戶的語氣。",
                    "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
                    "data": "帖子元數據：{metadata}\n帖子內容：{thread_data}\n篩選條件：{filters}",
                    "instructions": """
                    總結帖子內容，引用高關注回覆，字數400-600字。
                    若無回覆，基於標題推測主題，生成簡化總結（200-300字）。
                    若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子。」
                    """
                },
                "sentiment": {
                    "system": "你是 LIHKG 論壇的集體意見代表，以繁體中文回答，模擬論壇用戶的語氣。",
                    "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
                    "data": "帖子元數據：{metadata}\n帖子內容：{thread_data}\n篩選條件：{filters}",
                    "instructions": """
                    分析帖子情緒（正面、負面、中立），量化比例，字數300-500字。
                    若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子。」
                    """
                },
                "compare": {
                    "system": "你是 LIHKG 論壇助手，比較多個版塊的話題。",
                    "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
                    "data": "帖子元數據：{metadata}\n帖子內容：{thread_data}\n篩選條件：{filters}",
                    "instructions": """
                    比較指定版塊的話題和趨勢，突出差異和共同點，字數400-600字。
                    若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子。」
                    """
                },
                "introduce": {
                    "system": "你是 Grok 3，以繁體中文回答。",
                    "context": "問題：{query}\n對話歷史：{conversation_context}",
                    "data": "",
                    "instructions": """
                    回答：「我是 Grok 3，由 xAI 創建的智能助手，專為解答問題和分析 LIHKG 論壇數據設計。無論係想知吹水台熱話定係深入分析時事，我都可以幫到您！有咩問題，快啲問啦！」（50-100字）。
                    """
                },
                "general": {
                    "system": "你是 Grok 3，以繁體中文回答，模擬 LIHKG 論壇用戶的語氣。",
                    "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
                    "data": "帖子元數據：{metadata}\n篩選條件：{filters}",
                    "instructions": """
                    若問題與 LIHKG 相關，生成簡化總結（200-300字）。
                    若無關，提供上下文回應（200-400字）。
                    若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子。」
                    """
                },
                "themed": {
                    "system": "你是 LIHKG 論壇助手，尋找特定主題的帖子。",
                    "context": "問題：{query}\n分類：{selected_cat}\n對話歷史：{conversation_context}",
                    "data": "帖子元數據：{metadata}\n帖子內容：{thread_data}\n篩選條件：{filters}",
                    "instructions": """
                    根據主題詞（如「時事」「搞笑」），篩選相關帖子，總結內容，引用高關注回覆，字數400-600字。
                    若無帖子，回答：「在 {selected_cat} 中未找到符合條件的帖子。」
                    """
                }
            }
        }

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
        logger.debug(f"Built analyze prompt: length={len(prompt)}, query={query}")
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
            filters=json.dumps(filters or {}, ensure_ascii=False)
        )
        prompt = f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        logger.debug(f"Built response prompt: intent={intent}, length={len(prompt)}, query={query}")
        return prompt

def clean_html(text):
    """
    清理 HTML 標籤，規範化文本。
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}")
        return text

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，識別意圖，放寬語義要求，確保廣泛查詢進入分析流程。
    """
    conversation_context = conversation_context or []
    prompt_builder = PromptBuilder()
    
    # 檢查是否包含版塊或主題相關關鍵詞
    category_keywords = ["吹水台", "時事台", "娛樂台", "科技台"]
    theme_keywords = ["時事", "新聞", "熱話", "政治", "經濟", "社會", "國際", "娛樂", "科技"]
    is_category_related = any(keyword in user_query for keyword in category_keywords)
    is_theme_related = any(keyword in user_query for keyword in theme_keywords)
    
    prompt = prompt_builder.build_analyze(
        query=user_query,
        cat_name=cat_name,
        cat_id=cat_id,
        conversation_context=conversation_context,
        thread_titles=thread_titles,
        metadata=metadata,
        thread_data=thread_data
    )
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": "",
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
            "theme_keywords": []
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {
            "role": "system",
            "content": """
            你是由 xAI 創建的 Grok 3，代表 LIHKG 論壇的集體意見，以繁體中文回答。根據問題語義和提供數據直接回應，無需提及身份或語氣。
            - 若問題提及版塊（如「吹水台」）或主題（如「時事」「新聞」），設置 intent: "summarize_posts"，direct_response: false，進入帖子分析流程。
            - 若問題要求分析主題，設置 theme 和 theme_keywords，篩選條件放寬（min_replies=20, min_likes=5）。
            - 若問題明確要求自我介紹，設置 intent: "introduce"。
            - 若問題模糊或無法解析，設置 intent: "summarize_posts"，選擇版塊熱門帖子進行總結。
            """
        },
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.7
    }
    
    logger.info(
        json.dumps({
            "event": "grok3_api_call",
            "action": "发起問題意圖分析",
            "query": user_query,
            "prompt_length": len(prompt)
        }, ensure_ascii=False)
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    response_text = await response.text()
                    if status_code != 200:
                        logger.warning(
                            json.dumps({
                                "event": "grok3_api_call",
                                "action": "問題意圖分析失敗",
                                "query": user_query,
                                "status": "failed",
                                "status_code": status_code,
                                "response_text": response_text[:500],
                                "attempt": attempt + 1
                            }, ensure_ascii=False)
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {
                            "direct_response": False,
                            "intent": "summarize_posts",
                            "theme": "general",
                            "category_ids": [cat_id],
                            "data_type": "both",
                            "post_limit": 5,
                            "reply_limit": 50,
                            "filters": {"min_replies": 20, "min_likes": 5},
                            "processing": "summarize",
                            "candidate_thread_ids": [],
                            "top_thread_ids": [],
                            "needs_advanced_analysis": False,
                            "reason": f"API request failed with status {status_code}",
                            "theme_keywords": []
                        }
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(
                            json.dumps({
                                "event": "grok3_api_call",
                                "action": "問題意圖分析失敗",
                                "query": user_query,
                                "status": "failed",
                                "status_code": status_code,
                                "error": "Missing 'choices' in response",
                                "attempt": attempt + 1
                            }, ensure_ascii=False)
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {
                            "direct_response": False,
                            "intent": "summarize_posts",
                            "theme": "general",
                            "category_ids": [cat_id],
                            "data_type": "both",
                            "post_limit": 5,
                            "reply_limit": 50,
                            "filters": {"min_replies": 20, "min_likes": 5},
                            "processing": "summarize",
                            "candidate_thread_ids": [],
                            "top_thread_ids": [],
                            "needs_advanced_analysis": False,
                            "reason": "Invalid API response: missing 'choices'",
                            "theme_keywords": []
                        }
                    
                    result = json.loads(data["choices"][0]["message"]["content"])
                    logger.debug(f"Raw intent analysis response: {result}")
                    
                    # 放寬意圖識別：若提及版塊或主題，強制進入分析流程
                    if is_category_related or is_theme_related:
                        result["direct_response"] = False
                        result["intent"] = "summarize_posts"
                        result["theme"] = "時事" if is_theme_related else "general"
                        result["filters"] = {"min_replies": 20, "min_likes": 5}
                        result["theme_keywords"] = theme_keywords if is_theme_related else []
                        result["needs_advanced_analysis"] = is_theme_related
                        result["post_limit"] = 10
                        result["reply_limit"] = 50
                        result["category_ids"] = [cat_id]
                        result["data_type"] = "both"
                        result["processing"] = "summarize"
                    
                    result.setdefault("direct_response", False)
                    result.setdefault("intent", "summarize_posts")
                    result.setdefault("theme", "general")
                    result.setdefault("category_ids", [cat_id])
                    result.setdefault("data_type", "both")
                    result.setdefault("post_limit", 5)
                    result.setdefault("reply_limit", 50)
                    result.setdefault("filters", {"min_replies": 20, "min_likes": 5})
                    result.setdefault("processing", "summarize")
                    result.setdefault("candidate_thread_ids", [])
                    result.setdefault("top_thread_ids", [])
                    result.setdefault("needs_advanced_analysis", False)
                    result.setdefault("reason", "")
                    result.setdefault("theme_keywords", [])
                    logger.info(
                        json.dumps({
                            "event": "grok3_api_call",
                            "action": "完成問題意圖分析",
                            "query": user_query,
                            "status": "success",
                            "status_code": status_code,
                            "intent": result["intent"],
                            "theme": result["theme"],
                            "needs_advanced_analysis": result["needs_advanced_analysis"],
                            "filters": result["filters"],
                            "top_thread_ids": result["top_thread_ids"],
                            "theme_keywords": result["theme_keywords"]
                        }, ensure_ascii=False)
                    )
                    return result
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            status_code = getattr(e, 'status', None) or "unknown"
            logger.warning(
                json.dumps({
                    "event": "grok3_api_call",
                    "action": "問題意圖分析失敗",
                    "query": user_query,
                    "status": "failed",
                    "status_code": status_code,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "attempt": attempt + 1
                }, ensure_ascii=False)
            )
            if attempt < max_retries - 1:
                simplified_prompt = prompt_builder.build_analyze(
                    query=user_query,
                    cat_name=cat_name,
                    cat_id=cat_id,
                    conversation_context=conversation_context
                )
                messages[-1]["content"] = simplified_prompt
                payload["max_tokens"] = 300
                logger.info(
                    json.dumps({
                        "event": "grok3_api_call",
                        "action": "重試問題意圖分析（簡化提示詞）",
                        "query": user_query,
                        "prompt_length": len(simplified_prompt)
                    }, ensure_ascii=False)
                )
                await asyncio.sleep(2)
                continue
            return {
                "direct_response": False,
                "intent": "summarize_posts",
                "theme": "general",
                "category_ids": [cat_id],
                "data_type": "both",
                "post_limit": 5,
                "reply_limit": 50,
                "filters": {"min_replies": 20, "min_likes": 5},
                "processing": "summarize",
                "candidate_thread_ids": [],
                "top_thread_ids": [],
                "needs_advanced_analysis": False,
                "reason": f"Analysis failed after {max_retries} attempts: {str(e)}",
                "theme_keywords": []
            }

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id):
    """
    使用 Grok 3 根據問題語義排序帖子，返回最相關的帖子ID。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        return {"top_thread_ids": [], "reason": "Missing API key"}

    prompt = f"""
    你是LIHKG論壇助手，根據用戶問題和帖子數據，選擇最相關的帖子並排序，輸出JSON：
    {{
        "top_thread_ids": {{array}},
        "reason": {{string}}
    }}
    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    帖子：{json.dumps([{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads], ensure_ascii=False)}
    任務：
    1. 根據問題語義，選擇與主題最相關的帖子（最多10個）。
    2. 若問題包含「時事」「新聞」「熱話」，優先選擇標題含政治、經濟、社會等詞的帖子。
    3. 考慮標題內容、回覆數和點讚數，動態排序。
    4. 說明選擇理由。
    示例：
    - 問題："時事帖子" -> {{"top_thread_ids": ["123", "456"], "reason": "標題包含時事相關詞，點讚數高"}}
    """
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
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {"top_thread_ids": [], "reason": f"API request failed with status {response.status}"}
                    
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"Thread prioritization failed: missing choices, attempt={attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return {"top_thread_ids": [], "reason": "Invalid API response: missing choices"}
                    
                    result = json.loads(data["choices"][0]["message"]["content"])
                    logger.info(f"Thread prioritization succeeded: {result}")
                    return result
        except Exception as e:
            logger.warning(f"Thread prioritization error: {str(e)}, attempt={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return {"top_thread_ids": [], "reason": f"Prioritization failed: {str(e)}"}

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_cat, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None):
    """
    使用 Grok 3 API 生成流式回應，根據意圖和分類動態選擇模板。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 20, "min_likes": 5}
    prompt_builder = PromptBuilder()
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    # 動態調整回覆數量
    max_replies_per_thread = 20
    filtered_thread_data = {}
    for tid, data in thread_data.items():
        replies = data.get("replies", [])
        logger.info(
            json.dumps({
                "event": "raw_thread_data",
                "thread_id": tid,
                "raw_reply_count": len(replies),
                "sample_reply": replies[0] if replies else None
            }, ensure_ascii=False)
        )
        
        sorted_replies = sorted(
            [r for r in replies if r.get("msg")],
            key=lambda x: x.get("like_count", 0),
            reverse=True
        )[:max_replies_per_thread]
        
        if not sorted_replies and replies:
            logger.warning(f"No valid replies for thread_id={tid}, using raw replies")
            sorted_replies = replies[:max_replies_per_thread]
        
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
        
        like_counts = [r.get("like_count", 0) for r in sorted_replies]
        logger.info(
            json.dumps({
                "event": "filtered_thread_data",
                "thread_id": tid,
                "reply_count": len(sorted_replies),
                "like_counts_summary": {
                    "min": min(like_counts) if like_counts else 0,
                    "max": max(like_counts) if like_counts else 0,
                    "avg": sum(like_counts) / len(like_counts) if like_counts else 0
                }
            }, ensure_ascii=False)
        )
    
    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata:
        logger.warning(
            json.dumps({
                "event": "data_validation",
                "query": user_query,
                "reason": "Filtered thread data has no replies, using metadata for summary"
            }, ensure_ascii=False)
        )
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
    
    intent = processing.get('intent', 'summarize') if isinstance(processing, dict) else processing
    logger.debug(f"Selected intent: {intent}")
    
    prompt = prompt_builder.build_response(
        intent=intent,
        query=user_query,
        selected_cat=selected_cat,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=filtered_thread_data,
        filters=filters
    )
    
    prompt_length = len(prompt)
    if prompt_length > GROK3_TOKEN_LIMIT:
        max_replies_per_thread = max_replies_per_thread // 2
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
        prompt = prompt_builder.build_response(
            intent=intent,
            query=user_query,
            selected_cat=selected_cat,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=filtered_thread_data,
            filters=filters
        )
        logger.info(f"Truncated prompt: original_length={prompt_length}, new_length={len(prompt)}")
    
    if prompt_length < 500 and intent in ["summarize", "sentiment"]:
        logger.warning(
            json.dumps({
                "event": "prompt_validation",
                "query": user_query,
                "prompt_length": prompt_length,
                "reason": "Prompt too short, retrying with simplified data"
            }, ensure_ascii=False)
        )
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
        )
        logger.info(
            json.dumps({
                "event": "prompt_fallback",
                "query": user_query,
                "reason": "Using simplified prompt with reduced data",
                "new_prompt_length": len(prompt)
            }, ensure_ascii=False)
        )
    
    logger.debug(
        json.dumps({
            "event": "prompt_content",
            "query": user_query,
            "intent": intent,
            "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt
        }, ensure_ascii=False)
    )
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": "你是由 xAI 創建的 Grok 3，代表 LIHKG 論壇的集體意見，以繁體中文回答。根據問題語義和提供數據直接回應，無需提及身份或語氣。"},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": True
    }
    
    logger.info(
        json.dumps({
            "event": "grok3_api_call",
            "action": "发起回應生成",
            "query": user_query,
            "prompt_length": len(prompt)
        }, ensure_ascii=False)
    )
    
    response_content = ""
    async with aiohttp.ClientSession() as session:
        try:
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        status_code = response.status
                        if status_code != 200:
                            response_text = await response.text()
                            logger.warning(
                                json.dumps({
                                    "event": "grok3_api_call",
                                    "action": "回應生成失敗",
                                    "query": user_query,
                                    "status": "failed",
                                    "status_code": status_code,
                                    "response_text": response_text[:500],
                                    "attempt": attempt + 1
                                }, ensure_ascii=False)
                            )
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
                                            response_content += content
                                            yield content
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON decode error in stream chunk: {str(e)}")
                                        continue
                        if not response_content:
                            logger.warning(
                                json.dumps({
                                    "event": "grok3_api_call",
                                    "action": "回應生成失敗",
                                    "query": user_query,
                                    "reason": "No content generated",
                                    "attempt": attempt + 1
                                }, ensure_ascii=False)
                            )
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
                                )
                                payload["messages"][-1]["content"] = prompt
                                logger.info(
                                    json.dumps({
                                        "event": "grok3_api_call",
                                        "action": "重試回應生成（簡化提示詞）",
                                        "query": user_query,
                                        "prompt_length": len(prompt)
                                    }, ensure_ascii=False)
                                )
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
                                )
                                payload["messages"][-1]["content"] = fallback_prompt
                                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as fallback_response:
                                    if fallback_response.status == 200:
                                        data = await fallback_response.json()
                                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                        if content:
                                            response_content = content
                                            yield content
                                            logger.info(
                                                json.dumps({
                                                    "event": "grok3_api_call",
                                                    "action": "回退回應生成成功",
                                                    "query": user_query,
                                                    "response_length": len(content)
                                                }, ensure_ascii=False)
                                            )
                                            return
                            response_content = "無法生成詳細總結，可能是數據不足。以下是吹水台的通用概述：吹水台討論涵蓋時事、娛樂等多主題，網民觀點多元。"
                            yield response_content
                            logger.info(
                                json.dumps({
                                    "event": "grok3_api_call",
                                    "action": "完成回應生成",
                                    "query": user_query,
                                    "status": "success",
                                    "status_code": status_code,
                                    "response_length": len(response_content)
                                }, ensure_ascii=False)
                            )
                            return
                        logger.info(
                            json.dumps({
                                "event": "grok3_api_call",
                                "action": "完成回應生成",
                                "query": user_query,
                                "status": "success",
                                "status_code": status_code,
                                "response_length": len(response_content)
                            }, ensure_ascii=False)
                        )
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    status_code = getattr(e, 'status', None) or "unknown"
                    logger.warning(
                        json.dumps({
                            "event": "grok3_api_call",
                            "action": "回應生成失敗",
                            "query": user_query,
                            "status": "failed",
                            "status_code": status_code,
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "attempt": attempt + 1
                        }, ensure_ascii=False)
                    )
                    if attempt < 2:
                        max_replies_per_thread = max_replies_per_thread // 2
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
                        prompt = prompt_builder.build_response(
                            intent=intent,
                            query=user_query,
                            selected_cat=selected_cat,
                            conversation_context=conversation_context,
                            metadata=metadata,
                            thread_data=filtered_thread_data,
                            filters=filters
                        )
                        payload["messages"][-1]["content"] = prompt
                        logger.info(
                            json.dumps({
                                "event": "grok3_api_call",
                                "action": "重試回應生成（減少回覆數據）",
                                "query": user_query,
                                "prompt_length": len(prompt)
                            }, ensure_ascii=False)
                        )
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試。"
                    return
        except Exception as e:
            logger.error(
                json.dumps({
                    "event": "grok3_api_call",
                    "action": "回應生成異常",
                    "query": user_query,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error": str(e)
                }, ensure_ascii=False)
            )
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
    logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None, progress_callback=None):
    """
    處理用戶問題，分階段抓取並分析 LIHKG 帖子。
    """
    try:
        logger.info(
            json.dumps({
                "event": "process_user_question",
                "query": user_question,
                "category": selected_cat,
                "cat_id": cat_id,
                "intent": analysis.get("intent", "unknown"),
                "theme": analysis.get("theme", "")
            }, ensure_ascii=False)
        )
        
        # 清理緩存
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
        reply_limit = analysis.get("reply_limit", 50)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 20)
        min_likes = filters.get("min_likes", 5)
        sort_method = filters.get("sort", "popular")
        time_range = filters.get("time_range", "recent")
        top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
        previous_thread_ids = previous_thread_ids or []
        
        thread_data = []
        rate_limit_info = []
        initial_threads = []
        
        # 抓取帖子列表
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
            initial_threads.extend(items)
            logger.info(
                json.dumps({
                    "event": "thread_fetch",
                    "cat_id": cat_id,
                    "page": page,
                    "items_fetched": len(items)
                }, ensure_ascii=False)
            )
            if not items:
                logger.warning(f"No threads fetched for cat_id={cat_id}, page={page}")
            if len(initial_threads) >= 150:
                initial_threads = initial_threads[:150]
                break
            if progress_callback:
                progress_callback(f"已抓取第 {page}/5 頁帖子", 0.1 + 0.2 * (page / 5))
        
        if progress_callback:
            progress_callback("正在篩選帖子", 0.3)
        
        # 篩選帖子
        filtered_items = []
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            no_of_reply = item.get("no_of_reply", 0)
            like_count = int(item.get("like_count", 0))
            reasons = []
            
            if no_of_reply < min_replies:
                reasons.append(f"no_of_reply={no_of_reply} < min_replies={min_replies}")
            if like_count < min_likes:
                reasons.append(f"like_count={like_count} < min_likes={min_likes}")
            if thread_id in previous_thread_ids:
                reasons.append("thread_id in previous_thread_ids")
            
            if not reasons:
                filtered_items.append(item)
            else:
                logger.info(
                    json.dumps({
                        "event": "thread_filtering",
                        "thread_id": thread_id,
                        "status": "excluded",
                        "reason": "; ".join(reasons)
                    }, ensure_ascii=False)
                )
        
        logger.info(
            json.dumps({
                "event": "thread_filtering_summary",
                "initial_threads": len(initial_threads),
                "filtered_items": len(filtered_items),
                "filters": filters,
                "excluded_thread_ids": previous_thread_ids
            }, ensure_ascii=False)
        )
        
        # 若 filtered_items 為空，放寬篩選條件
        if not filtered_items and initial_threads:
            logger.warning("No filtered items, relaxing filters")
            min_replies = 10
            min_likes = 2
            filtered_items = [
                item for item in initial_threads
                if item.get("no_of_reply", 0) >= min_replies and int(item.get("like_count", 0)) >= min_likes
                and str(item["thread_id"]) not in previous_thread_ids
            ]
            logger.info(
                json.dumps({
                    "event": "thread_filtering_relaxed",
                    "filtered_items": len(filtered_items),
                    "new_filters": {"min_replies": min_replies, "min_likes": min_likes}
                }, ensure_ascii=False)
            )
        
        # 更新緩存
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            if thread_id not in st.session_state.thread_cache:
                st.session_state.thread_cache[thread_id] = {
                    "data": {
                        "thread_id": thread_id,
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item.get("last_reply_time", 0),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [],
                        "fetched_pages": []
                    },
                    "timestamp": time.time()
                }
        
        # 若無 top_thread_ids，嘗試用 Grok 3 排序
        if not top_thread_ids and filtered_items:
            if progress_callback:
                progress_callback("正在重新分析帖子選擇", 0.4)
            prioritization = await prioritize_threads_with_grok(user_question, filtered_items, selected_cat, cat_id)
            top_thread_ids = prioritization["top_thread_ids"]
            logger.info(f"Grok prioritized threads: {top_thread_ids}, reason: {prioritization['reason']}")
        
        # 若仍無 top_thread_ids，按得分排序
        if not top_thread_ids and filtered_items:
            if sort_method == "popular":
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                    reverse=True
                )
            else:
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: x.get("last_reply_time", "0"),
                    reverse=(time_range == "recent")
                )
            top_thread_ids = [item["thread_id"] for item in sorted_items[:post_limit]]
            logger.info(f"Generated top_thread_ids based on {sort_method}: {top_thread_ids}")
        
        # 候選帖子抓取
        if progress_callback:
            progress_callback("正在抓取候選帖子內容", 0.5)
        
        candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
        if not candidate_threads and filtered_items:
            candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items)))
            logger.info(f"No candidate threads, using random: {len(candidate_threads)} threads selected")
        
        logger.info(
            json.dumps({
                "event": "candidate_threads",
                "query": user_question,
                "candidate_count": len(candidate_threads),
                "candidate_thread_ids": [item["thread_id"] for item in candidate_threads]
            }, ensure_ascii=False)
        )
        
        # 抓取帖子內容
        for idx, item in enumerate(candidate_threads):
            thread_id = str(item["thread_id"])
            cache_key = thread_id
            cache_data = st.session_state.thread_cache.get(cache_key, {}).get("data", {})
            
            if cache_data and cache_data.get("replies") and cache_data.get("fetched_pages"):
                thread_data.append(cache_data)
                logger.info(
                    json.dumps({
                        "event": "cache_hit",
                        "thread_id": thread_id,
                        "reply_count": len(cache_data.get("replies", [])),
                        "fetched_pages": cache_data.get("fetched_pages", [])
                    }, ensure_ascii=False)
                )
                continue
            
            if progress_callback:
                progress_callback(f"正在抓取帖子 {idx + 1}/{len(candidate_threads)}", 0.5 + 0.3 * ((idx + 1) / len(candidate_threads)))
            
            content_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit,
                fetch_last_pages=0,
                specific_pages=None,
                start_page=1
            )
            
            request_counter = content_result.get("request_counter", request_counter)
            last_reset = content_result.get("last_reset", last_reset)
            rate_limit_until = content_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(content_result.get("rate_limit_info", []))
            
            if content_result.get("replies"):
                thread_info = {
                    "thread_id": thread_id,
                    "title": content_result.get("title", item["title"]),
                    "no_of_reply": item.get("no_of_reply", content_result.get("total_replies", 0)),
                    "last_reply_time": item.get("last_reply_time", "0"),
                    "like_count": item.get("like_count", 0),
                    "dislike_count": item.get("dislike_count", 0),
                    "replies": [
                        {
                            "post_id": reply.get("post_id"),
                            "msg": clean_html(reply.get("msg", "")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": reply.get("reply_time", "0")
                        } for reply in content_result["replies"] if reply.get("msg")
                    ],
                    "fetched_pages": content_result.get("fetched_pages", [])
                }
                thread_data.append(thread_info)
                
                # 更新緩存
                st.session_state.thread_cache[cache_key] = {
                    "data": thread_info,
                    "timestamp": time.time()
                }
                
                logger.info(
                    json.dumps({
                        "event": "thread_content_fetched",
                        "thread_id": thread_id,
                        "title": thread_info["title"],
                        "reply_count": len(thread_info["replies"]),
                        "fetched_pages": thread_info["fetched_pages"]
                    }, ensure_ascii=False)
                )
            else:
                logger.warning(
                    json.dumps({
                        "event": "thread_content_fetch_failed",
                        "thread_id": thread_id,
                        "reason": "No replies fetched"
                    }, ensure_ascii=False)
                )
        
        # 進階分析（如果需要）
        if analysis.get("needs_advanced_analysis", False) and thread_data:
            if progress_callback:
                progress_callback("正在進行進階分析", 0.8)
            
            advanced_analysis = await analyze_and_screen(
                user_query=user_question,
                cat_name=selected_cat,
                cat_id=cat_id,
                thread_titles=[item["title"] for item in thread_data],
                metadata=[{
                    "thread_id": item["thread_id"],
                    "title": item["title"],
                    "no_of_reply": item["no_of_reply"],
                    "like_count": item["like_count"],
                    "dislike_count": item["dislike_count"]
                } for item in thread_data],
                thread_data={item["thread_id"]: item for item in thread_data},
                is_advanced=True,
                conversation_context=conversation_context
            )
            
            logger.info(
                json.dumps({
                    "event": "advanced_analysis",
                    "query": user_question,
                    "new_intent": advanced_analysis.get("intent", "unknown"),
                    "new_top_thread_ids": advanced_analysis.get("top_thread_ids", [])
                }, ensure_ascii=False)
            )
            
            thread_data = [
                item for item in thread_data
                if str(item["thread_id"]) in map(str, advanced_analysis.get("top_thread_ids", []))
            ]
            analysis.update(advanced_analysis)
        
        # 若無 thread_data，從緩存恢復
        if not thread_data and st.session_state.thread_cache:
            logger.warning("No thread data, attempting cache recovery")
            thread_data = [
                cache["data"] for cache in st.session_state.thread_cache.values()
                if cache["data"].get("replies")
            ][:post_limit]
            logger.info(f"Recovered {len(thread_data)} threads from cache")
        
        # 最終結果
        if progress_callback:
            progress_callback("完成數據處理", 0.9)
        
        logger.info(
            json.dumps({
                "event": "process_user_question_completed",
                "query": user_question,
                "thread_data_count": len(thread_data),
                "rate_limit_info_count": len(rate_limit_info)
            }, ensure_ascii=False)
        )
        
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
        logger.error(
            json.dumps({
                "event": "process_user_question_failed",
                "query": user_question,
                "error_type": type(e).__name__,
                "error": str(e)
            }, ensure_ascii=False)
        )
        # 回退：嘗試從緩存中恢復部分數據
        fallback_thread_data = [
            cache["data"] for cache in st.session_state.thread_cache.values()
            if cache["data"].get("replies") and cache["data"]["thread_id"] in top_thread_ids
        ]
        return {
            "selected_cat": selected_cat,
            "thread_data": fallback_thread_data,
            "rate_limit_info": rate_limit_info + [{"message": f"Processing failed: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
```
