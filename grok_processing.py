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
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content, get_category_name
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
        
        logger.info(f"嘗試從 {config_path} 載入 prompts.json")
        
        if not os.path.exists(config_path):
            logger.error(f"prompts.json 未找到: {config_path}")
            raise FileNotFoundError(f"prompts.json 未找到: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.config = json.loads(content)
                logger.info("成功載入 prompts.json")
        except json.JSONDecodeError as e:
            logger.error(f"prompts.json JSON 解析錯誤: {e}")
            raise
        except Exception as e:
            logger.error(f"載入 prompts.json 失敗: {str(e)}")
            raise

    def build_prompt(self, mode, intent=None, query="", cat_name="", cat_id="", conversation_context=None, thread_titles=None, metadata=None, thread_data=None, filters=None, threads=None):
        """
        動態構建提示詞，根據模式（analyze, prioritize, response）和意圖選擇模板。
        """
        if mode == "analyze":
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
            return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        
        elif mode == "prioritize":
            config = self.config.get("prioritize")
            if not config:
                logger.error("未找到 'prioritize' 的提示詞配置")
                raise ValueError("未找到 'prioritize' 的提示詞配置")
            context = config["context"].format(
                query=query,
                cat_name=cat_name,
                cat_id=cat_id
            )
            data = config["data"].format(
                threads=json.dumps( rise=True).format(
                    threads=json.dumps(threads, ensure_ascii=False)
            )
            return f"{config['system']}\n{context}\n{data}\n{config['instructions']}"
        
        elif mode == "response":
            config = self.config["response"]["default"]
            intent_config = config["instructions"].get(intent, config["instructions"]["general"])
            context = config["context"].format(
                query=query,
                selected_cat=cat_name,
                conversation_context=json.dumps(conversation_context or [], ensure_ascii=False)
            )
            data = config["data"].format(
                metadata=json.dumps(metadata or [], ensure_ascii=False),
                thread_data=json.dumps(thread_data or {}, ensure_ascii=False),
                filters=json.dumps(filters, ensure_ascii=False)
            )
            instructions = (
                f"任務：{intent_config['task']}\n"
                f"輸出格式：{intent_config['output_format']}\n"
                f"字數：{intent_config['word_count']}\n"
                f"{config['common']}"
            )
            return f"{config['system']}\n{context}\n{data}\n{instructions}"

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
                logger.info(f"HTML 清理: 替換為 [表情符號], 原始: {original_text}")
            elif any(ext in original_text.lower() for ext in ['.webp', '.jpg', '.png']):
                text = "[圖片]"
                logger.info(f"HTML 清理: 過濾圖片, 原始: {original_text}")
            else:
                logger.info(f"HTML 清理: 清空後無內容, 原始: {original_text}")
                text = "[無內容]"
        return text
    except Exception as e:
        logger.error(f"HTML 清理失敗: {str(e)}, 原始: {original_text}")
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
        logger.info("清理回應: 已移除 post_id 字串")
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
        logger.error(f"Grok 3 API 金鑰缺失: {str(e)}")
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
                    logger.warning(f"對話摘要失敗: 狀態碼={response.status}")
                    return {"theme": "general", "keywords": []}
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                logger.info(f"對話摘要完成: 主題={result['theme']}, 關鍵詞={result['keywords']}")
                return result
    except Exception as e:
        logger.warning(f"對話摘要錯誤: {str(e)}")
        return {"theme": "general", "keywords": []}

async def call_grok_api(payload, max_retries=3):
    """
    通用的 Grok 3 API 調用函數，處理重試和錯誤。
    """
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API 金鑰缺失: {str(e)}")
        return None, f"缺失 API 金鑰: {str(e)}"
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"API 調用失敗: 狀態碼={response.status}, 嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"API 調用失敗: 缺少 choices, 嘗試次數={attempt + 1}")
                        continue
                    return data, None
        except Exception as e:
            logger.warning(f"API 調用錯誤: {str(e)}, 嘗試次數={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None, str(e)
    return None, "超過最大重試次數"

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
            logger.info(f"檢測到追問意圖, 參考帖子 ID: {referenced_thread_ids}, 標題重疊: {title_overlap}, 共同詞: {common_words}")
            if not referenced_thread_ids:
                logger.info("未找到參考帖子 ID，回退到關鍵詞搜索")
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
            "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": theme_keywords},
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
    
    messages = [
        {"role": "system", "content": PromptBuilder().get_system_prompt("analyze")},
        *conversation_context,
        {"role": "user", "content": semantic_prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    logger.info(f"開始語義意圖分析，查詢: {user_query}")
    
    data, error = await call_grok_api(payload)
    if error or not data:
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
            "reason": f"語義分析失敗: {error}, 默認使用歷史主題: {historical_theme}",
            "theme_keywords": historical_keywords
        }
    
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
    theme_keywords = historical_keywords if is_vague else extract_keywords(user_query)
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
        "filters": {"min_replies": 0, "min_likes": min_likes, "sort": "popular", "keywords": theme_keywords},
        "processing": intent,
        "candidate_thread_ids": [],
        "top_thread_ids": referenced_thread_ids,
        "needs_advanced_analysis": confidence < 0.7,
        "reason": reason,
        "theme_keywords": theme_keywords
    }

async def prioritize_threads_with_grok(user_query, threads, cat_name, cat_id, intent="summarize_posts"):
    """
    使用 Grok 3 根據問題語義排序帖子，返回最相關的帖子ID，強化錯誤處理。
    """
    if intent == "follow_up":
        referenced_thread_ids = []
        context = st.session_state.get("conversation_context", [])
        if context:
            last_response = context[-1].get("content", "")
            matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
            referenced_thread_ids = [int(tid) for tid in matches if any(t["thread_id"] == int(tid) for t in threads)]
        if referenced_thread_ids:
            logger.info(f"追問意圖，使用參考帖子 ID: {referenced_thread_ids}")
            return {"top_thread_ids": referenced_thread_ids[:2], "reason": "為追問使用參考帖子 ID"}
        else:
            logger.info("追問未找到參考帖子 ID，繼續進行優先排序")

    prompt_builder = PromptBuilder()
    try:
        prompt = prompt_builder.build_prompt(
            mode="prioritize",
            query=user_query,
            cat_name=cat_name,
            cat_id=cat_id,
            threads=[{"thread_id": t["thread_id"], "title": t["title"], "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads]
        )
    except Exception as e:
        logger.error(f"構建優先排序提示詞失敗: {str(e)}")
        return {"top_thread_ids": [], "reason": f"提示詞構建失敗: {str(e)}"}
    
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    data, error = await call_grok_api(payload)
    if error or not data:
        sorted_threads = sorted(
            threads,
            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
            reverse=True
        )
        top_thread_ids = [t["thread_id"] for t in sorted_threads[:5]]
        return {
            "top_thread_ids": top_thread_ids,
            "reason": f"優先排序失敗: {error}, 回退到流行度排序"
        }
    
    content = data["choices"][0]["message"]["content"]
    logger.info(f"優先排序原始 API 回應: {content}")
    try:
        result = json.loads(content)
        if not isinstance(result, dict) or "top_thread_ids" not in result or "reason" not in result:
            logger.warning(f"無效的優先排序結果格式: {content}")
            return {"top_thread_ids": [], "reason": "無效結果格式: 缺少必要鍵"}
        logger.info(f"帖子優先排序成功: {result}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"無法將優先排序結果解析為 JSON: {content}, 錯誤: {str(e)}")
        return {"top_thread_ids": [], "reason": f"無法解析 API 回應為 JSON: {str(e)}"}

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
        logger.error(f"Grok 3 API 金鑰缺失: {str(e)}")
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
    
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "user", "content": reply_count_prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    max_replies_per_thread = 100
    data, error = await call_grok_api(payload)
    if not error and data:
        result = json.loads(data["choices"][0]["message"]["content"])
        max_replies_per_thread = min(result.get("replies_per_thread", 100), 500)
        logger.info(f"Grok 選擇每帖子回覆數: {max_replies_per_thread}, 原因: {result.get('reason', '默認')}")
        if max_replies_per_thread == 0:
            logger.info(f"由於每帖子回覆數=0，跳過回覆下載，意圖: {processing}")
    else:
        logger.warning(f"每帖子回覆數選擇失敗: {error}, 使用默認 100")
    
    intent = processing.get('intent', 'summarize') if isinstance(processing, dict) else processing
    if intent == "follow_up":
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\d+)\]", conversation_context[-1].get("content", "") if conversation_context else "")
        if not referenced_thread_ids:
            prioritization = await prioritize_threads_with_grok(user_query, metadata, selected_cat, cat_id, intent)
            referenced_thread_ids = prioritization.get("top_thread_ids", [])[:2]
            logger.info(f"上下文中無參考 ID，使用優先排序 ID: {referenced_thread_ids}")
        prioritized_thread_data = {tid: data for tid, data in thread_data.items() if str(tid) in map(str, referenced_thread_ids)}
        supplemental_thread_data = {tid: data for tid, data in thread_data.items() if str(tid) not in map(str, referenced_thread_ids)}
        thread_data = {**prioritized_thread_data, **supplemental_thread_data}
        logger.info(f"為追問過濾 thread_data: 優先={list(prioritized_thread_data.keys())}, 補充={list(supplemental_thread_data.keys())}")

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
            logger.info(f"帖子 ID={tid} 無關鍵詞匹配回覆，使用原始回覆")
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
        logger.info(f"過濾後 thread data 無回覆，由於意圖 {intent} 使用 metadata 進行總結")
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
    max_tokens = 3600  # 放寬到 3600（約 2400 字）
    if total_replies_count == 0:
        target_tokens = min_tokens
    else:
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)
        target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)
    logger.info(f"動態 max_tokens: {target_tokens}, 基於總回覆數: {total_replies_count}")
    
    thread_id_prompt = "\n請在回應中明確包含相關帖子 ID，格式為 [帖子 ID: xxx]。禁止包含 [post_id: ...] 格式。"
    prompt = prompt_builder.build_prompt(
        mode="response",
        intent=intent,
        query=user_query,
        cat_name=selected_cat,
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
        prompt = prompt_builder.build_prompt(
            mode="response",
            intent=intent,
            query=user_query,
            cat_name=selected_cat,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=filtered_thread_data,
            filters=filters
        ) + thread_id_prompt
        target_tokens = min_tokens + (total_replies_count / 500) * (max_tokens - min_tokens)
        target_tokens = min(max(int(target_tokens), min_tokens), max_tokens)
        logger.info(f"截斷提示詞: 原始長度={prompt_length}, 新長度={len(prompt)}, 新 max_tokens: {target_tokens}")
    
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
    
    logger.info(f"開始生成回應，查詢: {user_query}")
    
    response_content = ""
    async with aiohttp.ClientSession() as session:
        try:
            try:
                GROK3_API_KEY = st.secrets["grok3key"]
            except KeyError as e:
                logger.error(f"Grok 3 API 金鑰缺失: {str(e)}")
                yield "錯誤: 缺少 API 密鑰"
                return
            
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
            
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        status_code = response.status
                        if status_code != 200:
                            response_text = await response.text()
                            logger.warning(f"回應生成失敗: 狀態碼={status_code}, 嘗試次數={attempt + 1}")
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
                                                logger.warning(f"檢測到內容審核: {content}")
                                                raise ValueError("檢測到內容審核")
                                            cleaned_content = clean_response(content)
                                            response_content += cleaned_content
                                            yield cleaned_content
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"流塊 JSON 解碼錯誤: {str(e)}")
                                        continue
                        if not response_content:
                            logger.warning(f"未生成內容，嘗試次數={attempt + 1}")
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
                                prompt = prompt_builder.build_prompt(
                                    mode="response",
                                    intent="summarize",
                                    query=user_query,
                                    cat_name=selected_cat,
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
                                fallback_prompt = prompt_builder.build_prompt(
                                    mode="response",
                                    intent="summarize",
                                    query=user_query,
                                    cat_name=selected_cat,
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
                        logger.info(f"回應生成完成: 長度={len(response_content)}")
                        logger.info(f"參考帖子 ID: {re.findall(r'\[帖子 ID: (\d+)\]', response_content)}")
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    logger.warning(f"回應生成錯誤: {str(e)}, 嘗試次數={attempt + 1}")
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
                        prompt = prompt_builder.build_prompt(
                            mode="response",
                            intent=intent,
                            query=user_query,
                            cat_name=selected_cat,
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
            logger.error(f"回應生成失敗: {str(e)}")
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
            return

async def process_user_question(user_query, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, conversation_context, progress_callback):
    """
    處理用戶查詢，整合 LIHKG 數據並更新速率限制。
    """
    intent = analysis.get("intent", "summarize_posts")
    post_limit = analysis.get("post_limit", 5)
    reply_limit = analysis.get("reply_limit", 0)
    filters = analysis.get("filters", {})
    top_thread_ids = analysis.get("top_thread_ids", [])
    data_type = analysis.get("data_type", "both")

    progress_callback("正在獲取帖子列表", 0.3)

    # 獲取帖子列表
    topic_result = await get_lihkg_topic_list(cat_id=cat_id, max_pages=3)
    items = topic_result.get("items", [])
    rate_limit_info = topic_result.get("rate_limit_info", [])
    request_counter = topic_result.get("request_counter", request_counter)
    last_reset = topic_result.get("last_reset", last_reset)
    rate_limit_until = topic_result.get("rate_limit_until", rate_limit_until)

    if not items:
        logger.error(f"無帖子數據: cat_id={cat_id}")
        return {
            "thread_data": [],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }

    # 過濾帖子
    filtered_items = []
    keywords = filters.get("keywords", [])
    min_replies = filters.get("min_replies", 0)
    min_likes = filters.get("min_likes", 0)
    sort = filters.get("sort", "popular")

    for item in items:
        title = item.get("title", "")
        no_of_reply = item.get("no_of_reply", 0)
        like_count = item.get("like_count", 0)
        if no_of_reply >= min_replies and like_count >= min_likes:
            if not keywords or any(kw in title for kw in keywords):
                filtered_items.append(item)

    # 排序帖子
    if sort == "popular":
        filtered_items = sorted(
            filtered_items,
            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
            reverse=True
        )
    elif sort == "recent":
        filtered_items = sorted(
            filtered_items,
            key=lambda x: x.get("last_reply_time", 0),
            reverse=True
        )

    # 優先使用追問的帖子 ID
    if top_thread_ids:
        filtered_items = [
            item for item in filtered_items
            if str(item.get("thread_id")) in top_thread_ids
        ] + [
            item for item in filtered_items
            if str(item.get("thread_id")) not in top_thread_ids
        ]

    selected_items = filtered_items[:post_limit]
    logger.info(f"過濾後帖子數: {len(selected_items)}，原始帖子數: {len(items)}")

    # 直接回應的情況
    if analysis.get("direct_response", False):
        return {
            "thread_data": [
                {
                    "thread_id": item["thread_id"],
                    "title": item["title"],
                    "no_of_reply": item.get("no_of_reply", 0),
                    "last_reply_time": item.get("last_reply_time", 0),
                    "like_count": item.get("like_count", 0),
                    "dislike_count": item.get("dislike_count", 0),
                    "replies": [],
                    "fetched_pages": []
                } for item in selected_items
            ],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until
        }

    # 獲取帖子內容
    thread_data = []
    if data_type in ["both", "replies"]:
        progress_callback("正在獲取帖子內容", 0.5)
        for idx, item in enumerate(selected_items):
            thread_id = item["thread_id"]
            cache_key = f"{cat_id}_{thread_id}_{reply_limit}"
            cached = st.session_state.get("thread_cache", {}).get(cache_key)

            if cached:
                logger.info(f"使用快取數據: thread_id={thread_id}")
                thread_data.append(cached)
            else:
                content_result = await get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=cat_id,
                    max_replies=reply_limit
                )
                replies = [
                    {
                        "msg": clean_html(reply.get("msg", "")),
                        "like_count": reply.get("like_count", 0),
                        "dislike_count": reply.get("dislike_count", 0),
                        "reply_time": reply.get("reply_time", "0")
                    } for reply in content_result.get("replies", [])
                ]
                thread_data.append({
                    "thread_id": thread_id,
                    "title": content_result.get("title", item["title"]),
                    "no_of_reply": content_result.get("total_replies", item.get("no_of_reply", 0)),
                    "last_reply_time": item.get("last_reply_time", 0),
                    "like_count": item.get("like_count", 0),
                    "dislike_count": item.get("dislike_count", 0),
                    "replies": replies,
                    "fetched_pages": content_result.get("fetched_pages", [])
                })
                st.session_state.setdefault("thread_cache", {})[cache_key] = thread_data[-1]
                request_counter = content_result.get("request_counter", request_counter)
                last_reset = content_result.get("last_reset", last_reset)
                rate_limit_until = content_result.get("rate_limit_until", rate_limit_until)
            progress_callback(f"已處理 {idx + 1}/{len(selected_items)} 個帖子", 0.5 + (idx + 1) / len(selected_items) * 0.3)

    # 優先排序帖子
    if selected_items and intent not in ["general_query", "introduce"]:
        progress_callback("正在排序帖子", 0.8)
        prioritization = await prioritize_threads_with_grok(
            user_query=user_query,
            threads=[
                {
                    "thread_id": item["thread_id"],
                    "title": item["title"],
                    "no_of_reply": item.get("no_of_reply", 0),
                    "like_count": item.get("like_count", 0)
                } for item in selected_items
            ],
            cat_name=selected_cat,
            cat_id=cat_id,
            intent=intent
        )
        top_thread_ids = prioritization.get("top_thread_ids", [])
        thread_data = sorted(
            thread_data,
            key=lambda x: top_thread_ids.index(str(x["thread_id"])) if str(x["thread_id"]) in top_thread_ids else len(top_thread_ids)
        )

    return {
        "thread_data": thread_data,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }
