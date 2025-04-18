import streamlit as st
import streamlit.logger
import aiohttp
import asyncio
import json
from typing import AsyncGenerator

logger = streamlit.logger.get_logger(__name__)
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 8000

async def analyze_question_nature(user_query, cat_name, cat_id, is_advanced=False, metadata=None, thread_data=None, initial_response=None):
    """分析問題性質，決定抓取和處理策略"""
    prompt = f"""
    你是一個智能助手，任務是分析用戶問題，決定從 LIHKG 討論區抓取的數據和處理方式。以繁體中文回覆，輸出結構化 JSON。

    輸入問題：{user_query}
    用戶選擇的分類：{cat_name}（cat_id={cat_id})
    {'初始數據和回應：' if is_advanced else ''}
    {f'帖子數據：{json.dumps(metadata, ensure_ascii=False)}' if is_advanced and metadata else ''}
    {f'回覆數據：{json.dumps(thread_data, ensure_ascii=False)}' if is_advanced and thread_data else ''}
    {f'初始回應：{initial_response}' if is_advanced and initial_response else ''}

    執行以下步驟：
    1. 識別問題主題（例如，搞笑、財經）。
    2. 判斷意圖（例如，總結、情緒分析）。
    3. {'若為進階分析，評估初始數據和回應是否充分，建議後續策略。' if is_advanced else '根據主題、意圖和分類，決定：'}
    {'- 是否需要進階分析（needs_advanced_analysis）。' if is_advanced else ''}
    - category_ids：優先 cat_id={cat_id}，可添加其他分類（1=吹水台，2=熱門台，5=時事台，14=上班台，15=財經台，29=成人台，31=創意台）。
    - data_type："title"、"replies"、"both"。
    - post_limit：1-20。
    - reply_limit：0-200。
    - filters：min_replies, min_likes, recent_only。
    - processing：summarize, sentiment, other。
    4. 若無關 LIHKG，返回空 category_ids。
    5. 提供 category_suggestion 或 reason。

    輸出格式：
    {'{'
      '"needs_advanced_analysis": false,
      "suggestions": {
        "category_ids": [],
        "data_type": "",
        "post_limit": 0,
        "reply_limit": 0,
        "filters": {},
        "processing": ""
      },
      "reason": ""
    }' if is_advanced else '{'
      '"category_ids": [],
      "data_type": "",
      "post_limit": 0,
      "reply_limit": 0,
      "filters": {},
      "processing": "",
      "category_suggestion": ""
    }'}
    """

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        return {
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 50,
            "filters": {"min_replies": 50, "min_likes": 10, "recent_only": True},
            "processing": "summarize",
            "category_suggestion": "缺少 API 密鑰，使用默認策略。"
        } if not is_advanced else {
            "needs_advanced_analysis": False,
            "suggestions": {},
            "reason": "缺少 API 密鑰，無需進階分析。"
        }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK3_API_KEY}"
    }
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": "你是 Grok 3，以繁體中文回答，確保回覆清晰、簡潔，僅基於提供數據。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=60) as response:
                response.raise_for_status()
                data = await response.json()
                return json.loads(data["choices"][0]["message"]["content"])
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
        logger.error(f"問題分析失敗: 錯誤={str(e)}")
        return {
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 50,
            "filters": {"min_replies": 50, "min_likes": 10, "recent_only": True},
            "processing": "summarize",
            "category_suggestion": "分析失敗，使用默認策略。"
        } if not is_advanced else {
            "needs_advanced_analysis": False,
            "suggestions": {},
            "reason": "分析失敗，無需進階分析。"
        }

async def stream_grok3_response(user_query, metadata, thread_data, processing) -> AsyncGenerator[str, None]:
    """以流式方式生成 Grok 3 回應"""
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    if processing == "summarize":
        prompt = f"""
        你是一個智能助手，任務是基於 LIHKG 數據總結網民觀點，回答用戶問題。以繁體中文回覆，150-200 字，僅用提供數據。

        使用者問題：{user_query}
        分類：{', '.join([f'{m["thread_id"]} ({m["title"]})' for m in metadata])}
        帖子數據：
        {json.dumps(metadata, ensure_ascii=False)}
        回覆數據：
        {json.dumps(thread_data, ensure_ascii=False)}

        執行以下步驟：
        1. 解析問題意圖。
        2. 總結觀點，適配分類語氣（吹水台輕鬆，財經台專業）。
        3. 引用正評 ≥ 5 的回覆。
        4. 若數據不足，建議進階分析。

        輸出格式：
        - 總結：150-200 字。
        - 進階分析建議：是否需要更多數據或改變處理方式，說明理由。
        """
    elif processing == "sentiment":
        prompt = f"""
        你是一個智能助手，任務是基於 LIHKG 數據分析網民情緒，回答用戶問題。以繁體中文回覆，僅用提供數據。

        使用者問題：{user_query}
        分類：{', '.join([f'{m["thread_id"]} ({m["title"]})' for m in metadata])}
        帖子數據：
        {json.dumps(metadata, ensure_ascii=False)}
        回覆數據：
        {json.dumps(thread_data, ensure_ascii=False)}

        執行以下步驟：
        1. 解析問題意圖。
        2. 分析標題和回覆，判斷情緒（正面、負面、中立），聚焦正負評 ≥ 5。
        3. 返回情緒分佈（百分比）。
        4. 若數據不足，建議進階分析。

        輸出格式：
        - 情緒分析：
          - 正面：XX%
          - 負面：XX%
          - 中立：XX%
        - 依據：...
        - 進階分析建議：...
        """
    else:  # other
        prompt = f"""
        你是一個智能助手，任務是直接回答用戶問題，無需 LIHKG 數據。以繁體中文回覆，50-100 字。

        使用者問題：{user_query}

        執行以下步驟：
        1. 解析問題意圖。
        2. 提供簡潔回應。

        輸出格式：
        - 回應：...
        """
    
    char_count = len(prompt)
    if char_count > GROK3_TOKEN_LIMIT:
        logger.warning(f"輸入超限: 字元數={char_count}")
        prompt = prompt[:GROK3_TOKEN_LIMIT - 100] + "\n[已截斷]"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK3_API_KEY}"
    }
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": "你是 Grok 3，以繁體中文回答，確保回覆清晰、簡潔，僅基於提供數據。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.7,
        "stream": True
    }
    
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=60) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if not line or line.isspace():
                            continue
                        line_str = line.decode('utf-8').strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            data = line_str[6:]
                            try:
                                chunk = json.loads(data)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                logger.warning(f"JSON 解析失敗: 數據={line_str}")
                                continue
            logger.info(f"Grok 3 流式完成: 輸入字元={char_count}")
            return
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Grok 3 請求失敗，第 {attempt+1} 次重試: 錯誤={str(e)}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            yield f"錯誤: 連線失敗，請稍後重試或檢查網路"
            return
