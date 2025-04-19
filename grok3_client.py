import streamlit as st
import streamlit.logger
import aiohttp
import asyncio
import json
import re
import random
from typing import AsyncGenerator, Dict, List, Any

logger = streamlit.logger.get_logger(__name__)
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000

async def analyze_question_nature(user_query, cat_name, cat_id, is_advanced=False, metadata=None, thread_data=None, initial_response=None):
    """分析問題性質，決定抓取和處理策略"""
    post_limit = 2
    match = re.search(r'(\d+)個', user_query)
    if match:
        post_limit = min(int(match.group(1)), 10)
    
    prompt = f"""
    你是一個智能助手，任務是分析用戶問題，決定從 LIHKG 討論區抓取的數據和處理方式。以繁體中文回覆，輸出結構化 JSON。

    輸入問題：{user_query}
    用戶選擇的分類：{cat_name}（cat_id={cat_id})
    {'初始數據和回應：' if is_advanced else ''}
    {f'帖子數據：{json.dumps(metadata, ensure_ascii=False)}' if is_advanced and metadata else ''}
    {f'回覆數據：{json.dumps(thread_data, ensure_ascii=False)}' if is_advanced and thread_data else ''}
    {f'初始回應：{initial_response}' if is_advanced and initial_response else ''}

    執行以下步驟：
    1. 識別問題主題（例如，感動、搞笑、財經、時事），並明確標記為 theme。
    2. 判斷意圖（例如，總結、情緒分析、主題聚焦總結）。
    3. {'若為進階分析，檢查初始數據是否涵蓋帖子80%的回覆數量（no_of_reply * 0.8）。若任一帖子未達標，設置 needs_advanced_analysis=True，建議抓取剩餘回覆。' if is_advanced else '根據主題、意圖和分類，執行以下篩選流程：'}
       - 初始抓取：瀏覽 30-90 個帖子標題，根據分類活躍度調整（例如，吹水台 90 個，創意台 30 個）。
       - 候選名單：從 30-90 個標題中，根據主題的語義相關性，選出 10 個候選帖子，優先包含與主題相關的關鍵詞（例如，感動：溫馨、感人、互助；搞笑：幽默、搞亂、on9；財經：股票、投資、經濟）。
       - 關聯性分析：抓取 10 個候選帖子的首頁回覆（每帖 25 條），分析與問題主題的語義相關性，排序並選出關聯性最高的 N 個帖子（N 由 post_limit 指定）。
       - 最終抓取：對於選定的 N 個帖子，抓取首 1 頁（每頁 25 條）和末 2 頁回覆，總計最多 75 條回覆。
    4. {'若為進階分析，設置 needs_advanced_analysis 和 suggestions。若所有帖子回覆數量已達80%，設置 needs_advanced_analysis=False。' if is_advanced else '決定以下參數：'}
       - theme：問題主題（如「感動」、「搞笑」、「財經」）。
       - category_ids：僅包含 cat_id={cat_id}，不得包含其他分類。
       - data_type："title"、"replies"、"both"。
       - post_limit：從問題中提取（如「3個」→ 3），默認 2，最大 10。
       - reply_limit：0-75，初始分析 25 條，最終分析 75 條。
       - filters：根據主題動態設置：
         - 感動：高 like_count（≥ 5），低 dislike_count（< 20），近期帖子，優先溫馨、感人、互助內容。
         - 搞笑：高 like_count（≥ 10），允許高 dislike_count（< 20），優先幽默、誇張、諷刺內容。
         - 財經：高 like_count（≥ 10），低 dislike_count（< 5），優先專業、數據驅動內容。
         - 其他主題：根據語義相關性，設置高互動性（min_replies ≥ 20，min_likes ≥ 10）。
       - processing：根據主題選擇：
         - 感動：emotion_focused_summary。
         - 搞笑：humor_focused_summary。
         - 財經：professional_summary。
         - 其他：summarize 或 sentiment（若涉及情緒分析）。
       - candidate_thread_ids：10 個候選帖子 ID（必須來自實際抓取的帖子）。
       - top_thread_ids：最終選定的 N 個帖子 ID（若為進階分析，優先使用 metadata 中的 thread_ids）。
    5. 若無關 LIHKG，返回空 category_ids。
    6. 提供 category_suggestion 或 reason。

    輸出格式：
    {{
      {"\"needs_advanced_analysis\": false, \"suggestions\": {{ \"theme\": \"\", \"category_ids\": [], \"data_type\": \"\", \"post_limit\": 0, \"reply_limit\": 0, \"filters\": {{}}, \"processing\": \"\", \"candidate_thread_ids\": [], \"top_thread_ids\": [] }}, \"reason\": \"\"" if is_advanced else "\"theme\": \"\", \"category_ids\": [], \"data_type\": \"\", \"post_limit\": 0, \"reply_limit\": 0, \"filters\": {{}}, \"processing\": \"\", \"candidate_thread_ids\": [], \"top_thread_ids\": [], \"category_suggestion\": \"\""}
    }}
    """

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        return {
            "theme": "未知",
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": post_limit,
            "reply_limit": 50,
            "filters": {"min_replies": 20, "min_likes": 10, "recent_only": True},
            "processing": "summarize",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
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
        prompt_summary = (prompt[:200] + "...") if len(prompt) > 200 else prompt
        logger.info(f"調用 Grok 3 API: 端點={GROK3_API_URL}, 提示詞摘要={prompt_summary}, 輸入字元={len(prompt)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=60) as response:
                status_code = response.status
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                if is_advanced:
                    result["suggestions"]["category_ids"] = [cat_id]
                else:
                    result["category_ids"] = [cat_id]
                reason = result.get("reason", "無原因提供")
                logger.info(f"Grok 3 API 回應: 狀態碼={status_code}, 回應摘要={str(result)[:50]}..., reason={reason}")
                # 檢查 candidate_thread_ids
                candidate_ids = result.get("candidate_thread_ids", [])
                if candidate_ids and not all(isinstance(id, (int, str)) and str(id).isdigit() for id in candidate_ids):
                    logger.warning(f"無效 candidate_thread_ids: {candidate_ids}")
                    result["candidate_thread_ids"] = []
                return result
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
        logger.error(f"問題分析失敗: 錯誤={str(e)}, 提示詞摘要={prompt[:200]}...")
        return {
            "theme": "未知",
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": post_limit,
            "reply_limit": 50,
            "filters": {"min_replies": 20, "min_likes": 10, "recent_only": True},
            "processing": "summarize",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "category_suggestion": "分析失敗，使用默認策略。"
        } if not is_advanced else {
            "needs_advanced_analysis": False,
            "suggestions": {"category_ids": [cat_id]},
            "reason": f"分析失敗，錯誤：{str(e)}"
        }

async def screen_thread_titles(
    user_query: str,
    thread_titles: List[Dict[str, Any]],
    post_limit: int
) -> Dict[str, Any]:
    """篩選 LIHKG 帖子標題，選出與用戶問題最相關的帖子"""
    valid_titles = [
        item for item in thread_titles
        if isinstance(item, dict) and
           item.get("thread_id") and
           isinstance(item.get("title"), str) and
           item.get("no_of_reply", 0) >= 0 and
           item.get("like_count", 0) >= 0
    ]
    
    if not valid_titles:
        logger.warning(f"標題篩選失敗: 無有效帖子，問題={user_query}, 原始帖子數={len(thread_titles)}")
        return {
            "top_thread_ids": [],
            "need_replies": False,
            "reason": "無有效帖子，數據結構無效或篩選條件過嚴。"
        }
    
    prompt = f"""
    你是一個智能助手，任務是從 LIHKG 討論區的帖子標題中篩選與用戶問題最相關的帖子。
    以繁體中文回覆，輸出結構化 JSON。

    輸入問題：{user_query}
    帖子標題數據：{json.dumps(valid_titles, ensure_ascii=False)}
    所需帖子數量：{post_limit}

    執行以下步驟：
    1. 分析用戶問題，確定主題（例如，搞笑、感動、財經）。
    2. 根據標題內容和元數據（thread_id, title, no_of_reply, like_count, dislike_count），篩選與問題主題最相關的 {post_limit} 個帖子。
       - 搞笑主題：優先幽默、誇張、諷刺關鍵詞（例如，on9、搞亂、爆笑），高 like_count（≥ 10），允許 dislike_count（< 20）。
       - 感動主題：優先溫馨、感人、互助關鍵詞，高 like_count（≥ 5），低 dislike_count（< 20）。
       - 財經主題：優先股票、投資、經濟關鍵詞，高 like_count（≥ 10），低 dislike_count（< 5）。
       - 其他主題：根據語義相關性和高互動性（no_of_reply ≥ 20，like_count ≥ 10）。
    3. 確保 top_thread_ids 只包含輸入數據中的 thread_id。
    4. 判斷是否需要抓取回覆（need_replies）：
       - 若標題足以回答問題（例如，標題明確且互動性低），設置 need_replies=False。
       - 若需要回覆來驗證內容（例如，標題含關鍵詞但語義模糊），設置 need_replies=True。
    5. 提供篩選理由（reason），說明為何選擇這些帖子。

    輸出格式：
    {{
        "top_thread_ids": [],
        "need_replies": true,
        "reason": ""
    }}

    示例：
    {{
        "top_thread_ids": [12345, 67890],
        "need_replies": true,
        "reason": "選擇了包含搞笑關鍵詞且高互動的帖子"
    }}
    """

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        return {
            "top_thread_ids": [],
            "need_replies": False,
            "reason": "缺少 API 密鑰，無法篩選標題。"
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

    char_count = len(prompt)
    if char_count > GROK3_TOKEN_LIMIT:
        logger.warning(f"標題篩選輸入超限: 字元數={char_count}")
        valid_titles = valid_titles[:60]
        prompt = prompt.replace(
            json.dumps(valid_titles, ensure_ascii=False),
            json.dumps(valid_titles[:60], ensure_ascii=False)
        )
        char_count = len(prompt)
        logger.info(f"截斷後字元數: {char_count}")

    try:
        prompt_summary = (prompt[:200] + "...") if len(prompt) > 200 else prompt
        logger.info(f"調用 Grok 3 API (標題篩選): 端點={GROK3_API_URL}, 提示詞摘要={prompt_summary}, 輸入字元={char_count}")
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=60) as response:
                status_code = response.status
                data = await response.json()
                content = data["choices"][0]["message"]["content"].strip()
                logger.info(f"Grok 3 API 原始回應: 狀態碼={status_code}, 內容={content[:200]}...")
                try:
                    result = json.loads(content)
                    # 驗證 top_thread_ids
                    valid_thread_ids = [str(item["thread_id"]) for item in valid_titles]
                    top_thread_ids = [id for id in result.get("top_thread_ids", []) if str(id) in valid_thread_ids]
                    result["top_thread_ids"] = top_thread_ids
                    if not top_thread_ids and valid_titles:
                        top_thread_ids = [item["thread_id"] for item in random.sample(valid_titles, min(post_limit, len(valid_titles)))]
                        result["top_thread_ids"] = top_thread_ids
                        result["reason"] = f"API 返回無效 ID，隨機選擇帖子：{top_thread_ids}"
                    logger.info(f"Grok 3 API 標題篩選回應: 回應摘要={str(result)[:50]}...")
                    return result
                except json.JSONDecodeError:
                    logger.warning(f"API 回應非有效 JSON: 內容={content[:200]}...")
                    match = re.search(r'"top_thread_ids"\s*:\s*\[\s*([0-9,\s]*)\s*\]', content, re.DOTALL)
                    top_thread_ids = []
                    if match:
                        try:
                            ids = [int(id.strip()) for id in match.group(1).split(",") if id.strip().isdigit()]
                            valid_thread_ids = [str(item["thread_id"]) for item in valid_titles]
                            top_thread_ids = [id for id in ids if str(id) in valid_thread_ids][:post_limit]
                        except ValueError:
                            logger.warning("正則提取 ID 失敗")
                    result = {
                        "top_thread_ids": top_thread_ids,
                        "need_replies": True,
                        "reason": "API 回應格式無效，嘗試提取 ID 或隨機選擇"
                    }
                    if not top_thread_ids and valid_titles:
                        top_thread_ids = [item["thread_id"] for item in random.sample(valid_titles, min(post_limit, len(valid_titles)))]
                        result["top_thread_ids"] = top_thread_ids
                        logger.warning(f"無法提取 ID，隨機選擇帖子: top_thread_ids={top_thread_ids}")
                    return result
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"標題篩選失敗: 錯誤={str(e)}, 提示詞摘要={prompt[:200]}...")
        if valid_titles:
            top_thread_ids = [item["thread_id"] for item in random.sample(valid_titles, min(post_limit, len(valid_titles)))]
            logger.warning(f"標題篩選失敗，隨機選擇帖子: top_thread_ids={top_thread_ids}")
            return {
                "top_thread_ids": top_thread_ids,
                "need_replies": True,
                "reason": f"篩選失敗，隨機選擇帖子：{str(e)}"
            }
        return {
            "top_thread_ids": [],
            "need_replies": False,
            "reason": f"篩選失敗，無可用帖子：{str(e)}"
        }

async def stream_grok3_response(user_query, metadata, thread_data, processing) -> AsyncGenerator[str, None]:
    """以流式方式生成 Grok 3 回應"""
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    filtered_thread_data = {}
    for thread_id, data in thread_data.items():
        replies = [
            r for r in data.get("replies", [])
            if r.get("like_count", 0) != 0 or r.get("dislike_count", 0) != 0
        ][:25]
        filtered_thread_data[thread_id] = {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": replies
        }
    
    # 檢查是否需要進階分析（回覆數量未達80%）
    needs_advanced_analysis = False
    reason = ""
    for thread_id, data in filtered_thread_data.items():
        reply_count = len(data["replies"])
        total_replies = data["no_of_reply"]
        if reply_count < total_replies * 0.8:
            needs_advanced_analysis = True
            reason += f"帖子 {thread_id} 僅抓取 {reply_count}/{total_replies} 條回覆，未達80%。"
    
    if processing == "emotion_focused_summary":
        prompt = f"""
        你是一個智能助手，任務是基於 LIHKG 數據總結與感動或溫馨相關的帖子內容，回答用戶問題。以繁體中文回覆，300-500 字，僅用提供數據。

        使用者問題：{user_query}
        分類：{', '.join([f'{m["thread_id"]} ({m["title"]})' for m in metadata])}
        帖子數據：
        {json.dumps(metadata, ensure_ascii=False)}
        回覆數據：
        {json.dumps(filtered_thread_data, ensure_ascii=False)}

        執行以下步驟：
        1. 解析問題意圖，聚焦感動或溫馨主題。
        2. 綜合每篇帖子的回覆，總結內容，突出感動情緒，優先引用具關注度的回覆（like_count 或 dislike_count 不為 0），確保內容溫馨或感人。
        3. 適配分類語氣（吹水台輕鬆，創意台溫馨）。
        4. 判斷數據是否足夠：
           - 若每篇帖子回覆數量已達80%（no_of_reply * 0.8），說明數據已足夠，無需進階分析。
           - 若任一帖子回覆數量不足80%，建議抓取剩餘回覆。
        5. 若為進階分析，明確標記「已完成進階分析」，避免進一步分析。

        輸出格式：
        - 總結：300-500 字，突出感動或溫馨內容。
        - 進階分析建議：
          - needs_advanced_analysis：{needs_advanced_analysis}。
          - reason：{reason}。
        """
    elif processing == "humor_focused_summary":
        prompt = f"""
        你是一個智能助手，任務是基於 LIHKG 數據總結與幽默或搞笑相關的帖子內容，回答用戶問題。以繁體中文回覆，300-500 字，僅用提供數據。

        使用者問題：{user_query}
        分類：{', '.join([f'{m["thread_id"]} ({m["title"]})' for m in metadata])}
        帖子數據：
        {json.dumps(metadata, ensure_ascii=False)}
        回覆數據：
        {json.dumps(filtered_thread_data, ensure_ascii=False)}

        執行以下步驟：
        1. 解析問題意圖，聚焦幽默或搞笑主題。
        2. 綜合每篇帖子的回覆，總結內容，突出幽默或誇張情緒，優先引用具關注度的回覆（like_count 或 dislike_count 不為 0），確保內容搞笑或諷刺。
        3. 適配分類語氣（吹水台輕鬆，成人台大膽）。
        4. 判斷數據是否足夠：
           - 若每篇帖子回覆數量已達80%（no_of_reply * 0.8），說明數據已足夠，無需進階分析。
           - 若任一帖子回覆數量不足80%，建議抓取剩餘回覆。

        輸出格式：
        - 總結：300-500 字，突出幽默或搞笑內容。
        - 進階分析建議：
          - needs_advanced_analysis：{needs_advanced_analysis}。
          - reason：{reason}。
        """
    elif processing == "professional_summary":
        prompt = f"""
        你是一個智能助手，任務是基於 LIHKG 數據總結與財經、時事等專業主題相關的帖子內容，回答用戶問題。以繁體中文回覆，300-500 字，僅用提供數據。

        使用者問題：{user_query}
        分類：{', '.join([f'{m["thread_id"]} ({m["title"]})' for m in metadata])}
        帖子數據：
        {json.dumps(metadata, ensure_ascii=False)}
        回覆數據：
        {json.dumps(filtered_thread_data, ensure_ascii=False)}

        執行以下步驟：
        1. 解析問題意圖，聚焦財經、時事等專業主題。
        2. 綜合每篇帖子的回覆，總結內容，突出專業觀點或數據驅動討論，優先引用具關注度的回覆（like_count 或 dislike_count 不為 0），確保內容客觀或權威。
        3. 適配分類語氣（財經台專業，時事台嚴肅）。
        4. 判斷數據是否足夠：
           - 若每篇帖子回覆數量已達80%（no_of_reply * 0.8），說明數據已足夠，無需進階分析。
           - 若任一帖子回覆數量不足80%，建議抓取剩餘回覆。

        輸出格式：
        - 總結：300-500 字，突出專業或數據驅動內容。
        - 進階分析建議：
          - needs_advanced_analysis：{needs_advanced_analysis}。
          - reason：{reason}。
        """
    elif processing == "summarize":
        prompt = f"""
        你是一個智能助手，任務是基於 LIHKG 數據總結帖子內容，回答用戶問題。以繁體中文回覆，300-500 字，僅用提供數據。

        使用者問題：{user_query}
        分類：{', '.join([f'{m["thread_id"]} ({m["title"]})' for m in metadata])}
        帖子數據：
        {json.dumps(metadata, ensure_ascii=False)}
        回覆數據：
        {json.dumps(filtered_thread_data, ensure_ascii=False)}

        執行以下步驟：
        1. 解析問題意圖，識別主題（若無明確主題，視為通用）。
        2. 綜合每篇帖子的回覆，總結內容，優先引用具關注度的回覆（like_count 或 dislike_count 不為 0），確保內容與問題主題相關。
        3. 適配分類語氣（吹水台輕鬆，財經台專業）。
        4. 判斷數據是否足夠：
           - 若每篇帖子回覆數量已達80%（no_of_reply * 0.8），說明數據已足夠，無需進階分析。
           - 若任一帖子回覆數量不足80%，建議抓取剩餘回覆。

        輸出格式：
        - 總結：300-500 字，突出與問題主題相關的內容。
        - 進階分析建議：
          - needs_advanced_analysis：{needs_advanced_analysis}。
          - reason：{reason}。
        """
    elif processing == "sentiment":
        prompt = f"""
        你是一個智能助手，任務是基於 LIHKG 數據分析網民情緒，回答用戶問題。以繁體中文回覆，僅用提供數據。

        使用者問題：{user_query}
        分類：{', '.join([f'{m["thread_id"]} ({m["title"]})' for m in metadata])}
        帖子數據：
        {json.dumps(metadata, ensure_ascii=False)}
        回覆數據：
        {json.dumps(filtered_thread_data, ensure_ascii=False)}

        執行以下步驟：
        1. 解析問題意圖。
        2. 分析標題和回覆，判斷情緒（正面、負面、中立），聚焦正負評不為 0 的回覆。
        3. 返回情緒分佈（百分比）。
        4. 判斷數據是否足夠：
           - 若每篇帖子回覆數量已達80%（no_of_reply * 0.8），說明數據已足夠，無需進階分析。
           - 若任一帖子回覆數量不足80%，建議抓取剩餘回覆。

        輸出格式：
        - 情緒分析：
          - 正面：XX%
          - 負面：XX%
          - 中立：XX%
        - 依據：...
        - 進階分析建議：
          - needs_advanced_analysis：{needs_advanced_analysis}。
          - reason：{reason}。
        """
    else:
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
        truncated_thread_data = {}
        for thread_id, data in filtered_thread_data.items():
            replies = data.get("replies", [])[:10]
            truncated_thread_data[thread_id] = {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "replies": replies
            }
        prompt = prompt.replace(
            json.dumps(filtered_thread_data, ensure_ascii=False),
            json.dumps(truncated_thread_data, ensure_ascii=False)
        )
        char_count = len(prompt)
        logger.info(f"截斷後字元數: {char_count}")
    
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
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": True
    }
    
    for attempt in range(3):
        try:
            prompt_summary = (prompt[:200] + "...") if len(prompt) > 200 else prompt
            logger.info(f"調用 Grok 3 API: 端點={GROK3_API_URL}, 提示詞摘要={prompt_summary}, 輸入字元={char_count}")
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=60) as response:
                    status_code = response.status
                    response_content = []
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
                                    response_content.append(content)
                                    yield content
                            except json.JSONDecodeError:
                                logger.warning(f"JSON 解析失敗: 數據={line_str}")
                                continue
                    response_summary = (''.join(response_content)[:50] + "...") if response_content else "無內容"
                    logger.info(f"Grok 3 API 回應: 狀態碼={status_code}, 回應摘要={response_summary}")
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Grok 3 請求失敗，第 {attempt+1} 次重試: 錯誤={str(e)}, 提示詞摘要={prompt[:200]}...")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            yield f"錯誤: 連線失敗，請稍後重試或檢查網路"
            return
