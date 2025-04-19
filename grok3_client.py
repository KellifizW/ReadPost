import aiohttp
import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def analyze_question_nature(
    user_query: str,
    cat_name: str,
    cat_id: int,
    is_advanced: bool = False,
    metadata: Dict[str, Any] = None,
    thread_data: Dict[str, Any] = None,
    initial_response: str = None
) -> Dict[str, Any]:
    if metadata is None:
        metadata = {}
    
    prompt = f"""
    你是一個智能助手，任務是分析用戶問題，決定從 LIHKG 討論區抓取的數據和處理方式。
    以繁體中文回覆，輸出結構化 JSON。

    輸入問題：{user_query}
    用戶選擇的分類：{cat_name}（cat_id={cat_id})

    執行以下步驟：
    1. 識別問題主題（例如，感動、搞笑、財經、時事），並明確標記為 theme。
    2. 確定需要抓取的 LIHKG 分類 ID（category_ids），默認包含用戶選擇的 cat_id。
    3. 指定帖子數量（post_limit），根據問題需求推斷。
    4. 設置數據類型（data_type），可選 'titles'（僅標題）或 'both'（標題和回覆）。
    5. 提供篩選條件（filters），包括 min_replies、min_likes、dislike_count_max。
    """

    if is_advanced:
        prompt += f"""
        6. 基於初始數據和回應，判斷是否需要進階分析（needs_advanced_analysis）。
           若需要，設置 needs_advanced_analysis=True，並在 suggestions 中包含 reason 字段，詳細說明為何初次分析不充分（例如，回應質量低、數據多樣性不足、數據量不夠）。
           若不需要，設置 needs_advanced_analysis=False。
        初始數據：{thread_data}
        初始回應：{initial_response}
        """
    else:
        prompt += """
        6. 設置 needs_advanced_analysis=False（初次分析無需進階）。
        """

    prompt += """
    輸出格式：
    ```json
    {
        "theme": "string",
        "category_ids": [int],
        "post_limit": int,
        "data_type": "string",
        "filters": {
            "min_replies": int,
            "min_likes": int,
            "dislike_count_max": int
        },
        "needs_advanced_analysis": bool,
        "suggestions": {
            "reason": "string" (僅當 needs_advanced_analysis=True 時提供),
            ...
        }
    }
    ```
    """

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.x.ai/v1/chat/completions",
            json={"prompt": prompt, "max_tokens": 500}
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"Grok 3 API 回應: 狀態碼={response.status}, 回應摘要={result}")
                return result.get("choices", [{}])[0].get("message", {})
            else:
                logger.error(f"Grok 3 API 錯誤: 狀態碼={response.status}")
                return {}

async def stream_grok3_response(
    user_query: str,
    thread_data: List[Dict[str, Any]],
    prompt_template: str
) -> str:
    prompt = prompt_template.format(
        user_query=user_query,
        thread_data=thread_data
    )
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.x.ai/v1/chat/completions",
            json={"prompt": prompt, "max_tokens": 1000}
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"Grok 3 API 回應: 狀態碼={response.status}, 回應摘要={result}")
                return result.get("choices", [{}])[0].get("message", "")
            else:
                logger.error(f"Grok 3 API 錯誤: 狀態碼={response.status}")
                return "無法生成回應，請稍後再試。"
