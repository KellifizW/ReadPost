import streamlit as st
import streamlit.logger
import aiohttp
import asyncio
import json
from typing import AsyncGenerator

logger = streamlit.logger.get_logger(__name__)
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 8000

async def stream_grok3_response(text: str, retries: int = 3) -> AsyncGenerator[str, None]:
    """以流式方式從 Grok 3 API 獲取回應"""
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        st.error("未找到 Grok 3 API 密鑰，請檢查配置")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    char_count = len(text)
    if char_count > GROK3_TOKEN_LIMIT:
        logger.warning(f"輸入超限: 字元數={char_count}")
        chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
        logger.info(f"分塊處理: 分塊數={len(chunks)}")
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"使用者問題：{st.session_state.get('last_user_query', '')}\n{chunk}"
            if len(chunk_prompt) > GROK3_TOKEN_LIMIT:
                chunk_prompt = chunk_prompt[:GROK3_TOKEN_LIMIT - 100] + "\n[已截斷]"
                logger.warning(f"分塊超限: 子塊 {i}, 字元數={len(chunk_prompt)}")
            async for chunk in stream_grok3_response(chunk_prompt, retries=retries):
                yield chunk
        return
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK3_API_KEY}"
    }
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": "你是 Grok 3，以繁體中文回答，確保回覆清晰、簡潔，僅基於提供數據。"},
            {"role": "user", "content": text}
        ],
        "max_tokens": 600,
        "temperature": 0.7,
        "stream": True
    }
    
    for attempt in range(retries):
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
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.error(f"Grok 3 異常: 錯誤={str(e)}")
            yield f"錯誤: 連線失敗，請稍後重試或檢查網路"
