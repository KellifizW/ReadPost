import aiohttp
import asyncio
import time
from datetime import datetime
import random
import uuid
import hashlib
import streamlit.logger

# 使用 Streamlit logger
logger = streamlit.logger.get_logger(__name__)

LIHKG_BASE_URL = "https://lihkg.com"

# 隨機化的 User-Agent 列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]

# 全局速率限制管理器
class RateLimiter:
    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests  # 每分鐘最大請求數
        self.period = period  # 時間週期（秒）
        self.requests = []

    async def acquire(self, context: dict = None):
        now = time.time()
        # 移除過期的請求
        self.requests = [t for t in self.requests if now - t < self.period]
        if len(self.requests) >= self.max_requests:
            wait_time = self.period - (now - self.requests[0])
            context_info = f", 上下文={context}" if context else ""
            logger.warning(f"達到內部速率限制，當前請求數={len(self.requests)}/{self.max_requests}，等待 {wait_time:.2f} 秒{context_info}")
            await asyncio.sleep(wait_time)
            self.requests = self.requests[1:]  # 移除最早的請求
        self.requests.append(now)

# 初始化速率限制器（每分鐘 20 次）
rate_limiter = RateLimiter(max_requests=20, period=60)

async def get_lihkg_topic_list(cat_id, sub_cat_id, start_page, max_pages, request_counter, last_reset, rate_limit_until):
    # 生成隨機設備 ID
    device_id = hashlib.sha1(str(uuid.uuid4()).encode()).hexdigest()

    # 隨機選擇 User-Agent
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "X-LI-DEVICE": device_id,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"{LIHKG_BASE_URL}/category/{cat_id}",
        "DNT": "1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    
    items = []
    rate_limit_info = []
    max_retries = 3  # 最大重試次數
    
    # 檢查速率限制
    current_time = time.time()
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - API 速率限制中，請在 {datetime.fromtimestamp(rate_limit_until)} 後重試")
        logger.warning(f"API 速率限制中，需等待至 {datetime.fromtimestamp(rate_limit_until)}")
        return {"items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until}
    
    async with aiohttp.ClientSession() as session:
        for page in range(start_page, start_page + max_pages):
            # 重置請求計數
            if current_time - last_reset >= 60:
                request_counter = 0
                last_reset = current_time
            
            url = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id={cat_id}&page={page}&count=60&type=now&order=now"
            
            # 記錄抓取條件
            fetch_conditions = {
                "cat_id": cat_id,
                "sub_cat_id": sub_cat_id,
                "page": page,
                "max_pages": max_pages,
                "user_agent": headers["User-Agent"],
                "device_id": device_id,
                "request_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "request_counter": request_counter,
                "last_reset": datetime.fromtimestamp(last_reset).strftime("%Y-%m-%d %H:%M:%S"),
                "rate_limit_until": datetime.fromtimestamp(rate_limit_until).strftime("%Y-%m-%d %H:%M:%S") if rate_limit_until > time.time() else "無"
            }
            
            for attempt in range(max_retries):
                try:
                    await rate_limiter.acquire(context=fetch_conditions)  # 傳遞上下文
                    request_counter += 1
                    async with session.get(url, headers=headers, timeout=10) as response:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After", "未知")
                            headers_info = dict(response.headers)
                            wait_time = int(retry_after) if retry_after.isdigit() else 5
                            wait_time = min(wait_time * (2 ** attempt), 60) + random.uniform(0

---

<xaiArtifact artifact_id="8841bcfc-d970-4906-87b9-e9309345ee33" artifact_version_id="d66f0182-309e-45f3-8245-863300ac9169" title="app.py" contentType="text/python">
import streamlit as st
from chat_page import chat_page
from test_page import test_page
import asyncio
import nest_asyncio

# 應用 nest_asyncio 來允許在 Streamlit 中嵌套運行事件迴圈
nest_asyncio.apply()

async def main():
    st.sidebar.title("導航")
    page = st.sidebar.selectbox("選擇頁面", ["聊天介面", "測試頁面"])
    
    if page == "聊天介面":
        await chat_page()
    elif page == "測試頁面":
        await test_page()

if __name__ == "__main__":
    # 在 Streamlit 中運行異步主函數
    asyncio.run(main())
