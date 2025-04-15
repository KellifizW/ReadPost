import streamlit as st
import streamlit.logger
import aiohttp
import asyncio
import json
import re
import time
import hashlib
from datetime import datetime
import pytz
from typing import AsyncGenerator, Dict, Any

logger = streamlit.logger.get_logger(__name__)

LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 8000
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

if "lihkg_data" not in st.session_state:
    st.session_state.lihkg_data = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "char_counts" not in st.session_state:
    st.session_state.char_counts = {}
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "is_fetching" not in st.session_state:
    st.session_state.is_fetching = False
if "last_call_id" not in st.session_state:
    st.session_state.last_call_id = None
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""
if "waiting_for_summary" not in st.session_state:
    st.session_state.waiting_for_summary = False
if "last_cat_id" not in st.session_state:
    st.session_state.last_cat_id = 1

def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

def try_parse_date(date_str):
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        try:
            return datetime.fromtimestamp(int(date_str), tz=HONG_KONG_TZ)
        except (ValueError, TypeError):
            return None

def chunk_text(texts, max_chars=3000):
    chunks = []
    current_chunk = ""
    for text in texts:
        if len(current_chunk) + len(text) + 1 < max_chars:
            current_chunk += text + "\n"
        else:
            if len(current_chunk) >= 100:
                chunks.append(current_chunk)
            current_chunk = text + "\n"
    if current_chunk and len(current_chunk) >= 100:
        chunks.append(current_chunk)
    return chunks

async def async_request(method, url, headers=None, json=None, retries=3, stream=False):
    connector = aiohttp.TCPConnector(limit=10)
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                if method == "get":
                    async with session.get(url, headers=headers, timeout=60) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method == "post":
                    async with session.post(url, headers=headers, json=json, timeout=60) as response:
                        response.raise_for_status()
                        if stream:
                            return response
                        return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                logger.warning(f"API 請求失敗，第 {attempt+1} 次重試: {url}, 錯誤: {str(e)}")
                await asyncio.sleep(2 ** attempt)
                continue
            logger.error(f"API 請求失敗: {url}, 錯誤: {str(e)}")
            raise e

async def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=5):
    all_items = []
    tasks = []
    
    endpoint = "thread/hot" if cat_id == 2 else "thread/category"
    sub_cat_ids = [0] if cat_id == 2 else ([0, 1, 2] if cat_id == 29 else [sub_cat_id])
    max_pages = 1 if cat_id == 2 else max_pages
    
    for sub_id in sub_cat_ids:
        for p in range(start_page, start_page + max_pages):
            if cat_id == 2:
                url = f"{LIHKG_BASE_URL}{endpoint}?cat_id={cat_id}&page={p}&count=60&type=now"
            else:
                url = f"{LIHKG_BASE_URL}{endpoint}?cat_id={cat_id}&sub_cat_id={sub_id}&page={p}&count=60&type=now"
            timestamp = int(time.time())
            digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
            
            headers = {
                "X-LI-DEVICE": LIHKG_DEVICE_ID,
                "X-LI-DEVICE-TYPE": "android",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "X-LI-REQUEST-TIME": str(timestamp),
                "X-LI-DIGEST": digest,
                "Cookie": LIHKG_COOKIE,
                "orginal": "https://lihkg.com",
                "referer": f"https://lihkg.com/category/{cat_id}",
                "accept": "application/json",
            }
            
            tasks.append((p, async_request("get", url, headers=headers)))
        
        for page, task in tasks:
            try:
                response = await task
                logger.info(f"LIHKG API 請求: cat_id={cat_id}, page={page}")
                data = response
                if data.get("success") == 0:
                    logger.info(f"LIHKG API 無帖子: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}")
                    break
                items = data.get("response", {}).get("items", [])
                filtered_items = [item for item in items if item.get("title")]
                logger.info(f"LIHKG 抓取: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 帖子數={len(filtered_items)}")
                all_items.extend(filtered_items)
                if not items:
                    logger.info(f"LIHKG 無更多帖子: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}")
                    break
            except Exception as e:
                logger.warning(f"LIHKG API 錯誤: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 錯誤: {str(e)}")
                break
        tasks = []
    
    if not all_items:
        logger.warning(f"無有效帖子: cat_id={cat_id}")
        st.warning(f"分類 {cat_id} 無帖子，請稍後重試")
    
    logger.info(f"元數據總計: cat_id={cat_id}, 帖子數={len(all_items)}")
    return all_items

async def get_lihkg_thread_content(thread_id, cat_id=None, max_replies=175):
    replies = []
    page = 1
    per_page = 50
    
    while len(replies) < max_replies:
        url = f"{LIHKG_BASE_URL}thread/{thread_id}/page/{page}?order=reply_time"
        timestamp = int(time.time())
        digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
        
        headers = {
            "X-LI-DEVICE": LIHKG_DEVICE_ID,
            "X-LI-DEVICE-TYPE": "android",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "Cookie": LIHKG_COOKIE,
            "orginal": "https://lihkg.com",
            "referer": f"https://lihkg.com/thread/{thread_id}",
            "accept": "application/json",
        }
        
        try:
            response = await async_request("get", url, headers=headers)
            logger.info(f"LIHKG 帖子內容: thread_id={thread_id}, page={page}")
            data = response
            page_replies = data.get("response", {}).get("item_data", [])
            replies.extend(page_replies)
            page += 1
            if not page_replies:
                logger.info(f"LIHKG 帖子無更多回覆: thread_id={thread_id}, page={page}")
                break
        except Exception as e:
            logger.warning(f"LIHKG 帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤: {str(e)}")
            break
    
    return replies[:max_replies]

async def get_lihkg_thread_rating(thread_id, cat_id=None):
    url = f"{LIHKG_BASE_URL}thread/{thread_id}"
    timestamp = int(time.time())
    digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
    
    headers = {
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-DEVICE-TYPE": "android",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest,
        "Cookie": LIHKG_COOKIE,
        "orginal": "https://lihkg.com",
        "referer": f"https://lihkg.com/thread/{thread_id}",
        "accept": "application/json",
    }
    
    try:
        response = await async_request("get", url, headers=headers)
        logger.info(f"LIHKG 帖子評分: thread_id={thread_id}")
        logger.debug(f"原始回應: {json.dumps(response, ensure_ascii=False)}")
        thread_data = response.get("response", {}).get("thread", {})
        like_count = int(thread_data.get("like_count", "0")) if thread_data.get("like_count") else 0
        dislike_count = int(thread_data.get("dislike_count", "0")) if thread_data.get("dislike_count") else 0
        if like_count == 0 and dislike_count == 0:
            logger.warning(f"正負評為 0，可能缺少資料: thread_id={thread_id}")
        return like_count, dislike_count
    except Exception as e:
        logger.error(f"LIHKG 帖子評分錯誤: thread_id={thread_id}, 錯誤: {str(e)}")
        return 0, 0

def build_post_context(post, replies):
    context = f"標題: {post['title']}\n"
    max_chars = 7000
    if replies:
        context += "回覆:\n"
        char_count = len(context)
        for reply in replies:
            msg = clean_html(reply['msg'])
            if len(msg) > 10:
                msg_line = f"- {msg}\n"
                if char_count + len(msg_line) > max_chars:
                    break
                context += msg_line
                char_count += len(msg_line)
    return context

async def stream_grok3_response(text: str, call_id: str = None, recursion_depth: int = 0, retries: int = 3) -> AsyncGenerator[str, None]:
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        st.error("未找到 Grok 3 API 密鑰，請檢查配置")
        yield "錯誤: 缺少 API 密鑰"
        return
    
    char_count = len(text)
    if call_id:
        st.session_state.char_counts[call_id] = char_count
    
    if len(text) > GROK3_TOKEN_LIMIT:
        if recursion_depth > 2:
            logger.error(f"輸入超限: call_id={call_id}, 字元數={char_count}, 遞迴過深")
            yield "錯誤: 輸入過長，無法分塊處理"
            return
        logger.warning(f"輸入超限: call_id={call_id}, 字元數={char_count}")
        chunks = chunk_text([text], max_chars=3000)
        logger.info(f"分塊處理: call_id={call_id}, 分塊數={len(chunks)}")
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"使用者問題：{st.session_state.get('last_user_query', '')}\n{chunk}"
            if len(chunk_prompt) > GROK3_TOKEN_LIMIT:
                chunk_prompt = chunk_prompt[:GROK3_TOKEN_LIMIT - 100] + "\n[已截斷]"
                logger.warning(f"分塊超限: call_id={call_id}_sub_{i}, 字元數={len(chunk_prompt)}")
            async for chunk in stream_grok3_response(chunk_prompt, call_id=f"{call_id}_sub_{i}", recursion_depth=recursion_depth + 1, retries=retries):
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
                                logger.warning(f"JSON 解析失敗: call_id={call_id}, 數據={line_str}")
                                continue
            logger.info(f"Grok 3 流式完成: call_id={call_id}, 輸入字元: {char_count}")
            return
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Grok 3 請求失敗，第 {attempt+1} 次重試: call_id={call_id}, 錯誤: {str(e)}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.error(f"Grok 3 異常: call_id={call_id}, 錯誤: {str(e)}")
            yield f"錯誤: 連線失敗，請稍後重試或檢查網路"

async def analyze_lihkg_metadata(user_query, cat_id=1, max_pages=5, max_metadata=50):
    logger.info(f"開始分析: 分類={cat_id}, 問題='{user_query}'")
    st.session_state.metadata = []
    items = await get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=max_pages)
    
    today_start = datetime.now(HONG_KONG_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today_start.timestamp())
    filtered_items = [
        item for item in items
        if item.get("no_of_reply", 0) >= 125 and int(item.get("last_reply_time", 0)) >= today_timestamp
    ]
    sorted_items = sorted(filtered_items, key=lambda x: x.get("no_of_reply", 0), reverse=True)
    sorted_items = sorted_items[:max_metadata]
    st.session_state.metadata = [
        {
            "thread_id": item["thread_id"],
            "title": item["title"],
            "no_of_reply": item.get("no_of_reply", 0),
            "last_reply_time": item.get("last_reply_time", "")
        }
        for item in sorted_items
    ]
    
    if not st.session_state.metadata:
        logger.warning(f"無有效帖子: 分類={cat_id}")
        cat_name = {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(cat_id, "未知分類")
        return f"今日 {cat_name} 無符合條件的帖子，建議查看熱門台（cat_id=2）。"
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}, 最後回覆: {item['last_reply_time']}"
        for item in st.session_state.metadata
    ])
    
    cat_name = {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(cat_id, "未知分類")
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區今日（2025-04-15）回覆數 ≥125 且 Unix Timestamp ≥ {today_timestamp} 的前 {max_metadata} 篇帖子元數據，分類為 {cat_name}（cat_id={cat_id}）：
{metadata_text}

以繁體中文回答，基於所有帖子標題，綜合分析討論區的廣泛意見，直接回答問題，禁止生成無關內容。執行以下步驟：
1. 解析問題意圖，識別核心主題（如財經、情緒、搞笑、爭議、時事、生活等）。
2. 若為其他主題：
   - 根據分類適配語氣：
     - 吹水台（cat_id=1）：輕鬆，提取搞笑、荒誕話題。
     - 熱門台（cat_id=2）：聚焦高熱度討論。
     - 時事台（cat_id=5）：關注爭議、事件。
     - 上班台（cat_id=14）：聚焦職場、生活。
     - 財經台（cat_id=15）：偏市場、投資。
     - 成人台（cat_id=29）：適度處理敏感話題。
     - 創意台（cat_id=31）：注重趣味、創意。
   - 總結網民整體觀點（100-150 字），提取標題關鍵詞，直接回答問題。
3. 若標題不足以詳細回答，註明：「可進一步分析帖子內容以提供補充細節。」
4. 若無相關帖子，說明：「今日 {cat_name} 無符合問題的帖子，建議查看熱門台（cat_id=2）。」

輸出格式：
- 總結：100-150 字，直接回答問題，概述網民整體觀點，說明依據（如標題關鍵詞）。
"""
    return prompt

async def select_relevant_threads(user_query, max_threads=3):
    if not st.session_state.metadata:
        logger.warning("無元數據可選擇帖子")
        return []
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}"
        for item in st.session_state.metadata
    ])
    
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區今日（2025-04-15）回覆數 ≥125 的帖子元數據：
{metadata_text}

基於標題，選擇與問題最相關的帖子 ID（每行一個數字），最多 {max_threads} 個。僅返回 ID，無其他內容。若無相關帖子，返回空列表。
"""
    
    call_id = f"select_{time.time()}"
    response = ""
    async for chunk in stream_grok3_response(prompt, call_id=call_id):
        response += chunk
    thread_ids = re.findall(r'\b(\d{5,})\b', response, re.MULTILINE)
    valid_ids = [str(item["thread_id"]) for item in st.session_state.metadata]
    selected_ids = [tid for tid in thread_ids if tid in valid_ids]
    
    logger.info(f"選擇帖子: 提取={thread_ids}, 有效={selected_ids}")
    if not selected_ids:
        logger.warning("無有效帖子 ID")
        return []
    
    return selected_ids[:max_threads]

async def summarize_thread(thread_id, cat_id=None) -> AsyncGenerator[str, None]:
    post = next((item for item in st.session_state.metadata if str(item["thread_id"]) == str(thread_id)), None)
    if not post:
        logger.error(f"找不到帖子: thread_id={thread_id}")
        yield f"錯誤: 找不到帖子 {thread_id}"
        return
    
    replies = await get_lihkg_thread_content(thread_id, cat_id=cat_id)
    like_count, dislike_count = await get_lihkg_thread_rating(thread_id, cat_id=cat_id)
    st.session_state.lihkg_data[thread_id] = {"post": post, "replies": replies}
    
    if len(replies) < 50:
        logger.info(f"帖子 {thread_id} 回覆數={len(replies)}，生成簡短總結")
        yield f"標題: {post['title']}\n總結: 討論參與度低，網民回應不足，話題未見熱烈討論。（回覆數: {len(replies)}）\n評分: 正評 {like_count}, 負評 {dislike_count}"
        return
    
    context = build_post_context(post, replies)
    chunks = chunk_text([context], max_chars=3000)
    
    logger.info(f"總結帖子 {thread_id}: 回覆數={len(replies)}, 分塊數={len(chunks)}")
    chunk_summaries = []
    cat_name = {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(cat_id, "未知分類")
    for i, chunk in enumerate(chunks):
        logger.debug(f"分塊內容: thread_id={thread_id}, chunk_{i}, 字元數={len(chunk)}")
        prompt = f"""
請將帖子（ID: {thread_id}）總結為 100-200 字，僅基於以下內容，聚焦標題與回覆，作為問題的補充細節，禁止引入無關話題。以繁體中文回覆。

帖子內容：
{chunk}

參考使用者問題「{st.session_state.get('last_user_query', '')}」與分類（cat_id={cat_id}，{cat_name}），執行以下步驟：
1. 識別問題意圖（如搞笑、爭議、時事、財經）。
2. 總結網民觀點，回答問題，適配分類語氣：
   - 吹水台：提取搞笑、輕鬆觀點。
   - 熱門台：反映熱門焦點。
   - 時事台：聚焦爭議或事件。
   - 財經台：分析市場情緒。
   - 其他：適配主題。
3. 若內容與問題無關，返回：「內容與問題不符，無法回答。」
4. 若數據不足，返回：「內容不足，無法生成總結。」

輸出格式：
- 標題：<標題>
- 總結：100-200 字，反映網民觀點，說明依據。
"""
        summary = ""
        async for chunk in stream_grok3_response(prompt, call_id=f"{thread_id}_chunk_{i}"):
            if chunk.startswith("錯誤:"):
                logger.error(f"帖子 {thread_id} 分塊 {i} 總結失敗: {chunk}")
                yield chunk
                return
            summary += chunk
        chunk_summaries.append(summary)
    
    final_prompt = f"""
請將帖子（ID: {thread_id}）的總結合併為 150-200 字，僅基於以下內容，聚焦標題與回覆，作為問題的補充細節，禁止引入無關話題。以繁體中文回覆。

分塊總結：
{'\n'.join(chunk_summaries)}

參考使用者問題「{st.session_state.get('last_user_query', '')}」與分類（cat_id={cat_id}，{cat_name}），執行以下步驟：
1. 識別問題意圖（如搞笑、爭議、時事、財經）。
2. 總結網民觀點，回答問題，適配分類語氣：
   - 吹水台：提取搞笑、輕鬆觀點。
   - 熱門台：反映熱門焦點。
   - 時事台：聚焦爭議或事件。
   - 財經台：分析市場情緒。
   - 其他：適配主題。
3. 若內容與問題無關，返回：「內容與問題不符，無法回答。」
4. 若數據不足，返回：「內容不足，無法生成總結。」

輸出格式：
- 標題：<標題>
- 總結：150-200 字，反映網民觀點，說明依據。
- 評分：正評 {like_count}, 負評 {dislike_count}
"""
    
    if len(final_prompt) > GROK3_TOKEN_LIMIT:
        chunks = chunk_text([final_prompt], max_chars=3000)
        for i, chunk in enumerate(chunks):
            async for sub_chunk in stream_grok3_response(chunk, call_id=f"{thread_id}_final_sub_{i}"):
                if sub_chunk.startswith("錯誤:"):
                    logger.error(f"帖子 {thread_id} 最終分塊 {i} 總結失敗: {sub_chunk}")
                    yield sub_chunk
                    return
                yield sub_chunk
    else:
        async for chunk in stream_grok3_response(final_prompt, call_id=f"{thread_id}_final"):
            if chunk.startswith("錯誤:"):
                logger.error(f"帖子 {thread_id} 最終總結失敗: {chunk}")
                yield chunk
                return
            yield chunk

def async_to_sync_stream(async_gen):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def get_next():
        try:
            return loop.run_until_complete(anext(async_gen))
        except StopAsyncIteration:
            return None
    
    while True:
        chunk = get_next()
        if chunk is None:
            break
        yield chunk
    
    loop.close()

def chat_page():
    st.title("LIHKG 討論區分析")
    
    cat_id_map = {
        "吹水台": 1,
        "熱門台": 2,
        "時事台": 5,
        "上班台": 14,
        "財經台": 15,
        "成人台": 29,
        "創意台": 31
    }
    
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_cat = st.selectbox(
            "選擇分類",
            options=list(cat_id_map.keys()),
            index=list(cat_id_map.keys()).index(
                {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(st.session_state.last_cat_id, "吹水台")
            )
        )
        st.session_state.last_cat_id = cat_id_map[selected_cat]
    
    with col1:
        user_query = st.chat_input("輸入問題（例如：吹水台有哪些搞笑話題？）或回應（『需要』、『ID 數字』、『不需要』）")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if user_query and not st.session_state.is_fetching:
        st.session_state.is_fetching = True
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
            
            if st.session_state.waiting_for_summary:
                prompt_lower = user_query.lower().strip()
                
                if prompt_lower == "不需要":
                    st.session_state.waiting_for_summary = False
                    response = "好的，已結束深入分析。你可以提出新問題！"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    logger.info("用戶選擇不需要，結束第二階段")
                
                elif prompt_lower == "需要":
                    with st.spinner("正在選擇並總結相關帖子..."):
                        try:
                            thread_ids = asyncio.run(select_relevant_threads(st.session_state.last_user_query))
                            if thread_ids:
                                summaries = []
                                for thread_id in thread_ids:
                                    with st.chat_message("assistant"):
                                        full_summary = st.write_stream(async_to_sync_stream(summarize_thread(thread_id, cat_id=st.session_state.last_cat_id)))
                                        if not full_summary.startswith("錯誤:"):
                                            summaries.append(full_summary)
                                            st.session_state.messages.append({"role": "assistant", "content": full_summary})
                                if summaries:
                                    response = "以上是相關帖子總結。你需要進一步分析其他帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                                else:
                                    response = "無法生成帖子總結，可能數據不足。你需要我嘗試其他分析嗎？（輸入『需要』或『不需要』）"
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                            else:
                                response = "無相關帖子可總結。你需要我嘗試其他分析嗎？（輸入『需要』或『不需要』）"
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                            logger.info(f"自動選擇帖子: {thread_ids}")
                        except Exception as e:
                            logger.error(f"帖子總結失敗: 錯誤={str(e)}")
                            response = f"總結失敗：{str(e)}。請重試或提出新問題。"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
                            st.session_state.waiting_for_summary = False
                
                elif re.match(r'^(id\s*)?\d{5,}$', prompt_lower.replace(" ", "")):
                    thread_id = re.search(r'\d{5,}', prompt_lower).group()
                    valid_ids = [str(item["thread_id"]) for item in st.session_state.metadata]
                    if thread_id in valid_ids:
                        with st.spinner(f"正在總結帖子 {thread_id}..."):
                            try:
                                with st.chat_message("assistant"):
                                    full_summary = st.write_stream(async_to_sync_stream(summarize_thread(thread_id, cat_id=st.session_state.last_cat_id)))
                                    if not full_summary.startswith("錯誤:"):
                                        response = f"帖子 {thread_id} 的總結：\n\n{full_summary}\n\n你需要進一步分析其他帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                                    else:
                                        response = f"帖子 {thread_id} 總結失敗：{full_summary}。你需要嘗試其他帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    st.markdown(response)
                            except Exception as e:
                                logger.error(f"帖子 {thread_id} 總結失敗: 錯誤={str(e)}")
                                response = f"總結失敗：{str(e)}。請重試或提出新問題。"
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                                st.session_state.waiting_for_summary = False
                    else:
                        response = f"無效帖子 ID {thread_id}，請確認 ID 是否正確。你需要我自動選擇帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        logger.warning(f"無效帖子 ID: {thread_id}")
                
                else:
                    response = "請輸入『需要』以自動選擇帖子、『ID 數字』以指定帖子，或『不需要』以結束。"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    logger.warning(f"無效回應: {user_query}")
            
            else:
                st.session_state.metadata = []
                st.session_state.char_counts = {}
                st.session_state.waiting_for_summary = False
                st.session_state.last_user_query = user_query
                
                with st.spinner("正在分析討論區數據..."):
                    try:
                        prompt = asyncio.run(analyze_lihkg_metadata(user_query, cat_id=st.session_state.last_cat_id, max_pages=5))
                        call_id = f"analyze_{st.session_state.last_cat_id}_{int(time.time())}"
                        with st.chat_message("assistant"):
                            full_response = st.write_stream(async_to_sync_stream(stream_grok3_response(prompt, call_id=call_id)))
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        if not full_response.startswith("今日") and not full_response.startswith("錯誤"):
                            response = "你需要我對某個帖子生成更深入的總結嗎？請輸入『需要』以自動選擇帖子、『ID 數字』以指定帖子，或『不需要』以結束。"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
                            st.session_state.waiting_for_summary = True
                    except Exception as e:
                        logger.error(f"分析失敗: 問題='{user_query}', 錯誤: {str(e)}")
                        response = f"錯誤: {str(e)}。請重試或檢查網路連線。"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
        
        st.session_state.is_fetching = False

def main():
    st.sidebar.title("導航")
    page = st.sidebar.selectbox("選擇頁面", ["聊天介面"])
    
    if page == "聊天介面":
        chat_page()

if __name__ == "__main__":
    main()
