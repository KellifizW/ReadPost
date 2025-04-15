import streamlit as st
import streamlit.logger
import requests
import hashlib
import time
import re
from datetime import datetime
import pytz
import asyncio
import json
from typing import AsyncGenerator, Dict, Any

logger = streamlit.logger.get_logger(__name__)

LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 8000
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 初始化 session state
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
    """清理 HTML 標籤與多餘空白"""
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

def try_parse_date(date_str):
    """嘗試解析日期，支援 ISO 與 timestamp 格式"""
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        try:
            return datetime.fromtimestamp(int(date_str), tz=HONG_KONG_TZ)
        except (ValueError, TypeError):
            return None

def chunk_text(texts, max_chars=3000):
    """將文本分塊，限制每塊最大字元數"""
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
    """異步 HTTP 請求，支援重試與流式回應"""
    for attempt in range(retries):
        try:
            if method == "get":
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=30) as response:
                        response.raise_for_status()
                        return await response.json()
            elif method == "post":
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=json, timeout=30) as response:
                        response.raise_for_status()
                        if stream:
                            return response
                        return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                logger.warning(f"API 請求失敗，第 {attempt+1} 次重試: {url}, 錯誤: {str(e)}")
                await asyncio.sleep(5)
                continue
            logger.error(f"API 請求失敗: {url}, 錯誤: {str(e)}")
            raise e

async def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=5):
    """抓取 LIHKG 帖子列表"""
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
            digest = hashlib.sha1(f"jeams$get${url}${timestamp}".encode()).hexdigest()
            
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
                logger.info(f"LIHKG API 請求: cat_id={cat_id}, page={page}, 狀態: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") == 0:
                        logger.info(f"LIHKG API 無帖子: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 訊息: {data.get('error_message', '無錯誤訊息')}")
                        break
                    items = data.get("response", {}).get("items", [])
                    filtered_items = [item for item in items if item.get("title")]
                    logger.info(f"LIHKG 抓取: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 帖子數={len(filtered_items)}")
                    all_items.extend(filtered_items)
                    if not items:
                        logger.info(f"LIHKG 無更多帖子: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}")
                        break
                else:
                    logger.warning(f"LIHKG API 錯誤: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 狀態: {response.status_code}")
                    break
            except Exception as e:
                logger.warning(f"LIHKG API 錯誤: cat_id={cat_id}, sub_cat_id={sub_id}, page={page}, 錯誤: {str(e)}")
                break
        tasks = []
    
    if not all_items:
        logger.warning(f"無有效帖子: cat_id={cat_id}")
        st.warning(f"分類 {cat_id} 無帖子，請稍後重試")
    
    logger.info(f"元數據總計: cat_id={cat_id}, 帖子數={len(all_items)}, 標題示例={[item['title'] for item in all_items[:3]]}")
    return all_items

async def get_lihkg_thread_content(thread_id, cat_id=None, max_replies=175):
    """抓取單個帖子內容"""
    replies = []
    page = 1
    per_page = 50
    
    while len(replies) < max_replies:
        url = f"{LIHKG_BASE_URL}thread/{thread_id}/page/{page}?order=reply_time"
        timestamp = int(time.time())
        digest = hashlib.sha1(f"jeams$get${url}${timestamp}".encode()).hexdigest()
        
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
            logger.info(f"LIHKG 帖子內容: thread_id={thread_id}, page={page}, 狀態: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                page_replies = data.get("response", {}).get("item_data", [])
                replies.extend(page_replies)
                page += 1
                if not page_replies:
                    logger.info(f"LIHKG 帖子無更多回覆: thread_id={thread_id}, page={page}")
                    break
            else:
                logger.warning(f"LIHKG 帖子內容錯誤: thread_id={thread_id}, page={page}, 狀態: {response.status_code}")
                break
        except Exception as e:
            logger.warning(f"LIHKG 帖子內容錯誤: thread_id={thread_id}, page={page}, 錯誤: {str(e)}")
            break
    
    return replies[:max_replies]

def build_post_context(post, replies):
    """構建帖子上下文，包含標題與回覆"""
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

async def stream_grok3_response(text: str, call_id: str = None, recursion_depth: int = 0) -> AsyncGenerator[str, None]:
    """調用 Grok 3 API，支援流式回應"""
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
            async for chunk in stream_grok3_response(chunk_prompt, call_id=f"{call_id}_sub_{i}", recursion_depth=recursion_depth + 1):
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
    
    try:
        response = await async_request("post", GROK3_API_URL, headers=headers, json=payload, stream=True)
        async for line in response.content.iter_lines():
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        logger.warning(f"JSON 解析失敗: call_id={call_id}, 數據={line_str}")
                        continue
        logger.info(f"Grok 3 流式完成: call_id={call_id}, 輸入字元: {char_count}")
    except Exception as e:
        logger.error(f"Grok 3 異常: call_id={call_id}, 錯誤: {str(e)}")
        yield f"錯誤: {str(e)}"

async def analyze_lihkg_metadata(user_query, cat_id=1, max_pages=5):
    """第一階段：分析所有帖子標題，生成廣泛意見總結"""
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
        return f"今日 {cat_name} 無符合條件的帖子，可能是討論量低，建議查看熱門台（cat_id=2）。"
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}, 最後回覆: {item['last_reply_time']}"
        for item in st.session_state.metadata
    ])
    
    cat_name = {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(cat_id, "未知分類")
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區今日（2025-04-15）回覆數 ≥125 且 Unix Timestamp ≥ {today_timestamp} 的所有帖子元數據，分類為 {cat_name}（cat_id={cat_id}）：
{metadata_text}

以繁體中文回答，基於所有帖子標題，綜合分析討論區的廣泛意見，直接回答問題，禁止生成無關內容。執行以下步驟：
1. 解析問題意圖，識別核心主題（如財經、情緒、搞笑、爭議、時事、生活等）。若含「股票」「市場」「投資」「態度」「情緒」，視為財經情緒問題。
2. 若為財經情緒問題（如「市場情緒」）：
   - 分析所有標題，推斷網民整體情緒（「淡友」「跌」=悲觀；「定期存款」「儲蓄」=謹慎；「認真討論」「分享」=中性或分歧）。
   - 提取常見關鍵詞（如「美股」「淡友」），反映討論趨勢。
   - 總結網民對財經主題的整體情緒（100-150 字），說明是否樂觀、悲觀、中性或分歧，並註明依據（如標題中的情緒線索）。
3. 若為其他主題：
   - 根據分類（cat_id）適配語氣：
     - 吹水台（cat_id=1）：輕鬆，提取搞笑、荒誕話題。
     - 熱門台（cat_id=2）：聚焦高熱度討論。
     - 時事台（cat_id=5）：關注爭議、事件。
     - 上班台（cat_id=14）：聚焦職場、生活。
     - 財經台（cat_id=15）：偏市場、投資。
     - 成人台（cat_id=29）：適度處理敏感話題。
     - 創意台（cat_id=31）：注重創意、技術。
   - 總結標題反映的網民意見（100-150 字），突出熱門話題或共識。
4. 若問題與標題無關，回覆「問題與今日帖子無關，請提供更具體問題」。
"""
    return prompt

async def summarize_thread(thread_id, cat_id, user_query):
    """第二階段：分析單個帖子內容，生成詳細總結"""
    logger.info(f"開始總結帖子: thread_id={thread_id}, 分類={cat_id}, 問題='{user_query}'")
    post = next((item for item in st.session_state.metadata if str(item["thread_id"]) == str(thread_id)), None)
    if not post:
        logger.warning(f"帖子不存在: thread_id={thread_id}")
        return f"錯誤: 帖子 ID {thread_id} 不存在"
    
    replies = await get_lihkg_thread_content(thread_id, cat_id=cat_id, max_replies=175)
    context = build_post_context(post, replies)
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區單個帖子的內容，分類 cat_id={cat_id}，帖子 ID={thread_id}：
{context}

以繁體中文回答，基於帖子標題與回覆內容，詳細總結網民意見，回答使用者問題，禁止生成無關內容。執行以下步驟：
1. 解析問題意圖，識別核心主題（如財經、情緒、搞笑、爭議、時事、生活等）。
2. 若為財經情緒問題：
   - 分析標題與回覆，提取情緒關鍵詞（如「淡友」「看好」）。
   - 總結網民對財經主題的看法（150-200 字），說明情緒傾向（樂觀、悲觀、中性、分歧），並引用具體回覆內容。
3. 若為其他主題：
   - 根據分類適配語氣（同上）。
   - 總結帖子反映的網民意見（150-200 字），突出共識或分歧，引用代表性回覆。
4. 若問題與帖子無關，回覆「問題與帖子內容無關，請提供更具體問題」。
"""
    call_id = f"summarize_{thread_id}_{int(time.time())}"
    return await stream_grok3_response(prompt, call_id=call_id)

def chat_page():
    """聊天頁面邏輯，支援流式顯示"""
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
        selected_cat = st.selectbox("選擇分類", options=list(cat_id_map.keys()), index=list(cat_id_map.keys()).index({1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(st.session_state.last_cat_id, "吹水台")))
        st.session_state.last_cat_id = cat_id_map[selected_cat]
    
    with col1:
        user_query = st.chat_input("輸入問題（例如：財經台的市場情緒如何？）")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if user_query and not st.session_state.is_fetching:
        st.session_state.is_fetching = True
        st.session_state.last_user_query = user_query
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("正在分析討論區數據..."):
                    try:
                        prompt = await analyze_lihkg_metadata(user_query, cat_id=st.session_state.last_cat_id, max_pages=5)
                        call_id = f"analyze_{st.session_state.last_cat_id}_{int(time.time())}"
                        full_response = ""
                        async for chunk in stream_grok3_response(prompt, call_id=call_id):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        error_msg = f"錯誤: {str(e)}"
                        placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        logger.error(f"分析失敗: 問題='{user_query}', 錯誤: {str(e)}")
        
        st.session_state.is_fetching = False
        
        if st.session_state.metadata and not full_response.startswith("今日") and not full_response.startswith("錯誤"):
            with chat_container:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown("正在加載帖子詳細分析...")
                    try:
                        st.session_state.waiting_for_summary = True
                        thread_id = st.session_state.metadata[0]["thread_id"]
                        full_response = ""
                        async for chunk in summarize_thread(thread_id, st.session_state.last_cat_id, user_query):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        error_msg = f"錯誤: 帖子分析失敗 - {str(e)}"
                        placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        logger.error(f"帖子總結失敗: thread_id={thread_id}, 錯誤: {str(e)}")
                    finally:
                        st.session_state.waiting_for_summary = False

if __name__ == "__main__":
    chat_page()
