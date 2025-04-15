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

logger = streamlit.logger.get_logger(__name__)

LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 8000
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

if "lihkg_data" not in st.session_state:
    st.session_state.lihkg_data = {}
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "char_counts" not in st.session_state:
    st.session_state.char_counts = {}
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "is_fetching" not in st.session_state:
    st.session_state.is_fetching = False
if "last_call_id" not in st.session_state:
    st.session_state.last_call_id = None

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

def chunk_text(texts, max_chars=3000):  # 降低 max_chars
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

async def async_request(method, url, headers=None, json=None, retries=3):  # 增加重試次數
    for attempt in range(retries + 1):
        try:
            loop = asyncio.get_event_loop()
            if method == "get":
                response = await loop.run_in_executor(None, lambda: requests.get(url, headers=headers, timeout=30))  # 增加超時
            elif method == "post":
                response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=json, timeout=30))
            response.raise_for_status()
            return response
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            if attempt < retries:
                logger.warning(f"API 請求失敗，第 {attempt+1} 次重試: {url}, 錯誤: {str(e)}")
                await asyncio.sleep(5)  # 延遲 5 秒
                continue
            logger.error(f"API 請求失敗: {url}, 錯誤: {str(e)}")
            raise e

async def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=5):
    all_items = []
    tasks = []
    
    endpoint = "thread/hot" if cat_id == 2 else "thread/category"
    sub_cat_ids = [0] if cat_id == 2 else ([0, 1, 2] if cat_id == 29 else [sub_cat_id])
    max_pages = 1 if cat_id == 2 else max_pages  # 限制熱門台 page=1
    
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
                        break  # 立即終止後續頁面
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

async def summarize_with_grok3(text, call_id=None, recursion_depth=0):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API 密鑰缺失")
        st.error("未找到 Grok 3 API 密鑰，請檢查配置")
        return "錯誤: 缺少 API 密鑰"
    
    char_count = len(text)
    if call_id:
        st.session_state.char_counts[call_id] = char_count
    
    if len(text) > GROK3_TOKEN_LIMIT:
        if recursion_depth > 2:
            logger.error(f"輸入超限: call_id={call_id}, 字元數={char_count}, 遞迴過深")
            return "錯誤: 輸入過長，無法分塊處理"
        logger.warning(f"輸入超限: call_id={call_id}, 字元數={char_count}")
        chunks = chunk_text([text], max_chars=3000)
        logger.info(f"分塊處理: call_id={call_id}, 分塊數={len(chunks)}")
        summaries = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"使用者問題：{st.session_state.get('last_user_query', '')}\n{chunk}"
            if len(chunk_prompt) > GROK3_TOKEN_LIMIT:
                chunk_prompt = chunk_prompt[:GROK3_TOKEN_LIMIT - 100] + "\n[已截斷]"
                logger.warning(f"分塊超限: call_id={call_id}_sub_{i}, 字元數={len(chunk_prompt)}")
            summary = await summarize_with_grok3(chunk_prompt, call_id=f"{call_id}_sub_{i}", recursion_depth=recursion_depth + 1)
            summaries.append(summary)
        return "\n".join(summaries)
    
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
        "temperature": 0.7
    }
    
    try:
        response = await async_request("post", GROK3_API_URL, headers=headers, json=payload)
        logger.info(f"Grok 3 API: call_id={call_id}, 輸入字元: {char_count}")
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Grok 3 異常: call_id={call_id}, 錯誤: {str(e)}")
        return f"錯誤: {str(e)}"

async def analyze_lihkg_metadata(user_query, cat_id=1, max_pages=5):
    logger.info(f"開始分析: 分類={cat_id}, 問題='{user_query}'")
    st.session_state.metadata = []
    items = await get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=max_pages)
    
    # 篩選今日帖子
    today_start = datetime.now(HONG_KONG_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today_start.timestamp())
    filtered_items = [
        item for item in items
        if item.get("no_of_reply", 0) >= 125 and int(item.get("last_reply_time", 0)) >= today_timestamp
    ]
    sorted_items = sorted(filtered_items, key=lambda x: x.get("no_of_reply", 0), reverse=True)[:100]  # 限制 100 帖
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
        return f"今日無符合回覆數 ≥125 的帖子，建議查看其他分類。"
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}, 最後回覆: {item['last_reply_time']}"
        for item in st.session_state.metadata
    ])
    
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區今日（2025-04-15）的帖子元數據：
{metadata_text}

以繁體中文回答，僅基於元數據，禁止生成無關內容。選擇回覆數 ≥125 且今日內（Unix Timestamp ≥ {today_timestamp}）的帖子，列出最多 3 個（格式：帖子 ID: <數字>, 標題: <標題>）。若問題含「新聞」或「熱門」，總結網民討論焦點（100-150 字）。若無符合帖子，說明「今日無符合條件的帖子」。
"""
    
    call_id = f"metadata_{time.time()}"
    st.session_state.last_call_id = call_id
    st.session_state.last_user_query = user_query
    logger.info(f"準備 Grok 3 分析: 分類={cat_id}, 元數據項目={len(st.session_state.metadata)}")
    response = await summarize_with_grok3(prompt, call_id=call_id)
    return response

async def select_relevant_threads(analysis_result, max_threads=3):
    prompt = f"""
以下是 LIHKG 帖子分析：
{analysis_result}

僅返回帖子 ID（每行一個數字），選擇回覆數 ≥125 且今日（2025-04-15）內的帖子，最多 {max_threads} 個。若無符合，返回空列表。
"""
    
    call_id = f"select_{time.time()}"
    response = await summarize_with_grok3(prompt, call_id=call_id)
    thread_ids = re.findall(r'\b(\d{5,})\b', response, re.MULTILINE)
    valid_ids = [str(item["thread_id"]) for item in st.session_state.metadata]
    selected_ids = [tid for tid in thread_ids if tid in valid_ids]
    
    logger.info(f"選擇帖子: 提取={thread_ids}, 有效={selected_ids}")
    if not selected_ids:
        logger.warning("無有效帖子 ID")
        return []
    
    return selected_ids[:max_threads]

async def summarize_thread(thread_id, cat_id=None):
    post = next((item for item in st.session_state.metadata if str(item["thread_id"]) == str(thread_id)), None)
    if not post:
        logger.error(f"找不到帖子: thread_id={thread_id}")
        return f"錯誤: 找不到帖子 {thread_id}"
    
    replies = await get_lihkg_thread_content(thread_id, cat_id=cat_id)
    st.session_state.lihkg_data[thread_id] = {"post": post, "replies": replies}
    
    if len(replies) < 50:
        logger.info(f"帖子 {thread_id} 回覆數={len(replies)}，生成簡短總結")
        return f"標題: {post['title']}\n總結: 討論參與度低，網民回應不足，話題未見熱烈討論。（回覆數: {len(replies)}）"
    
    context = build_post_context(post, replies)
    chunks = chunk_text([context], max_chars=3000)
    
    logger.info(f"總結帖子 {thread_id}: 回覆數={len(replies)}, 分塊數={len(chunks)}")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"分塊內容: thread_id={thread_id}, chunk_{i}, 字元數={len(chunk)}")
        summary = await summarize_with_grok3(
            f"""
請將帖子（ID: {thread_id}）總結為 100-200 字，僅基於以下內容，聚焦標題與回覆，禁止引入無關話題。以繁體中文回覆。

帖子內容：
{chunk}

若數據不足，返回「內容不足」。
""",
            call_id=f"{thread_id}_chunk_{i}"
        )
        if summary.startswith("錯誤:"):
            logger.error(f"帖子 {thread_id} 分塊 {i} 總結失敗: {summary}")
            return summary
        chunk_summaries.append(summary)
    
    reply_count = len(replies)
    word_range = "300-400" if reply_count >= 100 else "100-200"
    final_prompt = f"""
請將帖子（ID: {thread_id}）的分塊總結合併為 {word_range} 字，僅基於以下內容，聚焦標題與回覆，禁止引入無關話題。以繁體中文回覆。

分塊總結：
{'\n'.join(chunk_summaries)}

若數據不足，返回「內容不足」。
"""
    
    if len(final_prompt) > 1000:  # 檢查最終輸入
        chunks = chunk_text([final_prompt], max_chars=500)
        final_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await summarize_with_grok3(chunk, call_id=f"{thread_id}_final_sub_{i}")
            final_summaries.append(summary)
        final_summary = "\n".join(final_summaries)
    else:
        final_summary = await summarize_with_grok3(final_prompt, call_id=f"{thread_id}_final")
    
    if final_summary.startswith("錯誤:"):
        logger.error(f"帖子 {thread_id} 最終總結失敗: {final_summary}")
        return "\n".join(chunk_summaries) or "錯誤: 總結失敗"
    
    return final_summary

async def manual_fetch_and_summarize(cat_id, start_page, max_pages):
    st.session_state.is_fetching = True
    st.session_state.lihkg_data = {}
    st.session_state.summaries = {}
    logger.info(f"手動抓取: 分類={cat_id}, 頁數={start_page}-{start_page+max_pages-1}")
    
    items = await get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=start_page, max_pages=max_pages)
    today_start = datetime.now(HONG_KONG_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today_start.timestamp())
    filtered_items = [
        item for item in items
        if item.get("no_of_reply", 0) >= 125 and int(item.get("last_reply_time", 0)) >= today_timestamp
    ]
    sorted_items = sorted(filtered_items, key=lambda x: x.get("no_of_reply", 0), reverse=True)[:10]
    
    for item in sorted_items:
        thread_id = item["thread_id"]
        replies = await get_lihkg_thread_content(thread_id, cat_id=cat_id)
        st.session_state.lihkg_data[thread_id] = {"post": item, "replies": replies}
        
        summary = await summarize_thread(thread_id, cat_id=cat_id)
        if not summary.startswith("錯誤:"):
            st.session_state.summaries[thread_id] = summary
    
    st.session_state.is_fetching = False
    if not st.session_state.summaries:
        st.warning("無總結結果，可能無符合條件的帖子")
    st.rerun()

def main():
    st.title("LIHKG 總結聊天機器人")
    st.header("與 Grok 3 聊天")
    chat_cat_id = st.selectbox(
        "聊天分類",
        options=[1, 31, 5, 14, 15, 29, 2],
        format_func=lambda x: {1: "吹水台", 31: "創意台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 2: "熱門"}[x],
        key="chat_cat_id"
    )
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("輸入問題（如「今日有咩新聞?」）：", key="chat_input")
        submit_chat = st.form_submit_button("提交問題")
    
    if submit_chat and user_input:
        st.session_state.chat_history = []
        st.session_state.metadata = []
        st.session_state.char_counts = {}
        st.session_state.summaries = {}
        logger.info(f"開始新提問: 問題='{user_input}', 分類={chat_cat_id}")
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("正在分析 LIHKG 帖子..."):
            analysis_result = asyncio.run(analyze_lihkg_metadata(user_query=user_input, cat_id=chat_cat_id))
        st.session_state.chat_history.append({"role": "assistant", "content": analysis_result})
        
        with st.spinner("正在選擇並總結帖子..."):
            thread_ids = asyncio.run(select_relevant_threads(analysis_result))
            if thread_ids:
                for thread_id in thread_ids:
                    summary = asyncio.run(summarize_thread(thread_id, cat_id=chat_cat_id))
                    if not summary.startswith("錯誤:"):
                        st.session_state.summaries[thread_id] = summary
            if len(st.session_state.summaries) < 3 and thread_ids:  # 重試
                logger.info("總結數量不足，重新選擇帖子")
                thread_ids = asyncio.run(select_relevant_threads(analysis_result))
                for thread_id in thread_ids:
                    if thread_id not in st.session_state.summaries:
                        summary = asyncio.run(summarize_thread(thread_id, cat_id=chat_cat_id))
                        if not summary.startswith("錯誤:"):
                            st.session_state.summaries[thread_id] = summary
    
    if st.session_state.chat_history:
        st.subheader("聊天記錄")
        for chat in st.session_state.chat_history:
            role = "你" if chat["role"] == "user" else "Grok 3"
            st.markdown(f"**{role}**：{chat['content']}")
            st.write("---")
    
    st.header("帖子總結")
    if st.session_state.summaries:
        for thread_id, summary in st.session_state.summaries.items():
            post = st.session_state.lihkg_data[thread_id]["post"]
            st.write(f"**標題**: {post['title']} (ID: {thread_id})")
            st.write(f"**總結**: {summary}")
            st.write(f"**回覆數量**：{post['no_of_reply']} 條")
            chunk_counts = [st.session_state.char_counts.get(f"{thread_id}_chunk_{i}", 0) for i in range(10)]
            final_count = st.session_state.char_counts.get(f"{thread_id}_final", 0)
            st.write(f"**處理字元數**：分塊總結={sum(chunk_counts)} 字元，最終總結={final_count} 字元")
            st.write("---")
    else:
        st.info("尚無總結內容，請提交問題或手動抓取帖子。")

    st.header("手動抓取 LIHKG 帖子")
    with st.form("manual_fetch_form"):
        cat_id = st.text_input("分類 ID (如 5 為時事台)", "5")
        start_page = st.number_input("開始頁數", min_value=1, value=1)
        max_pages = st.number_input("最大頁數", min_value=1, value=5)
        submit_fetch = st.form_submit_button("抓取並總結")
    
    if submit_fetch:
        with st.spinner("正在抓取並總結..."):
            asyncio.run(manual_fetch_and_summarize(cat_id, start_page, max_pages))

if __name__ == "__main__":
    main()