import streamlit as st
import requests
import hashlib
import time
import re
from datetime import datetime
import pytz
import asyncio

# LIHKG 配置（公開，硬編碼）
LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"

# Grok 3 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 4000

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 儲存數據的全局變量
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

# 清理HTML標籤
def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

# 分塊文字
def chunk_text(texts, max_chars=GROK3_TOKEN_LIMIT):
    chunks = []
    current_chunk = ""
    for text in texts:
        if len(current_chunk) + len(text) + 1 < max_chars:
            current_chunk += text + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = text + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 非同步API調用（支援重試）
async def async_request(method, url, headers=None, json=None, retries=1):
    for attempt in range(retries + 1):
        try:
            loop = asyncio.get_event_loop()
            if method == "get":
                response = await loop.run_in_executor(None, lambda: requests.get(url, headers=headers, timeout=10))
            elif method == "post":
                response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=json, timeout=10))
            response.raise_for_status()
            return response
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            if attempt < retries:
                st.warning(f"API 請求失敗（{str(e)}），重試 {attempt + 1}/{retries}...")
                await asyncio.sleep(2)
            else:
                raise e

# 抓取LIHKG帖子列表（元數據）
async def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=1, count=50):
    all_items = []
    tasks = []
    
    for p in range(start_page, start_page + max_pages):
        url = f"{LIHKG_BASE_URL}thread/latest?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={p}&count={count}&type=now&order=reply_time"
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
            "referer": f"https://lihkg.com/category/{cat_id}?order=reply_time",
            "accept": "application/json",
        }
        
        tasks.append(async_request("get", url, headers=headers))
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for response in responses:
        if isinstance(response, Exception):
            st.error(f"LIHKG API 錯誤: {str(response)}")
            continue
        if response.status_code == 200:
            data = response.json()
            if data.get("success") == 0:
                st.write(f"API 錯誤: {data}")
                break
            items = data.get("response", {}).get("items", [])
            all_items.extend(items)
            if not items:
                break
        else:
            st.error(f"LIHKG API 錯誤: {response.status_code}")
            if response.status_code == 403:
                st.warning("LIHKG Cookie 可能過期，請更新 PHPSESSID")
            break
    
    return all_items

# 抓取帖子回覆
async def get_lihkg_thread_content(thread_id, max_replies=100):
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
        
        response = await async_request("get", url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            page_replies = data.get("response", {}).get("item_data", [])
            replies.extend(page_replies)
            page += 1
            if not page_replies:
                break
        else:
            st.error(f"LIHKG API 錯誤: {response.status_code}")
            if response.status_code == 403:
                st.warning("LIHKG Cookie 可能過期，請更新 PHPSESSID")
            break
    
    return replies[:max_replies]

# 構建帖子上下文
def build_post_context(post, replies):
    context = f"標題: {post['title']}\n"
    if replies:
        context += "回覆:\n"
        for reply in replies:
            msg = clean_html(reply['msg'])
            if len(msg) > 10:
                context += f"- {msg}\n"
    return context

# 調用Grok 3 API
async def summarize_with_grok3(text, call_id=None):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        st.error("未找到 Grok 3 API 密鑰，請在 secrets.toml 或 Streamlit Cloud 中配置 [grok3key]")
        return "錯誤: 缺少 API 密鑰"
    
    char_count = len(text)
    if call_id:
        st.session_state.char_counts[call_id] = char_count
    else:
        st.session_state.char_counts[f"temp_{time.time()}"] = char_count
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer убийстваGROK3_API_KEY}"
    }
    payload = {
        "model": "grok-3-beta",
        "messages": [
            {"role": "system", "content": "你是 Grok 3，請使用繁體中文回答所有問題，並確保回覆清晰、簡潔、符合繁體中文語法規範。"},
            {"role": "user", "content": text}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    try:
        response = await async_request("post", GROK3_API_URL, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            st.error("Grok 3 API 認證失敗：無效的 API 密鑰，請檢查 [grok3key]")
        elif response.status_code == 404:
            st.error(f"Grok 3 API 端點無效：{GROK3_API_URL}")
        elif response.status_code == 429:
            st.error("Grok 3 API 請求超限，請稍後重試")
        else:
            st.error(f"Grok 3 API 錯誤: {str(e)}")
        return f"錯誤: {str(e)}"
    except Exception as e:
        st.error(f"Grok 3 API 總結失敗: {str(e)}")
        return f"錯誤: {str(e)}"

# 分析LIHKG元數據
async def analyze_lihkg_metadata(user_query, cat_id=1, max_pages=1):
    if not st.session_state.metadata:
        items = await get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=max_pages)
        st.session_state.metadata = [
            {
                "thread_id": item["thread_id"],
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", "")
            }
            for item in items
        ]
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}, 最後回覆: {item['last_reply_time']}"
        for item in st.session_state.metadata
    ])
    
    prompt = f"""
    使用者問題：{user_query}
    
    以下是 LIHKG 討論區的帖子元數據（包含帖子 ID、標題、回覆數和最後回覆時間）：
    {metadata_text}
    
    請以繁體中文分析這些元數據，回答使用者的問題，並指出哪些帖子可能與問題最相關（列出帖子 ID 和標題）。若問題提到「膠post」，請優先選擇標題或內容看似荒唐、搞笑、誇張或非現實的帖子（例如包含「無厘頭」「趣怪」「惡趣味」「on9」等詞語，或描述不合常理的情境）。若問題涉及「熱門」，則考慮回覆數最多或最近更新的帖子。請確保回覆簡潔，包含具體的帖子 ID 和標題。
    """
    
    call_id = f"metadata_{len(st.session_state.chat_history)}"
    response = await summarize_with_grok3(prompt, call_id=call_id)
    return response

# 選擇相關帖子
async def select_relevant_threads(analysis_result, max_threads=3):
    prompt = f"""
    以下是對 LIHKG 帖子元數據的分析結果：
    {analysis_result}
    
    請從中挑選最多 {max_threads} 個最相關的帖子，僅返回帖子 ID 列表，格式為純數字（一行一個），例如：
    12345
    67890
    """
    
    call_id = f"select_{len(st.session_state.chat_history)}"
    response = await summarize_with_grok3(prompt, call_id=call_id)
    
    thread_ids = re.findall(r'^\d+$', response, re.MULTILINE)
    valid_ids = [item["thread_id"] for item in st.session_state.metadata]
    selected_ids = [tid for tid in thread_ids if tid in valid_ids]
    
    if not selected_ids:
        st.warning("無法解析帖子 ID，選擇回覆數最多的帖子")
        selected_ids = [
            item["thread_id"] for item in sorted(
                st.session_state.metadata,
                key=lambda x: x["no_of_reply"],
                reverse=True
            )[:max_threads]
        ]
    
    return selected_ids[:max_threads]

# 總結帖子
async def summarize_thread(thread_id):
    post = next((item for item in st.session_state.metadata if item["thread_id"] == thread_id), None)
    if not post:
        return f"錯誤: 找不到帖子 {thread_id}"
    
    replies = await get_lihkg_thread_content(thread_id)
    st.session_state.lihkg_data[thread_id] = {"post": post, "replies": replies}
    
    context = build_post_context(post, replies)
    chunks = chunk_text([context])
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = await summarize_with_grok3(
            f"請將以下討論區帖子和回覆總結為100-200字，聚焦主要主題和關鍵意見，並以繁體中文回覆：\n\n{chunk}",
            call_id=f"{thread_id}_chunk_{i}"
        )
        chunk_summaries.append(summary)
    
    final_summary = await summarize_with_grok3(
        f"請將以下分塊總結合併為100-200字的最終總結，聚焦主要主題和關鍵意見，並以繁體中文回覆：\n\n{'\n'.join(chunk_summaries)}",
        call_id=f"{thread_id}_final"
    )
    return final_summary

# 手動抓取和總結
async def manual_fetch_and_summarize(cat_id, sub_cat_id, start_page, max_pages, auto_sub_cat):
    st.session_state.lihkg_data = {}
    st.session_state.summaries = {}
    st.session_state.char_counts = {}
    all_items = []
    
    sub_cat_ids = [0, 1, 2, 3, 4, 5] if auto_sub_cat else [sub_cat_id]
    
    for sub_id in sub_cat_ids:
        items = await get_lihkg_topic_list(cat_id, sub_id, start_page, max_pages)
        existing_ids = {item["thread_id"] for item in all_items}
        new_items = [item for item in items if item["thread_id"] not in existing_ids]
        all_items.extend(new_items)
    
    filtered_items = [item for item in all_items if item.get("no_of_reply", 0) > 175]
    sorted_items = sorted(filtered_items, key=lambda x: x["last_reply_time"], reverse=True)
    top_items = sorted_items[:10]
    
    for item in top_items:
        thread_id = item["thread_id"]
        replies = await get_lihkg_thread_content(thread_id)
        st.session_state.lihkg_data[thread_id] = {"post": item, "replies": replies}
        
        context = build_post_context(item, replies)
        chunks = chunk_text([context])
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await summarize_with_grok3(
                f"請將以下討論區帖子和回覆總結為100-200字，聚焦主要主題和關鍵意見，並以繁體中文回覆：\n\n{chunk}",
                call_id=f"{thread_id}_chunk_{i}"
            )
            chunk_summaries.append(summary)
        
        final_summary = await summarize_with_grok3(
            f"請將以下分塊總結合併為100-200字的最終總結，聚焦主要主題和關鍵意見，並以繁體中文回覆：\n\n{'\n'.join(chunk_summaries)}",
            call_id=f"{thread_id}_final"
        )
        st.session_state.summaries[thread_id] = final_summary

# Streamlit主程式
def main():
    st.title("LIHKG 總結聊天機器人")

    st.header("與 Grok 3 聊天")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("輸入問題（例如「有無咩膠post分享?」）：", key="chat_input")
        submit_chat = st.form_submit_button("提交問題")
    
    if submit_chat and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("正在分析 LIHKG 帖子..."):
            analysis_result = asyncio.run(analyze_lihkg_metadata(user_input))
        st.session_state.chat_history.append({"role": "assistant", "content": analysis_result})
        
        with st.spinner("正在選擇並總結相關帖子..."):
            thread_ids = asyncio.run(select_relevant_threads(analysis_result))
            if thread_ids:
                for thread_id in thread_ids:
                    if thread_id not in st.session_state.summaries:
                        summary = asyncio.run(summarize_thread(thread_id))
                        st.session_state.summaries[thread_id] = summary
    
    if st.session_state.chat_history:
        st.subheader("聊天記錄")
        for i, chat in enumerate(st.session_state.chat_history):
            role = "你" if chat["role"] == "user" else "Grok 3"
            st.write(f"**{role}**：{chat['content']}")
            if chat["role"] == "assistant":
                call_id = f"metadata_{i//2}"
                char_count = st.session_state.char_counts.get(call_id, "處理中")
                st.write(f"**處理字元數**：{char_count if isinstance(char_count, int) else char_count} 字元")
            st.write("---")

    st.header("帖子總結")
    if st.session_state.summaries:
        for thread_id, summary in st.session_state.summaries.items():
            post = st.session_state.lihkg_data[thread_id]["post"]
            st.write(f"**標題**: {post['title']} (ID: {thread_id})")
            st.write(f"**總結**: {summary}")
            chunk_counts = [st.session_state.char_counts.get(f"{thread_id}_chunk_{i}", 0) for i in range(len(chunk_text([build_post_context(post, st.session_state.lihkg_data[thread_id]["replies"])])))]
            final_count = st.session_state.char_counts.get(f"{thread_id}_final", 0)
            st.write(f"**處理字元數**：分塊總結 {sum(chunk_counts)} 字元，最終總結 {final_count} 字元")
            if st.button(f"查看詳情 {thread_id}", key=f"detail_{thread_id}"):
                st.write("**回覆內容**：")
                for reply in st.session_state.lihkg_data[thread_id]["replies"]:
                    st.write(f"- {clean_html(reply['msg'])}")
            st.write("---")

    st.header("手動抓取 LIHKG 帖子")
    with st.form("manual_fetch_form"):
        cat_id = st.text_input("分類 ID (如 1 為吹水台)", "1")
        sub_cat_id = st.number_input("子分類 ID", min_value=0, value=0)
        start_page = st.number_input("開始頁數", min_value=1, value=1)
        max_pages = st.number_input("最大頁數", min_value=1, value=5)
        auto_sub_cat = st.checkbox("自動遍歷子分類 (0-5)", value=True)
        submit_fetch = st.form_submit_button("抓取並總結")
    
    if submit_fetch:
        with st.spinner("正在抓取並總結..."):
            asyncio.run(manual_fetch_and_summarize(cat_id, sub_cat_id, start_page, max_pages, auto_sub_cat))

if __name__ == "__main__":
    main()
