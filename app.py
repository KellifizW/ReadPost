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
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"  # 正確的 xAI API 端點
GROK3_TOKEN_LIMIT = 4000  # 假設的 token 限制

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 儲存數據的全局變量
if "lihkg_data" not in st.session_state:
    st.session_state.lihkg_data = {}
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# 非同步API調用
async def async_request(method, url, headers=None, json=None):
    loop = asyncio.get_event_loop()
    if method == "get":
        response = await loop.run_in_executor(None, lambda: requests.get(url, headers=headers))
    elif method == "post":
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=json))
    return response

# 抓取LIHKG帖子列表
async def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=5, count=100):
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
        if isinstance(response, requests.RequestException):
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

# 構建上下文
def build_post_context(post, replies):
    context = f"Title: {post['title']}\n"
    if replies:
        context += "Replies:\n"
        for reply in replies:
            msg = clean_html(reply['msg'])
            if len(msg) > 10:
                context += f"- {msg}\n"
    return context

# 調用Grok 3 API
async def summarize_with_grok3(text):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        st.error("未找到 Grok 3 API 密鑰，請在 secrets.toml 或 Streamlit Cloud 中配置 [grok3key]")
        return "錯誤: 缺少 API 密鑰"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK3_API_KEY}"
    }
    payload = {
        "model": "grok-3-beta",  # 假設模型名稱，需確認
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following forum post and replies into 100-200 words, focusing on main themes and key opinions:\n\n{text}"}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    try:
        response = await async_request("post", GROK3_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            st.error("Grok 3 API 認證失敗：無效的 API 密鑰，請檢查 [grok3key]")
        elif response.status_code == 404:
            st.error(f"Grok 3 API 端點無效：{GROK3_API_URL}，請確認 xAI API 文檔")
        elif response.status_code == 429:
            st.error("Grok 3 API 請求超限，請稍後重試")
        else:
            st.error(f"Grok 3 API 錯誤: {str(e)}")
        return f"錯誤: {str(e)}"
    except Exception as e:
        st.error(f"Grok 3 API 總結失敗: {str(e)}")
        return f"錯誤: {str(e)}"

# Streamlit主程式
def main():
    st.title("LIHKG 總結聊天機器人")

    st.header("抓取 LIHKG 帖子")
    cat_id = st.text_input("分類 ID (如 1 為吹水台)", "1")
    sub_cat_id = st.number_input("子分類 ID", min_value=0, value=0)
    start_page = st.number_input("開始頁數", min_value=1, value=1)
    max_pages = st.number_input("最大頁數", min_value=1, value=5)
    auto_sub_cat = st.checkbox("自動遍歷子分類 (0-5)", value=True)

    if st.button("抓取並總結"):
        st.session_state.lihkg_data = {}
        st.session_state.summaries = {}
        all_items = []
        
        sub_cat_ids = [0, 1, 2, 3, 4, 5] if auto_sub_cat else [sub_cat_id]
        
        for sub_id in sub_cat_ids:
            items = asyncio.run(get_lihkg_topic_list(cat_id, sub_id, start_page, max_pages))
            existing_ids = {item["thread_id"] for item in all_items}
            new_items = [item for item in items if item["thread_id"] not in existing_ids]
            all_items.extend(new_items)
        
        filtered_items = [item for item in all_items if item.get("no_of_reply", 0) > 175]
        sorted_items = sorted(filtered_items, key=lambda x: x["last_reply_time"], reverse=True)
        top_items = sorted_items[:10]
        
        for item in top_items:
            thread_id = item["thread_id"]
            replies = asyncio.run(get_lihkg_thread_content(thread_id))
            st.session_state.lihkg_data[thread_id] = {"post": item, "replies": replies}
            
            context = build_post_context(item, replies)
            chunks = chunk_text([context])
            
            chunk_summaries = []
            for chunk in chunks:
                summary = asyncio.run(summarize_with_grok3(chunk))
                chunk_summaries.append(summary)
            
            final_summary = asyncio.run(summarize_with_grok3("\n".join(chunk_summaries)))
            st.session_state.summaries[thread_id] = final_summary

    if st.session_state.summaries:
        st.header("帖子總結")
        for thread_id, summary in st.session_state.summaries.items():
            post = st.session_state.lihkg_data[thread_id]["post"]
            st.write(f"**標題**: {post['title']} (ID: {thread_id})")
            st.write(f"**總結**: {summary}")
            if st.button(f"查看詳情 {thread_id}", key=f"detail_{thread_id}"):
                st.write("**回覆內容**：")
                for reply in st.session_state.lihkg_data[thread_id]["replies"]:
                    st.write(f"- {clean_html(reply['msg'])}")
            st.write("---")

    st.header("與 Grok 3 互動")
    user_input = st.text_input("輸入問題或指令：", key="chat_input")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        context = "\n".join([f"Post {tid}: {summary}" for tid, summary in st.session_state.summaries.items()])
        prompt = f"以下是LIHKG帖子總結：\n{context}\n\n用戶問題：{user_input}"
        
        chunks = chunk_text([prompt])
        responses = []
        for chunk in chunks:
            response = asyncio.run(summarize_with_grok3(chunk))
            responses.append(response)
        
        final_response = "\n".join(responses)
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})

    st.subheader("聊天記錄")
    for chat in st.session_state.chat_history:
        role = "你" if chat["role"] == "user" else "Grok 3"
        st.write(f"**{role}**：{chat['content']}")

if __name__ == "__main__":
    main()
