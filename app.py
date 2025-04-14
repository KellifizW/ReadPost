import streamlit as st
import requests
import hashlib
import time
import re
from datetime import datetime

# LIHKG API 配置
LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = hashlib.sha1("random-uuid".encode()).hexdigest()
LIHKG_TOKEN = ""  # 初始 token 為空
LIHKG_USER_ID = ""  # 初始 user_id 為空

# 帖子類型選項（模擬原作者的 QueryType）
QUERY_TYPES = {
    "最新 (Now)": "now",
    "每日熱門 (Daily)": "daily",
    "每週熱門 (Weekly)": "weekly"
}

# 清理 HTML 標籤的輔助函數
def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# LIHKG 登錄功能
def login_lihkg(email, password):
    url = f"{LIHKG_BASE_URL}auth/login"
    payload = {"email": email, "password": password}
    timestamp = int(time.time())
    digest = hashlib.sha1(f"jeams$post${url}${payload}${LIHKG_TOKEN}${timestamp}".encode()).hexdigest()
    
    headers = {
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-DEVICE-TYPE": "android",
        "User-Agent": "LIHKG/16.0.4 Android/9.0.0 Google/Pixel XL",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest
    }
    
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            return data["response"]["token"], data["response"]["user"]["user_id"]
    return None, None

# LIHKG 抓取分類帖子列表
def get_lihkg_topic_list(cat_id, sub_cat_id=0, page=1, count=30, query_type="now"):
    url = f"{LIHKG_BASE_URL}thread/category?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={page}&count={count}&type={query_type}"
    timestamp = int(time.time())
    st.write(f"調試: 當前時間戳: {timestamp}, 對應時間: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    digest = hashlib.sha1(f"jeams$get${url}${LIHKG_TOKEN}${timestamp}".encode()).hexdigest()
    
    headers = {
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-DEVICE-TYPE": "android",
        "User-Agent": "LIHKG/16.0.4 Android/9.0.0 Google/Pixel XL",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest
    }
    if LIHKG_USER_ID:
        headers["X-LI-USER"] = LIHKG_USER_ID
    
    response = requests.get(url, headers=headers)
    st.write(f"調試: 請求 LIHKG 分類帖子 URL: {url}, 狀態碼: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"LIHKG API 錯誤: {response.status_code}"}

# LIHKG 搜索帖子（備選方案）
def search_lihkg_posts(cat_id, sub_cat_id=0, page=1, count=30, sort="desc_reply_time"):
    url = f"{LIHKG_BASE_URL}thread/search?cat_id={cat_id}&sub_cat_id={sub_cat_id}&q=&sort={sort}&page={page}&count={count}"
    timestamp = int(time.time())
    digest = hashlib.sha1(f"jeams$get${url}${LIHKG_TOKEN}${timestamp}".encode()).hexdigest()
    
    headers = {
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-DEVICE-TYPE": "android",
        "User-Agent": "LIHKG/16.0.4 Android/9.0.0 Google/Pixel XL",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest
    }
    if LIHKG_USER_ID:
        headers["X-LI-USER"] = LIHKG_USER_ID
    
    response = requests.get(url, headers=headers)
    st.write(f"調試: 搜索 LIHKG 帖子 URL: {url}, 狀態碼: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"LIHKG API 錯誤: {response.status_code}"}

# LIHKG 抓取帖子回覆內容
def get_lihkg_thread_content(thread_id, page=1, order="reply_time"):
    url = f"{LIHKG_BASE_URL}thread/{thread_id}/page/{page}?order={order}"
    timestamp = int(time.time())
    digest = hashlib.sha1(f"jeams$get${url}${LIHKG_TOKEN}${timestamp}".encode()).hexdigest()
    
    headers = {
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-DEVICE-TYPE": "android",
        "User-Agent": "LIHKG/16.0.4 Android/9.0.0 Google/Pixel XL",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest
    }
    if LIHKG_USER_ID:
        headers["X-LI-USER"] = LIHKG_USER_ID
    
    st.write(f"調試: 請求 LIHKG 帖子回覆 URL: {url}")
    response = requests.get(url, headers=headers)
    st.write(f"調試: LIHKG 回覆請求狀態碼: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        st.write(f"調試: LIHKG 回覆返回數據: {data}")
        return data
    else:
        return {"error": f"LIHKG API 錯誤: {response.status_code}, 回應: {response.text}"}

# Streamlit 主程式
def main():
    st.title("LIHKG 分類帖子抓取工具")

    # 初始化 session_state
    if "lihkg_posts" not in st.session_state:
        st.session_state.lihkg_posts = None
    if "lihkg_replies" not in st.session_state:
        st.session_state.lihkg_replies = {}

    # LIHKG 登錄（可選）
    st.subheader("LIHKG 登錄（可選）")
    email = st.text_input("輸入 LIHKG 郵箱")
    password = st.text_input("輸入 LIHKG 密碼", type="password")
    if st.button("登錄 LIHKG"):
        token, user_id = login_lihkg(email, password)
        if token and user_id:
            global LIHKG_TOKEN, LIHKG_USER_ID
            LIHKG_TOKEN = token
            LIHKG_USER_ID = user_id
            st.success(f"登錄成功！Token: {token}, User ID: {user_id}")
        else:
            st.error("登錄失敗，請檢查郵箱和密碼")

    # LIHKG 測試區
    st.header("LIHKG 分類帖子與回覆")
    lihkg_cat_id = st.text_input("輸入 LIHKG 分類 ID (例如 1 表示吹水台)", "1")
    lihkg_sub_cat_id = st.number_input("輸入 LIHKG 子分類 ID (默認為 0)", min_value=0, value=0)
    lihkg_page = st.number_input("LIHKG 分類頁數", min_value=1, value=1)
    
    # 添加帖子類型選擇（模擬 QueryType）
    query_type_label = st.selectbox("選擇帖子類型", list(QUERY_TYPES.keys()), index=0)
    query_type = QUERY_TYPES[query_type_label]

    # 添加搜索選項
    use_search = st.checkbox("使用搜索端點 (可能獲取更新的帖子)", value=False)

    if st.button("抓取 LIHKG 分類帖子"):
        # 清除舊數據
        st.session_state.lihkg_posts = None
        if use_search:
            data = search_lihkg_posts(lihkg_cat_id, lihkg_sub_cat_id, lihkg_page, sort="desc_reply_time")
        else:
            data = get_lihkg_topic_list(lihkg_cat_id, lihkg_sub_cat_id, lihkg_page, query_type=query_type)
        if "error" in data:
            st.error(data["error"])
        else:
            st.session_state.lihkg_posts = data

    # 顯示 LIHKG 帖子列表並檢查最新帖子時間
    if st.session_state.lihkg_posts and "response" in st.session_state.lihkg_posts and "items" in st.session_state.lihkg_posts["response"]:
        items = st.session_state.lihkg_posts["response"]["items"]
        if items:
            latest_post = max(items, key=lambda x: x["last_reply_time"])
            st.write(f"調試: 最新帖子的最後回覆時間: {datetime.fromtimestamp(latest_post['last_reply_time']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        for item in items:
            st.write(f"**標題**: {item['title']}")
            st.write(f"**帖子 ID**: {item['thread_id']}")
            st.write(f"**用戶**: {item['user_nickname']} (性別: {item['user_gender']})")
            st.write(f"**回覆數**: {item['no_of_reply']}, **點讚數**: {item['like_count']}, **負評數**: {item['dislike_count']}")
            st.write(f"**創建時間**: {datetime.fromtimestamp(item['create_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**最後回覆時間**: {datetime.fromtimestamp(item['last_reply_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            reply_page = st.number_input(f"選擇帖子 {item['thread_id']} 的回覆頁數", min_value=1, value=1, key=f"reply_page_{item['thread_id']}")
            if st.button(f"查看帖子 {item['thread_id']} 的回覆"):  # 已修復
                thread_data = get_lihkg_thread_content(item["thread_id"], reply_page)
                if "error" in thread_data:
                    st.error(thread_data["error"])
                else:
                    st.session_state.lihkg_replies[item["thread_id"]] = thread_data
            
            if item["thread_id"] in st.session_state.lihkg_replies:
                thread_data = st.session_state.lihkg_replies[item["thread_id"]]
                if "response" in thread_data and "item_data" in thread_data["response"]:
                    st.subheader(f"帖子 {item['thread_id']} 的回覆（第 {reply_page} 頁）")
                    for reply in thread_data["response"]["item_data"]:
                        st.write(f"**回覆用戶**: {reply['user_nickname']} (性別: {reply['user_gender']})")
                        st.write(f"**回覆內容**: {clean_html(reply['msg'])}")
                        st.write(f"**回覆時間**: {datetime.fromtimestamp(reply['reply_time']).strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write("---")
            st.write("---")

if __name__ == "__main__":
    main()
