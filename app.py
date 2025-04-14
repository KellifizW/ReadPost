import streamlit as st
import requests
import hashlib
import time
import re
from datetime import datetime  # 新增這一行來導入 datetime

# LIHKG API 配置
LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = hashlib.sha1("random-uuid".encode()).hexdigest()
LIHKG_TOKEN = ""  # 如果有 token 可填入

# 清理 HTML 標籤的輔助函數
def clean_html(text):
    # 移除 HTML 標籤
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    # 移除多餘的換行和空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# LIHKG 抓取分類帖子列表
def get_lihkg_topic_list(cat_id, page=1, count=30):
    url = f"{LIHKG_BASE_URL}thread/category?cat_id={cat_id}&page={page}&count={count}&type=now"
    timestamp = int(time.time())
    digest = hashlib.sha1(f"jeams$get${url}${LIHKG_TOKEN}${timestamp}".encode()).hexdigest()
    
    headers = {
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-DEVICE-TYPE": "android",
        "User-Agent": "LIHKG/16.0.4 Android/9.0.0 Google/Pixel XL",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest
    }
    
    response = requests.get(url, headers=headers)
    st.write(f"調試: 請求 LIHKG 分類帖子 URL: {url}, 狀態碼: {response.status_code}")
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

    # LIHKG 測試區
    st.header("LIHKG 分類帖子與回覆")
    lihkg_cat_id = st.text_input("輸入 LIHKG 分類 ID (例如 1 表示吹水台)", "1")
    lihkg_page = st.number_input("LIHKG 分類頁數", min_value=1, value=1)
    if st.button("抓取 LIHKG 分類帖子"):
        data = get_lihkg_topic_list(lihkg_cat_id, lihkg_page)
        if "error" in data:
            st.error(data["error"])
        else:
            st.session_state.lihkg_posts = data

    # 顯示 LIHKG 帖子列表
    if st.session_state.lihkg_posts and "response" in st.session_state.lihkg_posts and "items" in st.session_state.lihkg_posts["response"]:
        for item in st.session_state.lihkg_posts["response"]["items"]:
            st.write(f"**標題**: {item['title']}")
            st.write(f"**帖子 ID**: {item['thread_id']}")
            st.write(f"**用戶**: {item['user_nickname']} (性別: {item['user_gender']})")
            st.write(f"**回覆數**: {item['no_of_reply']}, **點讚數**: {item['like_count']}, **負評數**: {item['dislike_count']}")
            st.write(f"**創建時間**: {datetime.fromtimestamp(item['create_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 選擇回覆頁數
            reply_page = st.number_input(f"選擇帖子 {item['thread_id']} 的回覆頁數", min_value=1, value=1, key=f"reply_page_{item['thread_id']}")
            if st.button(f"查看帖子 {item['thread_id']} 的回覆", key=f"view_replies_{item['thread_id']}"):
                thread_data = get_lihkg_thread_content(item["thread_id"], reply_page)
                if "error" in thread_data:
                    st.error(thread_data["error"])
                else:
                    st.session_state.lihkg_replies[item["thread_id"]] = thread_data
            
            # 顯示已抓取的回覆
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
