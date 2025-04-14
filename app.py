import streamlit as st
import requests
import hashlib
import time
import re
from datetime import datetime
import pytz

# LIHKG API 配置
LIHKG_BASE_URL = "https://lihkg.com Ascendants.io API 可能會限制你的請求次數。如果你覺得超時，或是遇到了問題，請聯繫 Atlassian 支援。

# 帖子類型選項（模擬原作者的 QueryType）
QUERY_TYPES = {
    "最新 (Now)": "now",
    "每日熱門 (Daily)": "daily",
    "每週熱門 (Weekly)": "weekly"
}

# 設置香港時區 (UTC+8)
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 清理 HTML 標籤的輔助函數
def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# LIHKG 抓取分類帖子列表（遍歷多頁）
def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=10, count=100, query_type="now"):
    all_items = []
    
    for p in range(start_page, start_page + max_pages):
        url = f"{LIHKG_BASE_URL}thread/category?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={p}&count={count}&type={query_type}"
        timestamp = int(time.time())
        hk_time = datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        st.write(f"調試: 當前時間戳: {timestamp}, 對應時間: {hk_time.strftime('%Y-%m-%d %H:%M:%S')}")
        digest = hashlib.sha1(f"jeams$get${url}${timestamp}".encode()).hexdigest()
        
        headers = {
            "X-LI-DEVICE": LIHKG_DEVICE_ID,
            "X-LI-DEVICE-TYPE": "android",
            "User-Agent": "LIHKG/16.0.4 Android/9.0.0 Google/Pixel XL",
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "orginal": "https://lihkg.com",
            "referer": f"https://lihkg.com/category/{cat_id}",
        }
        
        response = requests.get(url, headers=headers)
        st.write(f"調試: 請求 LIHKG 分類帖子 URL: {url}, 狀態碼: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            st.write(f"調試: 第 {p} 頁返回數據: {data}")
            if "response" in data and "items" in data["response"]:
                items = data["response"]["items"]
                all_items.extend(items)
                st.write(f"調試: 第 {p} 頁抓取到 {len(items)} 個帖子，總計 {len(all_items)} 個帖子")
                # 如果返回空列表，停止抓取
                if not items:
                    st.write(f"調試: 第 {p} 頁無帖子數據，停止抓取")
                    break
            else:
                st.write(f"調試: 第 {p} 頁無帖子數據，停止抓取")
                break
        else:
            st.error(f"LIHKG API 錯誤: {response.status_code}")
            break
    
    return {"response": {"items": all_items}}

# LIHKG 抓取帖子回覆內容
def get_lihkg_thread_content(thread_id, page=1, order="reply_time"):
    url = f"{LIHKG_BASE_URL}thread/{thread_id}/page/{page}?order={order}"
    timestamp = int(time.time())
    hk_time = datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
    st.write(f"調試: 當前時間戳: {timestamp}, 對應時間: {hk_time.strftime('%Y-%m-%d %H:%M:%S')}")
    digest = hashlib.sha1(f"jeams$get${url}${timestamp}".encode()).hexdigest()
    
    headers = {
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-DEVICE-TYPE": "android",
        "User-Agent": "LIHKG/16.0.4 Android/9.0.0 Google/Pixel XL",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest,
        "orginal": "https://lihkg.com",
        "referer": f"https://lihkg.com/thread/{thread_id}",
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
    lihkg_sub_cat_id = st.number_input("輸入 LIHKG 子分類 ID (默認為 0)", min_value=0, value=0)
    lihkg_start_page = st.number_input("開始頁數", min_value=1, value=1)
    lihkg_max_pages = st.number_input("最大抓取頁數", min_value=1, value=10)
    
    # 添加帖子類型選擇（模擬 QueryType）
    query_type_label = st.selectbox("選擇帖子類型", list(QUERY_TYPES.keys()), index=0)
    query_type = QUERY_TYPES[query_type_label]

    if st.button("抓取 LIHKG 分類帖子"):
        # 清除舊數據
        st.session_state.lihkg_posts = None
        # 嘗試不同 sub_cat_id 抓取更多帖子
        all_items = []
        sub_cat_ids = [0, 1, 2, 3]  # 嘗試多個子分類
        for sub_id in sub_cat_ids:
            st.write(f"正在抓取子分類 ID: {sub_id}")
            data = get_lihkg_topic_list(lihkg_cat_id, sub_id, start_page=lihkg_start_page, max_pages=lihkg_max_pages, query_type=query_type)
            if "error" in data:
                st.error(data["error"])
            elif "response" in data and "items" in data["response"]:
                items = data["response"]["items"]
                all_items.extend(items)
                st.write(f"子分類 {sub_id} 抓取到 {len(items)} 個帖子，總計 {len(all_items)} 個帖子")
        
        st.session_state.lihkg_posts = {"response": {"items": all_items}}

    # 顯示 LIHKG 帖子列表並檢查最新帖子時間
    if st.session_state.lihkg_posts and "response" in st.session_state.lihkg_posts and "items" in st.session_state.lihkg_posts["response"]:
        items = st.session_state.lihkg_posts["response"]["items"]
        st.write(f"總共抓取到 {len(items)} 個帖子")
        if items:
            # 按最後回覆時間排序
            items.sort(key=lambda x: x["last_reply_time"], reverse=True)
            latest_post = items[0]
            latest_reply_time = datetime.fromtimestamp(latest_post['last_reply_time'], tz=HONG_KONG_TZ)
            st.write(f"調試: 最新帖子的最後回覆時間: {latest_reply_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for item in items:
            st.write(f"**標題**: {item['title']}")
            st.write(f"**帖子 ID**: {item['thread_id']}")
            st.write(f"**用戶**: {item['user_nickname']} (性別: {item['user_gender']})")
            st.write(f"**回覆數**: {item['no_of_reply']}, **點讚數**: {item['like_count']}, **負評數**: {item['dislike_count']}")
            create_time = datetime.fromtimestamp(item['create_time'], tz=HONG_KONG_TZ)
            last_reply_time = datetime.fromtimestamp(item['last_reply_time'], tz=HONG_KONG_TZ)
            st.write(f"**創建時間**: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**最後回覆時間**: {last_reply_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            reply_page = st.number_input(f"選擇帖子 {item['thread_id']} 的回覆頁數", min_value=1, value=1, key=f"reply_page_{item['thread_id']}")
            if st.button(f"查看帖子 {item['thread_id']} 的回覆"):
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
                        reply_time = datetime.fromtimestamp(reply['reply_time'], tz=HONG_KONG_TZ)
                        st.write(f"**回覆時間**: {reply_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write("---")
            st.write("---")

if __name__ == "__main__":
    main()
