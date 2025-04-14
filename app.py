import streamlit as st
import requests
import hashlib
import time
import re
from datetime import datetime
import pytz

# LIHKG API 配置
LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = hashlib.sha1("random-uuid".encode()).hexdigest()

# 帖子類型選項
QUERY_TYPES = {
    "最新 (Now)": "now",
    "每日熱門 (Daily)": "daily",
    "每週熱門 (Weekly)": "weekly",
    "最新回覆 (Latest)": "latest",
    "最近 (Recent)": "recent"
}

# 排序選項
ORDER_TYPES = {
    "現在 (Now)": "now",
    "按回覆時間 (Reply Time)": "reply_time",
    "最新 (New)": "new",
    "最新排序 (Latest)": "latest"
}

# 設置香港時區 (UTC+8)
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 清理 HTML 標籤的輔助函數
def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# LIHKG 抓取最新帖子列表（使用 thread/latest 端點）
def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=10, count=100, query_type="now", order="now"):
    all_items = []
    
    for p in range(start_page, start_page + max_pages):
        url = f"{LIHKG_BASE_URL}thread/latest?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={p}&count={count}&type={query_type}&order={order}"
        timestamp = int(time.time())
        hk_time = datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        st.write(f"調試: 當前時間戳: {timestamp}, 對應時間: {hk_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        digest = hashlib.sha1(f"jeams$get${url}${timestamp}".encode()).hexdigest()
        
        headers = {
            "X-LI-DEVICE": LIHKG_DEVICE_ID,
            "X-LI-DEVICE-TYPE": "android",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "orginal": "https://lihkg.com",
            "referer": f"https://lihkg.com/category/{cat_id}?order={order}",
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
            "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        }
        
        response = requests.get(url, headers=headers)
        st.write(f"調試: 請求 LIHKG 最新帖子 URL: {url}, 狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and "items" in data["response"]:
                items = data["response"]["items"]
                all_items.extend(items)
                st.write(f"調試: 第 {p} 頁抓取到 {len(items)} 個帖子，總計 {len(all_items)} 個帖子")
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "X-LI-REQUEST-TIME": str(timestamp),
        "X-LI-DIGEST": digest,
        "orginal": "https://lihkg.com",
        "referer": f"https://lihkg.com/thread/{thread_id}",
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
        "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }
    
    st.write(f"調試: 請求 LIHKG 帖子回覆 URL: {url}")
    response = requests.get(url, headers=headers)
    st.write(f"調試: LIHKG 回覆請求狀態碼: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return {"error": f"LIHKG API 錯誤: {response.status_code}, 回應: {response.text}"}

# Streamlit 主程式
def main():
    st.title("LIHKG 最新帖子抓取工具")

    # 初始化 session_state
    if "lihkg_posts" not in st.session_state:
        st.session_state.lihkg_posts = None
    if "lihkg_replies" not in st.session_state:
        st.session_state.lihkg_replies = {}

    # LIHKG 測試區
    st.header("LIHKG 最新帖子與回覆")
    lihkg_cat_id = st.text_input("輸入 LIHKG 分類 ID (例如 1 表示吹水台)", "1")
    lihkg_sub_cat_id = st.number_input("輸入 LIHKG 子分類 ID (默認為 0)", min_value=0, value=0)
    lihkg_start_page = st.number_input("開始頁數", min_value=1, value=1)
    lihkg_max_pages = st.number_input("最大抓取頁數", min_value=1, value=10)
    
    # 添加帖子類型選擇
    query_type_label = st.selectbox("選擇帖子類型", list(QUERY_TYPES.keys()), index=0)
    query_type = QUERY_TYPES[query_type_label]

    # 添加排序方式選擇
    order_type_label = st.selectbox("選擇排序方式", list(ORDER_TYPES.keys()), index=0)
    order_type = ORDER_TYPES[order_type_label]

    # 添加遍歷子分類選項
    auto_sub_cat = st.checkbox("自動遍歷多個子分類 (0-5)", value=True)

    if st.button("抓取 LIHKG 最新帖子"):
        # 清除舊數據
        st.session_state.lihkg_posts = None
        all_items = []
        
        if auto_sub_cat:
            sub_cat_ids = [0, 1, 2, 3, 4, 5]
        else:
            sub_cat_ids = [lihkg_sub_cat_id]
        
        # 遍歷每個子分類、帖子類型和排序方式
        for sub_id in sub_cat_ids:
            for qt in QUERY_TYPES.values():
                for ot in ORDER_TYPES.values():
                    st.write(f"正在抓取子分類 ID: {sub_id}, 帖子類型: {qt}, 排序方式: {ot}")
                    data = get_lihkg_topic_list(lihkg_cat_id, sub_id, start_page=lihkg_start_page, max_pages=lihkg_max_pages, query_type=qt, order=ot)
                    if "error" in data:
                        st.error(data["error"])
                    elif "response" in data and "items" in data["response"]:
                        items = data["response"]["items"]
                        # 避免重複帖子（根據 thread_id 去重）
                        existing_ids = {item["thread_id"] for item in all_items}
                        new_items = [item for item in items if item["thread_id"] not in existing_ids]
                        all_items.extend(new_items)
                        st.write(f"子分類 {sub_id} (類型: {qt}, 排序: {ot}) 抓取到 {len(new_items)} 個新帖子，總計 {len(all_items)} 個帖子")
        
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
