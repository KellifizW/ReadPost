import streamlit as st
import requests
import hashlib
import time
from urllib.parse import urlencode
from datetime import datetime

# LIHKG API 配置
LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = hashlib.sha1("random-uuid".encode()).hexdigest()
LIHKG_TOKEN = ""  # 如果有 token 可填入

# HKG API 配置
HKG_BASE_URL = "https://api-1.hkgolden.com/"
HKG_USER_ID = "0"
HKG_PASSWORD_HASH = ""

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
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"LIHKG API 錯誤: {response.status_code}"}

# HKG 抓取分類帖子列表
def get_hkg_topic_list(topic_type, page=1):
    date_str = datetime.now().strftime("%Y%m%d")
    key = hashlib.md5(f"{date_str}_HKGOLDEN_{HKG_USER_ID}_$API#Android_1_2^{topic_type}_{page}_N_N".encode()).hexdigest()
    
    params = {
        "s": key,
        "type": topic_type,
        "page": page,
        "pagesize": "50",
        "user_id": HKG_USER_ID,
        "pass": HKG_PASSWORD_HASH,
        "block": "Y",
        "sensormode": "N",
        "filterMode": "N",
        "hotOnly": "N",
        "returntype": "json"
    }
    url = f"{HKG_BASE_URL}topics.aspx?{urlencode(params)}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"HKG API 錯誤: {response.status_code}"}

# Streamlit 主程式
def main():
    st.title("LIHKG 和 HKG 分類帖子抓取測試")

    # LIHKG 測試區
    st.header("LIHKG 分類帖子")
    lihkg_cat_id = st.text_input("輸入 LIHKG 分類 ID (例如 1 表示最新)", "1")
    lihkg_page = st.number_input("LIHKG 頁數", min_value=1, value=1)
    if st.button("抓取 LIHKG 分類帖子"):
        data = get_lihkg_topic_list(lihkg_cat_id, lihkg_page)
        if "error" in data:
            st.error(data["error"])
        else:
            st.json(data)  # 顯示原始 JSON
            if "response" in data and "items" in data["response"]:
                for item in data["response"]["items"]:
                    st.write(f"標題: {item['title']}")
                    st.write(f"帖子 ID: {item['thread_id']}")
                    st.write(f"用戶: {item['user_nickname']}")
                    st.write("---")

    # HKG 測試區
    st.header("HKG 分類帖子")
    hkg_type = st.text_input("輸入 HKG 分類類型 (例如 BW 表示吹水台)", "BW")
    hkg_page = st.number_input("HKG 頁數", min_value=1, value=1)
    if st.button("抓取 HKG 分類帖子"):
        data = get_hkg_topic_list(hkg_type, hkg_page)
        if "error" in data:
            st.error(data["error"])
        else:
            st.json(data)  # 顯示原始 JSON
            if "Topics" in data:
                for topic in data["Topics"]:
                    st.write(f"標題: {topic['Title']}")
                    st.write(f"帖子 ID: {topic['Message']}")
                    st.write(f"用戶: {topic['Author']}")
                    st.write("---")

if __name__ == "__main__":
    main()
