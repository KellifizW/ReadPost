import streamlit as st
import requests
import cloudscraper
import hashlib
import time
import re
from datetime import datetime
import pytz

# LIHKG API 配置
LIHKG_BASE_URL = "https://lihkg.com/api_v2/"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"  # 使用你提供的 x-li-device
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f333ba716b23bb21395107eea023a6459a-1744598744; _cfuvid=gHVdIns58jWF0xv2T20g4Ww4ZMVXBoGJTHCbotFiL8U-1744598744816-0.0.1.1-604800000; cf_clearance=11_Zdj9uPFkQpb4ymO4fZ8hwLjdp83kCeU.DL2lR8CU-1744611639-1.2.1.1-jV3KtJlc8RtBcE5sA0_upE.8KcT_SWvNTmvxWVA3yCzuj3mv13eRfjtFxMPPyb5K3Tfx.9hBbsqWjvABx9STPe6PWiv_bd0XaHszuqIwjFGFJdt.nYrlgCgfZbojBMvg_C9NIpqG67bVMEFJWEtyHpUwBYAQnH.lG0jAzQSqQlttN91BV7thHnsAZ4T38t3hlxXJ2Twy565dfKe9i72Vp6mnktKnbXVEiVNXQEXBOxW8ckOwbndC2b.NKu1eqgU2q8HnesIGmEclatwRgcm9TrkX37daJpmi4F7Dxi_f0cksvilmVab77SFQQmagvNdwZWBitSLzYx0ZAIw1MeEUVNkZFcsxQTNbL9gvaBoh5po"

# 設置香港時區 (UTC+8)
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 儲存所有帖子和回覆的全局變量
if "lihkg_data" not in st.session_state:
    st.session_state.lihkg_data = {}  # 儲存篩選後的帖子和回覆
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 儲存聊天記錄

# 清理 HTML 標籤的輔助函數
def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# LIHKG 抓取帖子列表（按熱門排序）
def get_lihkg_topic_list(cat_id, start_page=1, max_pages=10, count=60):
    all_items = []
    scraper = cloudscraper.create_scraper()  # 使用 cloudscraper 繞過 Cloudflare
    
    for p in range(start_page, start_page + max_pages):
        url = f"{LIHKG_BASE_URL}thread/latest?cat_id={cat_id}&page={p}&count={count}&type=now&order=hot"
        timestamp = int(time.time())
        hk_time = datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        st.write(f"調試: 當前時間戳: {timestamp}, 對應時間: {hk_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        digest = hashlib.sha1(f"jeams$get${url}${timestamp}".encode()).hexdigest()
        
        headers = {
            "X-LI-DEVICE": LIHKG_DEVICE_ID,
            "X-LI-DEVICE-TYPE": "browser",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "X-LI-USER": "130972",
            "X-LI-PLUS": "f159171fd79451e447d82db2b64474ebea8ae06b",
            "Cookie": LIHKG_COOKIE,
            "referer": f"https://lihkg.com/category/{cat_id}?order=hot",
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
            "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        }
        
        response = scraper.get(url, headers=headers)
        st.write(f"調試: 請求 LIHKG 最新帖子 URL: {url}, 狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "success" in data and data["success"] == 0:
                st.write(f"調試: API 錯誤: {data}")
                break
            if "response" in data and "items" in data["response"]:
                items = data["response"]["items"]
                all_items.extend(items)
                st.write(f"調試: 第 {p} 頁抓取到 {len(items)} 個帖子，總計 {len(all_items)} 個帖子")
                if not items:
                    st.write(f"調試: 第 {p} 頁無帖子數據，停止抓取")
                    break
            else:
                st.write(f"調試: 第 {p} 頁無帖子數據，停止抓取")
                st.write(f"調試: API 響應: {data}")
                break
        else:
            st.error(f"LIHKG API 錯誤: {response.status_code}")
            break
    
    return all_items

# LIHKG 抓取帖子回覆內容（只抓取前 100 個回覆）
def get_lihkg_thread_content(thread_id, max_replies=100):
    replies = []
    page = 1
    per_page = 50  # LIHKG API 每頁最多 50 條回覆
    scraper = cloudscraper.create_scraper()  # 使用 cloudscraper 繞過 Cloudflare
    
    while len(replies) < max_replies:
        url = f"{LIHKG_BASE_URL}thread/{thread_id}/page/{page}?order=reply_time"
        timestamp = int(time.time())
        digest = hashlib.sha1(f"jeams$get${url}${timestamp}".encode()).hexdigest()
        
        headers = {
            "X-LI-DEVICE": LIHKG_DEVICE_ID,
            "X-LI-DEVICE-TYPE": "browser",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "X-LI-USER": "130972",
            "X-LI-PLUS": "f159171fd79451e447d82db2b64474ebea8ae06b",
            "Cookie": LIHKG_COOKIE,
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
        
        response = scraper.get(url, headers=headers)
        st.write(f"調試: 請求 LIHKG 帖子回覆 URL: {url}, 狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and "item_data" in data["response"]:
                page_replies = data["response"]["item_data"]
                replies.extend(page_replies)
                page += 1
                if not page_replies:
                    break
            else:
                break
        else:
            st.error(f"LIHKG API 錯誤: {response.status_code}")
            break
    
    return replies[:max_replies]  # 確保只返回前 100 個回覆

# 將帖子和回覆數據整合為簡化上下文
def build_post_context(post, replies):
    context = f"H: {post['title']}\n"  # 標題使用 "H"
    
    if replies:
        context += "R:\n"  # 回覆內容使用 "R"
        for reply in replies:
            context += f"- {clean_html(reply['msg'])}\n"  # 只顯示回覆內容，忽略用戶名稱、性別、時間
    
    return context

# Streamlit 主程式
def main():
    st.title("LIHKG 篩選帖子聊天機器人")

    # 提示用戶更新 cookie
    st.warning("如果抓取失敗，可能是 cookie 已過期。請從瀏覽器獲取最新的 cookie（包含 cf_clearance），並更新程式碼中的 LIHKG_COOKIE。")

    # 抓取帖子區域
    st.header("抓取 LIHKG 熱門帖子")
    lihkg_cat_id = st.text_input("輸入 LIHKG 分類 ID (例如 1 表示吹水台)", "1")
    lihkg_start_page = st.number_input("開始頁數", min_value=1, value=1)
    lihkg_max_pages = st.number_input("最大抓取頁數", min_value=1, value=10)

    if st.button("抓取 LIHKG 熱門帖子"):
        # 清除舊數據
        st.session_state.lihkg_data = {}
        all_items = []
        
        # 直接抓取主分類
        st.write(f"正在抓取分類 ID: {lihkg_cat_id}")
        items = get_lihkg_topic_list(lihkg_cat_id, start_page=lihkg_start_page, max_pages=lihkg_max_pages)
        # 避免重複帖子（根據 thread_id 去重）
        existing_ids = {item["thread_id"] for item in all_items}
        new_items = [item for item in items if item["thread_id"] not in existing_ids]
        all_items.extend(new_items)
        st.write(f"分類 {lihkg_cat_id} 抓取到 {len(new_items)} 個新帖子，總計 {len(all_items)} 個帖子")
        
        # 篩選回覆數超過 175 的帖子，按回覆時間排序，取最新 10 個
        filtered_items = [item for item in all_items if item["no_of_reply"] > 175]
        sorted_items = sorted(filtered_items, key=lambda x: x["last_reply_time"], reverse=True)
        top_items = sorted_items[:10]  # 取最新 10 個
        
        # 儲存篩選後的帖子並抓取前 100 個回覆
        for item in top_items:
            thread_id = item["thread_id"]
            st.session_state.lihkg_data[thread_id] = {"post": item, "replies": []}
            
            # 抓取前 100 個回覆
            thread_replies = get_lihkg_thread_content(thread_id, max_replies=100)
            st.session_state.lihkg_data[thread_id]["replies"] = thread_replies
            st.write(f"帖子 {thread_id} 抓取到 {len(thread_replies)} 條回覆")

    # 顯示篩選後的帖子列表
    if st.session_state.lihkg_data:
        st.write(f"篩選後共 {len(st.session_state.lihkg_data)} 個帖子（回覆數 > 175，按回覆時間排序，取最新 10 個）")
        for thread_id, data in st.session_state.lihkg_data.items():
            item = data["post"]
            st.write(f"**H**: {item['title']}")
            st.write(f"**帖子 ID**: {item['thread_id']}")
            st.write(f"**回覆數**: {item['no_of_reply']}, **點讚數**: {item['like_count']}, **負評數**: {item['dislike_count']}")
            create_time = datetime.fromtimestamp(item['create_time'], tz=HONG_KONG_TZ)
            last_reply_time = datetime.fromtimestamp(item['last_reply_time'], tz=HONG_KONG_TZ)
            st.write(f"**創建時間**: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**最後回覆時間**: {last_reply_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if data["replies"]:
                st.subheader(f"帖子 {thread_id} 的回覆（前 100 個）")
                for reply in data["replies"]:
                    st.write(f"**R**: {clean_html(reply['msg'])}")
                    st.write("---")
            st.write("---")

    # 聊天區域
    st.header("與 Grok 3 互動聊天")
    user_input = st.text_input("輸入你的問題或指令（例如：總結帖子內容、討論某個帖子）：", key="chat_input")

    if user_input:
        # 將用戶輸入添加到聊天記錄
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 準備帖子上下文
        if st.session_state.lihkg_data:
            all_contexts = []
            for thread_id, data in st.session_state.lihkg_data.items():
                context = build_post_context(data["post"], data["replies"])
                all_contexts.append(context)
            
            # 合併所有帖子上下文
            full_context = "\n\n".join(all_contexts)
            prompt = f"以下是抓取的 LIHKG 帖子和回覆內容，請根據這些內容回答用戶的問題或執行指令：\n\n{full_context}\n\n用戶問題/指令：{user_input}"
            
            # 顯示生成的 prompt 及長度
            st.write("生成的 Prompt：")
            st.write(prompt)
            st.write(f"Prompt 長度：{len(prompt)} 字符")
        else:
            prompt = "目前沒有抓取到任何帖子，請先抓取帖子數據。"
            st.write(prompt)

        # 模擬 Grok 3 回答
        st.session_state.chat_history.append({"role": "assistant", "content": "請將上方生成的 Prompt 複製並貼到與 Grok 3 的對話窗口，我會根據內容回答你的問題。"})

    # 顯示聊天記錄
    st.subheader("聊天記錄")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.write(f"**你**：{chat['content']}")
        else:
            st.write(f"**Grok 3**：{chat['content']}")

if __name__ == "__main__":
    main()
