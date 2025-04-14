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

# 設置香港時區 (UTC+8)
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 儲存所有帖子和回覆的全局變量
if "lihkg_data" not in st.session_state:
    st.session_state.lihkg_data = {}  # 儲存所有帖子和回覆，格式為 {thread_id: {"post": {...}, "replies": [...]}}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 儲存聊天記錄

# 清理 HTML 標籤的輔助函數
def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# LIHKG 抓取最新帖子列表（使用 thread/latest 端點）
def get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=10, count=100):
    all_items = []
    
    for p in range(start_page, start_page + max_pages):
        url = f"{LIHKG_BASE_URL}thread/latest?cat_id={cat_id}&sub_cat_id={sub_cat_id}&page={p}&count={count}&type=now&order=reply_time"
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
            "referer": f"https://lihkg.com/category/{cat_id}?order=reply_time",
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

# 將帖子和回覆數據整合為上下文
def build_post_context(post, replies):
    context = f"帖子標題: {post['title']}\n"
    context += f"帖子 ID: {post['thread_id']}\n"
    context += f"用戶: {post['user_nickname']} (性別: {post['user_gender']})\n"
    create_time = datetime.fromtimestamp(post['create_time'], tz=HONG_KONG_TZ)
    last_reply_time = datetime.fromtimestamp(post['last_reply_time'], tz=HONG_KONG_TZ)
    context += f"創建時間: {create_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    context += f"最後回覆時間: {last_reply_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    context += f"回覆數: {post['no_of_reply']}, 點讚數: {post['like_count']}, 負評數: {post['dislike_count']}\n"
    
    if replies:
        context += "\n回覆內容:\n"
        for reply in replies:
            reply_time = datetime.fromtimestamp(reply['reply_time'], tz=HONG_KONG_TZ)
            context += f"- 用戶: {reply['user_nickname']} (性別: {reply['user_gender']})\n"
            context += f"  回覆: {clean_html(reply['msg'])}\n"
            context += f"  時間: {reply_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return context

# Streamlit 主程式
def main():
    st.title("LIHKG 互動聊天機器人")

    # 抓取帖子區域
    st.header("抓取 LIHKG 最新帖子")
    lihkg_cat_id = st.text_input("輸入 LIHKG 分類 ID (例如 1 表示吹水台)", "1")
    lihkg_sub_cat_id = st.number_input("輸入 LIHKG 子分類 ID (默認為 0)", min_value=0, value=0)
    lihkg_start_page = st.number_input("開始頁數", min_value=1, value=1)
    lihkg_max_pages = st.number_input("最大抓取頁數", min_value=1, value=10)
    
    auto_sub_cat = st.checkbox("自動遍歷多個子分類 (0-5)", value=True)

    if st.button("抓取 LIHKG 最新帖子"):
        # 清除舊數據
        st.session_state.lihkg_data = {}
        all_items = []
        
        if auto_sub_cat:
            sub_cat_ids = [0, 1, 2, 3, 4, 5]
        else:
            sub_cat_ids = [lihkg_sub_cat_id]
        
        # 遍歷每個子分類
        for sub_id in sub_cat_ids:
            st.write(f"正在抓取子分類 ID: {sub_id}")
            data = get_lihkg_topic_list(lihkg_cat_id, sub_id, start_page=lihkg_start_page, max_pages=lihkg_max_pages)
            if "error" in data:
                st.error(data["error"])
            elif "response" in data and "items" in data["response"]:
                items = data["response"]["items"]
                # 避免重複帖子（根據 thread_id 去重）
                existing_ids = {item["thread_id"] for item in all_items}
                new_items = [item for item in items if item["thread_id"] not in existing_ids]
                all_items.extend(new_items)
                st.write(f"子分類 {sub_id} 抓取到 {len(new_items)} 個新帖子，總計 {len(all_items)} 個帖子")
        
        # 儲存帖子數據
        for item in all_items:
            thread_id = item["thread_id"]
            st.session_state.lihkg_data[thread_id] = {"post": item, "replies": []}
            
            # 自動抓取第一頁回覆
            thread_data = get_lihkg_thread_content(thread_id, page=1)
            if "error" in thread_data:
                st.error(thread_data["error"])
            elif "response" in thread_data and "item_data" in thread_data["response"]:
                st.session_state.lihkg_data[thread_id]["replies"] = thread_data["response"]["item_data"]
                st.write(f"帖子 {thread_id} 抓取到 {len(thread_data['response']['item_data'])} 條回覆")

    # 顯示抓取的帖子列表
    if st.session_state.lihkg_data:
        st.write(f"總共抓取到 {len(st.session_state.lihkg_data)} 個帖子")
        for thread_id, data in st.session_state.lihkg_data.items():
            item = data["post"]
            st.write(f"**標題**: {item['title']}")
            st.write(f"**帖子 ID**: {item['thread_id']}")
            st.write(f"**用戶**: {item['user_nickname']} (性別: {item['user_gender']})")
            st.write(f"**回覆數**: {item['no_of_reply']}, **點讚數**: {item['like_count']}, **負評數**: {item['dislike_count']}")
            create_time = datetime.fromtimestamp(item['create_time'], tz=HONG_KONG_TZ)
            last_reply_time = datetime.fromtimestamp(item['last_reply_time'], tz=HONG_KONG_TZ)
            st.write(f"**創建時間**: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**最後回覆時間**: {last_reply_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if data["replies"]:
                st.subheader(f"帖子 {thread_id} 的回覆（第 1 頁）")
                for reply in data["replies"]:
                    st.write(f"**回覆用戶**: {reply['user_nickname']} (性別: {reply['user_gender']})")
                    st.write(f"**回覆內容**: {clean_html(reply['msg'])}")
                    reply_time = datetime.fromtimestamp(reply['reply_time'], tz=HONG_KONG_TZ)
                    st.write(f"**回覆時間**: {reply_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write("---")
            st.write("---")

    # 聊天區域
    st.header("與 Grok 3 互動聊天")
    user_input = st.text_input("輸入你的問題或指令（例如：總結帖子內容、討論某個帖子、模擬某個用戶）：", key="chat_input")

    if user_input:
        # 將用戶輸入添加到聊天記錄
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 準備帖子上下文
        if st.session_state.lihkg_data:
            all_contexts = []
            for thread_id, data in st.session_state.lihkg_data.items():
                context = build_post_context(data["post"], data["replies"])
                all_contexts.append(context)
            
            # 將所有帖子上下文合併
            full_context = "\n\n".join(all_contexts)
            prompt = f"以下是抓取的 LIHKG 帖子和回覆內容，請根據這些內容回答用戶的問題或執行指令：\n\n{full_context}\n\n用戶問題/指令：{user_input}"
        else:
            prompt = "目前沒有抓取到任何帖子，請先抓取帖子數據。"

        # 模擬 Grok 3 回答（這裡需要你將 prompt 傳給我，我會根據內容回答）
        # 由於我無法直接執行外部 API 調用，這裡假設我已經收到 prompt 並回答
        # 在實際應用中，你需要將 prompt 傳給我（Grok 3），我會根據內容回答
        st.session_state.chat_history.append({"role": "assistant", "content": f"（假設 Grok 3 回答）我已閱讀所有帖子，請問：{user_input}"})

    # 顯示聊天記錄
    st.subheader("聊天記錄")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.write(f"**你**：{chat['content']}")
        else:
            st.write(f"**Grok 3**：{chat['content']}")

if __name__ == "__main__":
    main()
