import streamlit as st
import asyncio
from datetime import datetime
import pytz
from data_processor import process_user_question
from grok3_client import get_grok3_response
import time

# 定義香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 使用 Streamlit 的 logger
logger = st.logger.get_logger(__name__)

async def chat_page():
    st.title("LIHKG 聊天介面")
    
    # 初始化 session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "topic_list_cache" not in st.session_state:
        st.session_state.topic_list_cache = {}
    if "rate_limit_until" not in st.session_state:
        st.session_state.rate_limit_until = 0
    if "request_counter" not in st.session_state:
        st.session_state.request_counter = 0
    if "last_reset" not in st.session_state:
        st.session_state.last_reset = time.time()
    
    # 分類映射
    cat_id_map = {
        "吹水台": 1,
        "熱門台": 2,
        "時事台": 5,
        "上班台": 14,
        "財經台": 15,
        "成人台": 29,
        "創意台": 31
    }
    
    # 顯示聊天歷史
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
    
    # 聊天輸入框
    user_question = st.chat_input("請輸入您想查詢的 LIHKG 話題（例如：吹水台有哪些搞笑話題？）")
    
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.spinner("正在處理您的問題..."):
            try:
                # 儲存最新問題以供 Grok 3 使用
                st.session_state.last_user_query = user_question
                
                # 解析用戶問題並抓取數據
                result = await process_user_question(
                    user_question,
                    cat_id_map=cat_id_map,
                    request_counter=st.session_state.request_counter,
                    last_reset=st.session_state.last_reset,
                    rate_limit_until=st.session_state.rate_limit_until
                )
                
                # 更新 session state
                st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
                st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
                st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
                
                # 檢查速率限制
                if time.time() < st.session_state.rate_limit_until:
                    answer = f"API 速率限制中，請在 {datetime.fromtimestamp(st.session_state.rate_limit_until)} 後重試。"
                    logger.warning(f"速率限制: 問題={user_question}, 需等待至 {datetime.fromtimestamp(st.session_state.rate_limit_until)}")
                else:
                    items = result.get("items", [])
                    rate_limit_info = result.get("rate_limit_info", [])
                    selected_cat = result.get("selected_cat", "吹水台")  # 預設吹水台
                    max_pages = result.get("max_pages", 1)  # 預設抓取 1 頁
                    
                    # 準備元數據
                    metadata_list = [
                        {
                            "thread_id": item["thread_id"],
                            "title": item["title"],
                            "no_of_reply": item["no_of_reply"],
                            "last_reply_time": (
                                datetime.fromtimestamp(int(item.get("last_reply_time", 0)), tz=HONG_KONG_TZ)
                                .strftime("%Y-%m-%d %H:%M:%S")
                                if item.get("last_reply_time")
                                else "未知"
                            ),
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0),
                        }
                        for item in items
                    ]
                    
                    # 篩選回覆數 ≥ 125 的帖子
                    min_replies = 125
                    filtered_items = [item for item in metadata_list if item["no_of_reply"] >= min_replies]
                    
                    # 構建回應
                    if filtered_items:
                        answer = f"### 來自 {selected_cat} 的話題（回覆數 ≥ {min_replies}）：\n"
                        for item in filtered_items:
                            answer += (
                                f"- 帖子 ID: {item['thread_id']}，標題: {item['title']}，"
                                f"回覆數: {item['no_of_reply']}，最後回覆時間: {item['last_reply_time']}\n"
                            )
                        answer += f"\n共找到 {len(filtered_items)} 篇符合條件的帖子。"
                        logger.info(f"成功處理: 問題={user_question}, 分類={selected_cat}, 帖子數={len(filtered_items)}")
                    else:
                        answer = f"在 {selected_cat} 中未找到回覆數 ≥ {min_replies} 的帖子。"
                        logger.warning(f"無符合條件的帖子: 問題={user_question}, 分類={selected_cat}")
                    
                    # 顯示速率限制信息（若有）
                    if rate_limit_info:
                        answer += "\n#### 速率限制信息：\n"
                        for info in rate_limit_info:
                            answer += f"- {info}\n"
                    
                    # 可選：調用 Grok 3 增強回應
                    try:
                        grok_context = f"問題: {user_question}\n帖子數據:\n{answer}"
                        grok_response = await get_grok3_response(grok_context)
                        if grok_response and not grok_response.startswith("錯誤:"):
                            answer += f"\n#### Grok 3 建議：\n{grok_response}"
                    except Exception as e:
                        logger.warning(f"Grok 3 增強失敗: 問題={user_question}, 錯誤={str(e)}")
                
                # 顯示回應
                with st.chat_message("assistant"):
                    st.markdown(answer)
                
                # 更新聊天歷史
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer
                })
                
            except Exception as e:
                error_message = f"處理失敗，原因：{str(e)}"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                logger.error(f"處理錯誤: 問題={user_question}, 錯誤={str(e)}")
