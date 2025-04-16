import streamlit as st
import asyncio
from datetime import datetime
import pytz
from data_processor import process_user_question
from grok3_client import stream_grok3_response
import time

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = st.logger.get_logger(__name__)

async def chat_page():
    st.title("LIHKG 聊天介面")
    
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
    
    cat_id_map = {
        "吹水台": 1,
        "熱門台": 2,
        "時事台": 5,
        "上班台": 14,
        "財經台": 15,
        "成人台": 29,
        "創意台": 31
    }
    
    selected_cat = st.selectbox("選擇分類", options=list(cat_id_map.keys()), index=0)
    
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
    
    user_question = st.chat_input("請輸入您想查詢的 LIHKG 話題（例如：有哪些搞笑話題？）")
    
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.spinner("正在處理您的問題..."):
            try:
                st.session_state.last_user_query = user_question
                
                result = await process_user_question(
                    user_question,
                    cat_id_map=cat_id_map,
                    selected_cat=selected_cat,
                    request_counter=st.session_state.request_counter,
                    last_reset=st.session_state.last_reset,
                    rate_limit_until=st.session_state.rate_limit_until
                )
                
                st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
                st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
                st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
                
                thread_data = result.get("thread_data", [])
                rate_limit_info = result.get("rate_limit_info", [])
                question_cat = result.get("selected_cat", selected_cat)
                
                if not thread_data:
                    answer = f"在 {question_cat} 中未找到回覆數 ≥ 125 的帖子。"
                    if rate_limit_info:
                        answer += "\n#### 錯誤信息：\n" + "\n".join(f"- {info}" for info in rate_limit_info)
                        if "重試失敗" in str(rate_limit_info):
                            answer += "\n所有代理不可用，請檢查 PROXY_LIST 配置或稍後重試。"
                    logger.warning(f"無有效帖子數據: 問題={user_question}, 分類={question_cat}, 錯誤={rate_limit_info}")
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
                    return
                
                answer = f"### 來自 {question_cat} 的話題（回覆數 ≥ 125）：\n"
                failed_threads = []
                for item in thread_data:
                    if not item["replies"] and item["no_of_reply"] >= 125:
                        failed_threads.append(item["thread_id"])
                    answer += (
                        f"- 帖子 ID: {item['thread_id']}，標題: {item['title']}，"
                        f"回覆數: {item['no_of_reply']}，"
                        f"最後回覆時間: {datetime.fromtimestamp(int(item['last_reply_time']), tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S') if item['last_reply_time'] else '未知'}，"
                        f"正評: {item['like_count']}，負評: {item['dislike_count']}\n"
                    )
                    if item["replies"]:
                        answer += "  回覆:\n"
                        for idx, reply in enumerate(item["replies"], 1):
                            answer += (
                                f"    - 回覆 {idx}: {reply['msg'][:100]}"
                                f"{'...' if len(reply['msg']) > 100 else ''} "
                                f"(正評: {reply['like_count']}, 負評: {reply['dislike_count']})\n"
                            )
                answer += f"\n共找到 {len(thread_data)} 篇符合條件的帖子。"
                if failed_threads:
                    answer += f"\n警告：以下帖子無法獲取回覆（可能因錯誤 998）：{', '.join(map(str, failed_threads))}。"
                
                if rate_limit_info:
                    answer += "\n#### 錯誤信息：\n" + "\n".join(f"- {info}" for info in rate_limit_info)
                
                logger.info(f"成功處理: 問題={user_question}, 分類={question_cat}, 帖子數={len(thread_data)}")
                
                try:
                    grok_context = f"問題: {user_question}\n帖子數據:\n{answer}"
                    if len(grok_context) > 7000:
                        logger.warning(f"上下文接近限制: 字元數={len(grok_context)}，縮減回覆數據")
                        for item in thread_data:
                            item["replies"] = item["replies"][:1]
                        answer = f"### 來自 {question_cat} 的話題（回覆數 ≥ 125）：\n"
                        for item in thread_data:
                            answer += (
                                f"- 帖子 ID: {item['thread_id']}，標題: {item['title']}，"
                                f"回覆數: {item['no_of_reply']}，"
                                f"最後回覆時間: {datetime.fromtimestamp(int(item['last_reply_time']), tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S') if item['last_reply_time'] else '未知'}，"
                                f"正評: {item['like_count']}，負評: {item['dislike_count']}\n"
                            )
                            if item["replies"]:
                                answer += "  回覆:\n"
                                for idx, reply in enumerate(item["replies"], 1):
                                    answer += (
                                        f"    - 回覆 {idx}: {reply['msg'][:100]}"
                                        f"{'...' if len(reply['msg']) > 100 else ''} "
                                        f"(正評: {reply['like_count']}, 負評: {reply['dislike_count']})\n"
                                    )
                        answer += f"\n共找到 {len(thread_data)} 篇符合條件的帖子。"
                        if failed_threads:
                            answer += f"\n警告：以下帖子無法獲取回覆（可能因錯誤 998）：{', '.join(map(str, failed_threads))}。"
                        grok_context = f"問題: {user_question}\n帖子數據:\n{answer}"
                        logger.info(f"縮減後上下文: 字元數={len(grok_context)}")
                    
                    grok_response = ""
                    with st.chat_message("assistant"):
                        grok_container = st.empty()
                        async for chunk in stream_grok3_response(grok_context):
                            grok_response += chunk
                            grok_container.markdown(grok_response)
                    if grok_response and not grok_response.startswith("錯誤:"):
                        final_answer = grok_response
                    else:
                        final_answer = grok_response or "無法獲取 Grok 3 建議。"
                except Exception as e:
                    logger.warning(f"Grok 3 增強失敗: 問題={user_question}, 錯誤={str(e)}")
                    final_answer = f"錯誤：無法獲取 Grok 3 建議，原因：{str(e)}"
                    if failed_threads:
                        final_answer += f"\n警告：以下帖子無法獲取回覆（可能因錯誤 998）：{', '.join(map(str, failed_threads))}。"
                
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": final_answer
                })
                
            except Exception as e:
                error_message = f"處理失敗，原因：{str(e)}"
                if result.get("rate_limit_info"):
                    error_message += "\n#### 錯誤信息：\n" + "\n".join(f"- {info}" for info in result["rate_limit_info"])
                    if "重試失敗" in str(result["rate_limit_info"]):
                        error_message += "\n所有代理不可用，請檢查 PROXY_LIST 配置或稍後重試。"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                logger.error(f"處理錯誤: 問題={user_question}, 錯誤={str(e)}")
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": error_message
                })
