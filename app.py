"""
Streamlit 聊天介面模組，提供 LIHKG 數據查詢和顯示功能。
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
import nest_asyncio
import logging
import json
from grok_processing import analyze_and_screen, stream_grok3_response, process_user_question

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(message)s")
logger.handlers.clear()
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 應用 asyncio 補丁
nest_asyncio.apply()

def validate_input(user_question):
    """
    驗證用戶輸入。
    """
    if not user_question:
        return False, "輸入不能為空"
    if len(user_question) < 5:
        return False, "輸入過短，至少5個字"
    if len(user_question) > 200:
        return False, "輸入過長，最多200個字"
    if not any(c.isalnum() for c in user_question):
        return False, "輸入需包含字母或數字"
    return True, ""

async def main():
    """
    主函數，初始化 Streamlit 應用。
    """
    st.set_page_config(page_title="LIHKG 聊天介面", layout="wide")
    st.title("LIHKG 聊天介面")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_cache" not in st.session_state:
        st.session_state.thread_cache = {}
    if "rate_limit_until" not in st.session_state:
        st.session_state.rate_limit_until = 0
    if "request_counter" not in st.session_state:
        st.session_state.request_counter = 0
    if "last_reset" not in st.session_state:
        st.session_state.last_reset = time.time()
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

    cat_id_map = {
        "吹水台": 1, "熱門台": 2, "時事台": 5, "上班台": 14,
        "財經台": 15, "成人台": 29, "創意台": 31
    }
    selected_cat = st.selectbox("選擇分類", options=list(cat_id_map.keys()), index=0)
    cat_id = str(cat_id_map[selected_cat])
    st.write(f"當前討論區：{selected_cat}")

    logger.info(f"Selected category: {selected_cat}, cat_id: {cat_id}")

    st.markdown("#### 速率限制狀態")
    st.markdown(f"- 請求計數: {st.session_state.request_counter}")
    st.markdown(f"- 最後重置: {datetime.fromtimestamp(st.session_state.last_reset, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
    st.markdown(f"- 速率限制解除: {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S if st.session_state.rate_limit_until > time.time() else '無限制'}")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    user_question = st.chat_input("請輸入 LIHKG 話題或一般問題")
    if user_question and not st.session_state.awaiting_response:
        is_valid, error_message = validate_input(user_question)
        if not is_valid:
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.chat_history.append({"question": user_question, "answer": error_message})
            return
        
        logger.info(f"User query: {user_question}, category: {selected_cat}, cat_id: {cat_id}")
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.awaiting_response = True

        status_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(message, progress):
            status_text.write(f"正在處理... {message}")
            progress_bar.progress(min(max(progress, 0.0), 1.0))

        try:
            update_progress("正在初始化", 0.0)

            if time.time() < st.session_state.rate_limit_until:
                error_message = f"速率限制中，請在 {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} 後重試。"
                logger.warning(error_message)
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.chat_history.append({"question": user_question, "answer": error_message})
                update_progress("處理失敗", 1.0)
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                st.session_state.awaiting_response = False
                return

            if "last_user_query" not in st.session_state:
                st.session_state.last_user_query = None
            if not st.session_state.last_user_query or len(set(user_question.split()).intersection(set(st.session_state.last_user_query.split()))) < 2:
                st.session_state.chat_history = [{"question": user_question, "answer": ""}]
                st.session_state.thread_cache = {}
                st.session_state.conversation_context = []
                st.session_state.last_user_query = user_question

            update_progress("正在分析問題意圖", 0.1)
            analysis = await analyze_and_screen(
                user_query=user_question,
                cat_name=selected_cat,
                cat_id=cat_id,
                conversation_context=st.session_state.conversation_context
            )
            logger.info(f"Analysis completed: intent={analysis.get('intent')}")

            update_progress("正在處理查詢", 0.2)
            result = await process_user_question(
                user_question=user_question,
                selected_cat=selected_cat,
                cat_id=cat_id,
                analysis=analysis,
                request_counter=st.session_state.request_counter,
                last_reset=st.session_state.last_reset,
                rate_limit_until=st.session_state.rate_limit_until,
                conversation_context=st.session_state.conversation_context,
                progress_callback=update_progress
            )

            st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
            st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
            st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)

            response = ""
            with st.chat_message("assistant"):
                grok_container = st.empty()
                update_progress("正在生成回應", 0.9)
                logger.info(f"Starting stream_grok3_response for query: {user_question}, intent: {analysis.get('intent')}")
                async for chunk in stream_grok3_response(
                    user_query=user_question,
                    metadata=[{
                        "thread_id": item["thread_id"],
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item.get("last_reply_time", "0"),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0)
                    } for item in result.get("thread_data", [])],
                    thread_data={item["thread_id"]: item for item in result.get("thread_data", [])},
                    intent=analysis.get("intent", "summarize_posts"),
                    selected_cat=selected_cat,
                    conversation_context=st.session_state.conversation_context,
                    filters=analysis.get("filters", {})
                ):
                    try:
                        parsed_chunk = json.loads(chunk)
                        if "error" in parsed_chunk:
                            response += parsed_chunk["error"]
                        elif analysis["intent"] == "summarize_posts":
                            response += f"**簡介**：{parsed_chunk.get('intro', '')}\n\n"
                            for item in parsed_chunk.get("analysis", []):
                                response += f"**標題**：{item.get('title', '')}\n"
                                response += f"**主題**：{item.get('theme', '')}\n"
                                response += f"**觀點**：{item.get('views', '')}\n"
                                response += f"**趨勢**：{item.get('trend', '')}\n\n"
                            response += f"**總結**：{parsed_chunk.get('summary', '')}"
                        elif analysis["intent"] == "list_titles":
                            for item in parsed_chunk.get("titles", []):
                                response += f"帖子 ID: {item.get('thread_id', '')} 標題: {item.get('title', '')}\n"
                            response = response.strip()
                        elif analysis["intent"] == "analyze_sentiment":
                            sentiments = parsed_chunk.get("sentiments", {})
                            response += f"**情緒分佈**：正面 {sentiments.get('positive', 0)}%，負面 {sentiments.get('negative', 0)}%，中立 {sentiments.get('neutral', 0)}%\n"
                            response += f"**原因**：{parsed_chunk.get('reasons', '')}"
                        elif analysis["intent"] == "fetch_dates":
                            response += f"**日期資料**：\n\n"
                            for item in parsed_chunk.get("dates", []):
                                response += f"**標題**：{item.get('title', '')}\n"
                                response += f"**最後回覆時間**：{item.get('last_reply_time', '')}\n"
                                response += f"**熱門回覆時間**：{item.get('top_reply_time', '')}\n\n"
                            response += f"**總結**：{parsed_chunk.get('summary', '')}"
                        elif analysis["intent"] == "general_query":
                            response += parsed_chunk.get("response", "")
                        else:
                            response += json.dumps(parsed_chunk, ensure_ascii=False)
                    except json.JSONDecodeError:
                        response += chunk
                    grok_container.markdown(response)
                if not response:
                    logger.warning(f"No response generated for query: {user_question}")
                    response = "無法生成回應，請稍後重試。"
                    grok_container.markdown(response)

            st.session_state.chat_history[-1]["answer"] = response
            st.session_state.conversation_context.append({"role": "user", "content": user_question})
            st.session_state.conversation_context.append({"role": "assistant", "content": response})
            update_progress("完成", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

        except Exception as e:
            error_message = f"處理失敗：{str(e)}"
            logger.error(f"Error processing query: {user_question}, error: {str(e)}")
            with st.chat_message("assistant"):
                st.markdown(error_message)
            st.session_state.chat_history.append({"question": user_question, "answer": error_message})
            update_progress("處理失敗", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
        finally:
            st.session_state.awaiting_response = False

if __name__ == "__main__":
    asyncio.run(main())