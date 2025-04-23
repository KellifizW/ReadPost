"""
Streamlit 聊天介面模組，提供 LIHKGAt 數據查詢和顯示功能。
僅負責用戶交互、聊天記錄管理和速率限制狀態顯示。
主要函數：
- main：初始化應用，處理用戶輸入，渲染介面。
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
import nest_asyncio
import logging
from grok_processing import analyze_and_screen, stream_grok3_response, process_user_question

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器，設置為香港時區
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 自定義日誌格式器，將時間戳設為香港時區
class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(message)s")

# 防止重複添加處理器
if not logger.handlers:
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()  # 修正：使用 StreamHandler
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# 應用 asyncio 補丁
nest_asyncio.apply()

# 定義全局錯誤類別
class AppError(Exception):
    def __init__(self, message, user_message=None):
        self.message = message
        self.user_message = user_message or message
        super().__init__(self.message)

def validate_input(user_question):
    """
    驗證用戶輸入，確保長度有效。
    """
    if not user_question:
        return False, "輸入不能為空"
    if len(user_question) > 200:
        return False, "輸入過長，最多200個字"
    return True, ""

def suggest_queries():
    """
    顯示建議問題按鈕，引導用戶輸入。
    """
    suggestions = [
        "吹水台最近的搞笑話題",
        "熱門台的熱門新聞",
        "分析時事台情緒",
        "財經台最新帖子"
    ]
    st.markdown("**試試這些問題：**")
    for suggestion in suggestions:
        if st.button(suggestion):
            return suggestion
    return None

async def main():
    """
    主函數，初始化 Streamlit 應用，處理用戶輸入並渲染聊天介面。
    """
    # 設置 Streamlit 頁面配置
    st.set_page_config(page_title="LIHKG 聊天介面", layout="wide")
    st.title("LIHKG 聊天介面")

    # 初始化 session_state
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

    # 分類選擇
    cat_id_map = {
        "吹水台": {"id": 1, "order": "now"},
        "熱門台": {"id": 1, "order": "hot"},
        "時事台": {"id": 5, "order": "now"},
        "上班台": {"id": 14, "order": "now"},
        "財經台": {"id": 15, "order": "now"},
        "成人台": {"id": 29, "order": "now"},
        "創意台": {"id": 31, "order": "now"}
    }
    selected_cat = st.selectbox("選擇分類", options=list(cat_id_map.keys()), index=0)
    cat_id = str(cat_id_map[selected_cat]["id"])
    order = cat_id_map[selected_cat]["order"]
    st.write(f"當前討論區：{selected_cat}（排序：{'最新' if order == 'now' else '熱門'}）")

    # 記錄選單選擇
    logger.info(f"Selected category: {selected_cat}, cat_id: {cat_id}, order: {order}")

    # 顯示速率限制狀態
    st.markdown("#### 速率限制狀態")
    st.markdown(f"- 請求計數: {st.session_state.request_counter}")
    st.markdown(f"- 最後重置: {datetime.fromtimestamp(st.session_state.last_reset, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
    st.markdown(f"- 速率限制解除: {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S if st.session_state.rate_limit_until > time.time() else '無限制'}")

    # 顯示建議問題
    suggested_query = suggest_queries()
    if suggested_query:
        user_question = suggested_query
    else:
        user_question = st.chat_input("請輸入 LIHKG 話題或一般問題")

    # 顯示聊天記錄
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    # 用戶輸入
    if user_question and not st.session_state.awaiting_response:
        # 驗證輸入
        is_valid, error_message = validate_input(user_question)
        if not is_valid:
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.chat_history.append({"question": user_question, "answer": error_message})
            return
        
        logger.info(f"User query: {user_question}, category: {selected_cat}, cat_id: {cat_id}, order: {order}")
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.awaiting_response = True

        # 初始化進度條和狀態顯示
        status_text = st.empty()
        progress_bar = st.progress(0)

        # 進度回調函數
        def update_progress(message, progress):
            status_text.write(f"正在處理... {message}")
            progress_bar.progress(min(max(progress, 0.0), 1.0))

        try:
            update_progress("正在初始化", 0.0)

            # 檢查速率限制
            if time.time() < st.session_state.rate_limit_until:
                error_message = f"速率限制中，請在 {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} 後重試。"
                logger.warning(error_message)
                raise AppError(error_message, user_message=error_message)

            # 重置聊天記錄
            if "last_user_query" not in st.session_state:
                st.session_state.last_user_query = None
            if not st.session_state.last_user_query or len(set(user_question.split()).intersection(set(st.session_state.last_user_query.split()))) < 2:
                st.session_state.chat_history = [{"question": user_question, "answer": ""}]
                st.session_state.thread_cache = {}
                st.session_state.conversation_context = []
                st.session_state.last_user_query = user_question

            # 分析問題
            update_progress("正在分析問題意圖", 0.1)
            analysis = await analyze_and_screen(
                user_query=user_question,
                cat_name=selected_cat,
                cat_id=cat_id,
                conversation_context=st.session_state.conversation_context
            )
            logger.info(f"Analysis completed: intents={[i['type'] for i in analysis.get('intents', [])]}")

            # 處理問題
            update_progress("正在處理查詢", 0.2)
            result = await process_user_question(
                user_question=user_question,
                selected_cat=selected_cat,
                cat_id=cat_id,
                order=order,
                analysis=analysis,
                request_counter=st.session_state.request_counter,
                last_reset=st.session_state.last_reset,
                rate_limit_until=st.session_state.rate_limit_until,
                conversation_context=st.session_state.conversation_context,
                progress_callback=update_progress
            )

            # 更新速率限制
            st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
            st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
            st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)

            # 顯示回應
            response = ""
            with st.chat_message("assistant"):
                grok_container = st.empty()
                update_progress("正在生成回應", 0.9)
                logger.info(f"Starting stream_grok3_response for query: {user_question}, intents: {[i['type'] for i in analysis.get('intents', [])]}")
                async for chunk in stream_grok3_response(
                    user_query=user_question,
                    metadata=[{"thread_id": item["thread_id"], "title": item["title"], "no_of_reply": item.get("no_of_reply", 0), "last_reply_time": item.get("last_reply_time", "0"), "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0)} for item in result.get("thread_data", [])],
                    thread_data={item["thread_id"]: item for item in result.get("thread_data", [])},
                    processing=analysis,
                    selected_cat=selected_cat,
                    conversation_context=st.session_state.conversation_context,
                    needs_advanced_analysis=analysis.get("needs_advanced_analysis", False),
                    reason=analysis.get("reason", ""),
                    filters=analysis.get("filters", {})
                ):
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

        except AppError as e:
            logger.error(f"Application error: {e.message}")
            with st.chat_message("assistant"):
                st.error(e.user_message)
            st.session_state.chat_history.append({"question": user_question, "answer": e.user_message})
            update_progress("處理失敗", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
        except Exception as e:
            error_message = f"系統錯誤：{str(e)}。請稍後重試或聯繫支持。"
            logger.error(f"Unexpected error: {str(e)}")
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.chat_history.append({"question": user_question, "answer": error_message})
            update_progress("處理失敗", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
        finally:
            st.session_state.awaiting_response = False

if __name__ == "__main__":
    asyncio.run(main())
