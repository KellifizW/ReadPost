"""
Streamlit 聊天介面，提供 LIHKG 數據查詢功能。
負責用戶交互、聊天記錄管理和速率限制顯示。
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
from streamlit.components.v1 import html
from grok_processing import analyze_and_screen, stream_grok3_response, process_user_question
import logging

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌（簡化版）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# 應用 asyncio 補丁
import nest_asyncio
nest_asyncio.apply()

def validate_input(user_query):
    """驗證用戶輸入"""
    if not user_query:
        return False, "輸入不能為空"
    if len(user_query) < 3:
        return False, "輸入過短，至少3個字"
    if len(user_query) > 200:
        return False, "輸入過長，最多200個字"
    return True, ""

def render_copy_button(content, key):
    """渲染複製按鈕"""
    escaped_content = content.replace("`", "\\`").replace("\n", "\\n")
    html_code = f"""
    <button onclick="navigator.clipboard.writeText(`{escaped_content}`)"
            title="複製回應"
            style="border: none; background: none; cursor: pointer; font-size: 20px;">
        📋
    </button>
    """
    html(html_code, height=30)

def render_new_conversation_button():
    """渲染新對話按鈕"""
    html_code = """
    <button onclick="window.location.reload()"
            title="開始新對話"
            style="border: none; background: none; cursor: pointer; font-size: 20px;">
        🆕
    </button>
    """
    html(html_code, height=30)

async def main():
    """主函數，初始化 Streamlit 應用"""
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
    if "context_timestamps" not in st.session_state:
        st.session_state.context_timestamps = []
    if "last_selected_cat" not in st.session_state:
        st.session_state.last_selected_cat = None

    logger.info(f"Page reloaded, last_selected_cat: {st.session_state.get('last_selected_cat', 'None')}")

    # 分類選擇
    cat_id_map = {
        "吹水台": 1, "熱門台": 2, "時事台": 5, "上班台": 14,
        "財經台": 15, "成人台": 29, "創意台": 31
    }

    selected_cat = st.selectbox("選擇分類", options=list(cat_id_map.keys()), index=0)
    cat_id = str(cat_id_map[selected_cat])

    # 清理歷史記錄（分類變化時）
    if st.session_state.last_selected_cat != selected_cat:
        st.session_state.chat_history = []
        st.session_state.conversation_context = []
        st.session_state.context_timestamps = []
        st.session_state.thread_cache = {}
        st.session_state.last_user_query = None
        logger.info(f"Category changed to {selected_cat}, cleared history")
        st.session_state.last_selected_cat = selected_cat

    st.write(f"當前討論區：{selected_cat}")

    # 新對話按鈕
    st.markdown("#### 操作")
    render_new_conversation_button()

    # 顯示速率限制
    st.markdown("#### 速率限制狀態")
    st.markdown(f"- 請求計數: {st.session_state.request_counter}")
    st.markdown(f"- 最後重置: {datetime.fromtimestamp(st.session_state.last_reset, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
    st.markdown(f"- 速率限制解除: {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.rate_limit_until > time.time() else '無限制'}")

    # 顯示聊天記錄
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(chat["answer"])
            with col2:
                render_copy_button(chat["answer"], key=f"copy_{idx}")

    # 用戶輸入
    user_query = st.chat_input("請輸入 LIHKG 話題或問題")
    if user_query and not st.session_state.awaiting_response:
        is_valid, error_message = validate_input(user_query)
        if not is_valid:
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.chat_history.append({"question": user_query, "answer": error_message})
            return

        logger.info(f"User query: {user_query}, category: {selected_cat}")
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.awaiting_response = True

        # 清理過舊上下文
        current_time = time.time()
        valid_context = []
        valid_timestamps = []
        for msg, ts in zip(st.session_state.conversation_context, st.session_state.context_timestamps):
            if current_time - ts < 3600:
                valid_context.append(msg)
                valid_timestamps.append(ts)
        st.session_state.conversation_context = valid_context[:20]
        st.session_state.context_timestamps = valid_timestamps[:20]

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
                st.session_state.chat_history.append({"question": user_query, "answer": error_message})
                update_progress("處理失敗", 1.0)
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                st.session_state.awaiting_response = False
                return

            if "last_user_query" not in st.session_state:
                st.session_state.last_user_query = None
            if not st.session_state.last_user_query or len(set(user_query.split()).intersection(set(st.session_state.last_user_query.split()))) < 2:
                st.session_state.chat_history = [{"question": user_query, "answer": ""}]
                st.session_state.thread_cache = {}
                st.session_state.last_user_query = user_query

            update_progress("正在分析問題", 0.1)
            analysis = await analyze_and_screen(
                user_query=user_query,
                cat_name=selected_cat,
                cat_id=cat_id,
                conversation_context=st.session_state.conversation_context
            )

            update_progress("正在處理查詢", 0.2)
            result = await process_user_question(
                user_query=user_query,
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
                col1, col2 = st.columns([0.95, 0.05])
                with col1:
                    grok_container = st.empty()
                with col2:
                    copy_container = st.empty()
                update_progress("正在生成回應", 0.9)
                async for chunk in stream_grok3_response(
                    user_query=user_query,
                    metadata=[{"thread_id": item["thread_id"], "title": item["title"], "no_of_reply": item.get("no_of_reply", 0), "last_reply_time": item.get("last_reply_time", "0"), "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0)} for item in result.get("thread_data", [])],
                    thread_data={item["thread_id"]: item for item in result.get("thread_data", [])},
                    processing=analysis.get("processing", "general"),
                    selected_cat=selected_cat,
                    conversation_context=st.session_state.conversation_context,
                    needs_advanced_analysis=analysis.get("needs_advanced_analysis", False),
                    reason=analysis.get("reason", ""),
                    filters=analysis.get("filters", {}),
                    cat_id=cat_id
                ):
                    response += chunk
                    grok_container.markdown(response)
                if not response:
                    response = "無法生成回應，請稍後重試。"
                    grok_container.markdown(response)
                copy_container.empty()
                render_copy_button(response, key=f"copy_new_{len(st.session_state.chat_history)}")

            st.session_state.chat_history[-1]["answer"] = response
            st.session_state.conversation_context.append({"role": "user", "content": user_query})
            st.session_state.conversation_context.append({"role": "assistant", "content": response})
            st.session_state.context_timestamps.append(time.time())
            st.session_state.context_timestamps.append(time.time())
            st.session_state.conversation_context = st.session_state.conversation_context[-20:]
            st.session_state.context_timestamps = st.session_state.context_timestamps[-20:]
            update_progress("完成", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

        except Exception as e:
            error_message = f"處理失敗：{str(e)}"
            logger.error(f"Error processing query: {user_query}, error: {str(e)}")
            with st.chat_message("assistant"):
                st.markdown(error_message)
            st.session_state.chat_history.append({"question": user_query, "answer": error_message})
            update_progress("處理失敗", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
        finally:
            st.session_state.awaiting_response = False

if __name__ == "__main__":
    asyncio.run(main())
