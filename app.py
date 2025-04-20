"""
Streamlit 聊天介面模組，提供 LIHKG 數據查詢和顯示功能。
負責用戶交互、聊天記錄管理和速率限制狀態顯示。
"""

import streamlit as st
import asyncio
import time
import json
import os
from datetime import datetime
import pytz
import nest_asyncio
import logging
from grok_processing import analyze_and_screen, stream_grok3_response, process_user_question, clean_html
from lihkg_api import get_category_name

# 應用 asyncio 補丁
nest_asyncio.apply()

# 香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 日誌過濾器：完全過濾 in-event 日誌
class InEventFilter(logging.Filter):
    def filter(self, record):
        # 完全過濾掉所有 in-event 日誌
        if "in-event" in record.msg.lower():
            return False
        return True

# 配置日誌記錄器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")  # 保存日誌到文件
    ]
)
logger = logging.getLogger(__name__)
logger.addFilter(InEventFilter())

# 確保全局日誌級別為 INFO
logging.getLogger().setLevel(logging.INFO)

# 設置日誌時間為香港時區
logging.Formatter.converter = lambda *args: datetime.now(HONG_KONG_TZ).timetuple()

# 聊天記錄持久化文件
CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """從文件加載聊天記錄"""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chat history: {str(e)}")
            return []
    return []

def save_chat_history(chat_history):
    """保存聊天記錄到文件"""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat history: {str(e)}")

async def main():
    """
    主函數，初始化 Streamlit 應用，處理用戶輸入並渲染聊天介面。
    """
    st.title("LIHKG 聊天介面")

    # 初始化 session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if "thread_cache" not in st.session_state:
        st.session_state.thread_cache = {}
    if "rate_limit_info" not in st.session_state:
        st.session_state.rate_limit_info = {"counter": 0, "last_reset": time.time(), "until": 0}
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = None
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    # 分類選擇
    cat_id_map = {
        "吹水台": 1, "熱門台": 2, "時事台": 5, "上班台": 14,
        "財經台": 15, "成人台": 29, "創意台": 31
    }
    selected_cat = st.selectbox("選擇分類", options=list(cat_id_map.keys()), index=0)
    cat_id = cat_id_map[selected_cat]

    # 顯示速率限制狀態
    st.markdown("#### 速率限制狀態")
    current_time = time.time()
    rate_limit_info = st.session_state.rate_limit_info
    st.markdown(f"- 請求計數: {rate_limit_info['counter']}")
    st.markdown(f"- 最後重置: {datetime.fromtimestamp(rate_limit_info['last_reset'], tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
    st.markdown(f"- 速率限制解除: {datetime.fromtimestamp(rate_limit_info['until'], tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S if rate_limit_info['until'] > current_time else '無限制'}")

    # 顯示聊天記錄
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(clean_html(chat["question"]))
        with st.chat_message("assistant"):
            st.markdown(clean_html(chat["answer"]))

    # 用戶輸入
    user_question = st.chat_input("請輸入 LIHKG 話題（例如：有哪些搞笑話題？）")
    if user_question and not st.session_state.awaiting_response:
        user_question = clean_html(user_question)  # 清理用戶輸入
        logger.info(f"User question: {user_question}")
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.awaiting_response = True
        st.session_state.chat_history.append({"question": user_question, "answer": ""})
        save_chat_history(st.session_state.chat_history)  # 保存聊天記錄

        with st.spinner("正在處理..."):
            try:
                # 檢查速率限制
                if current_time < rate_limit_info["until"]:
                    error_message = f"速率限制中，請在 {datetime.fromtimestamp(rate_limit_info['until'], tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} 後重試。"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    st.session_state.chat_history[-1]["answer"] = error_message
                    st.session_state.awaiting_response = False
                    save_chat_history(st.session_state.chat_history)
                    logger.info(f"Program response: {error_message}")
                    logger.warning(f"Rate limit triggered: {error_message}")
                    return

                # 重置聊天記錄（若問題變化較大）
                if not st.session_state.last_user_query or len(set(user_question.split()).intersection(set(st.session_state.last_user_query.split()))) < 2:
                    st.session_state.thread_cache = {k: v for k, v in st.session_state.thread_cache.items() if time.time() - v["timestamp"] < 3600}
                    st.session_state.last_user_query = user_question

                # 分析問題並篩選帖子
                analysis = await analyze_and_screen(user_query=user_question, cat_name=selected_cat, cat_id=cat_id)
                logger.info(f"Analysis completed: theme={analysis.get('theme')}, post_limit={analysis.get('post_limit')}")

                # 若無相關分類，直接生成回應
                if not analysis.get("category_ids"):
                    response = ""
                    response_buffer = ""
                    with st.chat_message("assistant"):
                        grok_container = st.empty()
                        async for chunk in stream_grok3_response(user_question, [], {}, "summarize"):
                            response_buffer += chunk
                            # 每 50 字或完整語句更新顯示
                            if len(response_buffer) >= 50 or chunk.endswith(('。', '！', '？')):
                                response += response_buffer
                                grok_container.markdown(response)
                                response_buffer = ""
                        # 顯示剩餘內容
                        if response_buffer:
                            response += response_buffer
                            grok_container.markdown(response)
                    st.session_state.chat_history[-1]["answer"] = response
                    st.session_state.awaiting_response = False
                    save_chat_history(st.session_state.chat_history)
                    logger.info(f"Program response: {response if len(response) <= 1000 else response[:1000] + '...'}")
                    return

                # 處理用戶問題
                result = await process_user_question(
                    user_question=user_question, selected_cat=selected_cat, cat_id=cat_id, analysis=analysis,
                    rate_limit_info=rate_limit_info
                )

                # 更新速率限制狀態
                st.session_state.rate_limit_info = result.get("rate_limit_info", rate_limit_info)
                thread_data = result.get("thread_data", [])
                question_cat = result.get("selected_cat", selected_cat)

                # 若無帖子數據
                if not thread_data:
                    answer = f"在 {question_cat} 中未找到符合條件的帖子，可能回覆數或點贊數過低，請嘗試其他分類。"
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.chat_history[-1]["answer"] = answer
                    st.session_state.awaiting_response = False
                    save_chat_history(st.session_state.chat_history)
                    logger.info(f"Program response: {answer}")
                    logger.warning(f"No threads found for category {question_cat}")
                    return

                # 準備回應
                post_limit = analysis.get("post_limit", 2)
                thread_data = thread_data[:post_limit]
                theme = analysis.get("theme", "相關")
                response = f"以下分享{post_limit}個被認為『{theme}』的帖子：\n\n"
                metadata = [
                    {
                        "thread_id": item["thread_id"], "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0), "last_reply_time": item.get("last_reply_time", "0"),
                        "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0)
                    } for item in thread_data
                ]
                for meta in metadata:
                    response += f"帖子 ID: {meta['thread_id']}\n標題: {meta['title']}\n"
                response += "\n"

                # 顯示回應
                response_buffer = ""
                with st.chat_message("assistant"):
                    grok_container = st.empty()
                    async for chunk in stream_grok3_response(user_question, metadata, {item["thread_id"]: item for item in thread_data}, analysis["processing"]):
                        response_buffer += chunk
                        # 每 50 字或完整語句更新顯示
                        if len(response_buffer) >= 50 or chunk.endswith(('。', '！', '？')):
                            response += response_buffer
                            grok_container.markdown(response)
                            response_buffer = ""
                    # 顯示剩餘內容
                    if response_buffer:
                        response += response_buffer
                        grok_container.markdown(response)

                logger.info(f"Program response: {response if len(response) <= 1000 else response[:1000] + '...'}")
                logger.info(f"Response generated: category={question_cat}, threads={len(thread_data)}")
                st.session_state.chat_history[-1]["answer"] = response
                st.session_state.last_user_query = user_question
                st.session_state.awaiting_response = False
                save_chat_history(st.session_state.chat_history)

            except asyncio.TimeoutError:
                error_message = "處理超時，請稍後重試。"
                logger.error(f"Timeout error: {error_message}")
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.chat_history[-1]["answer"] = error_message
                st.session_state.awaiting_response = False
                save_chat_history(st.session_state.chat_history)
                logger.info(f"Program response: {error_message}")

            except Exception as e:
                error_message = f"處理失敗：{str(e)}"
                logger.error(f"Processing error: {str(e)}")
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.chat_history[-1]["answer"] = error_message
                st.session_state.awaiting_response = False
                save_chat_history(st.session_state.chat_history)
                logger.info(f"Program response: {error_message}")

if __name__ == "__main__":
    asyncio.run(main())
