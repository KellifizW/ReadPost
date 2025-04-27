# app.py

import streamlit as st
import asyncio
import json
import time
import logging
import datetime
import pytz
from grok_processing import analyze_and_screen, process_user_question, stream_grok3_response

# 設置香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

# 自定義日誌格式器，將時間戳設為香港時區
class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] + " HKT"

formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

# 控制台處理器
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 檔案處理器
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# LIHKG 分類映射
CATEGORY_MAPPING = {
    "時事台": "5",
    "創意台": "6",
    "財經台": "15",
    "娛樂台": "7",
    "硬件台": "10",
    "學術台": "13",
    "汽車台": "26",
    "旅遊台": "29",
    "運動台": "12",
    "手機台": "31",
    "遊戲台": "11",
    "潮流台": "28",
    "動漫台": "20",
    "音樂台": "23",
    "影視台": "19",
    "講故台": "16",
    "感情台": "8",
    "飲食台": "17",
    "女性台": "27",
    "寵物台": "25",
    "攝影台": "24",
    "上班台": "14",
    "吹水台": "2"
}

# 初始化 session_state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_cache" not in st.session_state:
        st.session_state.thread_cache = {}
    if "request_counter" not in st.session_state:
        st.session_state.request_counter = {"count": 0, "reset_time": time.time()}
    if "last_reset" not in st.session_state:
        st.session_state.last_reset = time.time()
    if "rate_limit_until" not in st.session_state:
        st.session_state.rate_limit_until = 0
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []
    if "previous_thread_ids" not in st.session_state:
        st.session_state.previous_thread_ids = []
    if "previous_thread_data" not in st.session_state:
        st.session_state.previous_thread_data = {}

# 進度回調函數
def progress_callback(message, progress):
    progress_bar.progress(progress, text=message)

# 格式化時間戳
def format_timestamp(timestamp):
    try:
        dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "未知時間"

# 主應用函數
async def main():
    st.set_page_config(page_title="LIHKG 討論區分析助手", page_icon="🗣️", layout="wide")
    st.title("🗣️ LIHKG 討論區分析助手")
    
    initialize_session_state()
    
    # 側邊欄
    with st.sidebar:
        st.header("設置")
        selected_cat = st.selectbox("選擇討論區", list(CATEGORY_MAPPING.keys()), index=0)
        cat_id = CATEGORY_MAPPING[selected_cat]
        
        st.subheader("聊天記錄")
        if st.button("清除聊天記錄"):
            st.session_state.messages = []
            st.session_state.conversation_context = []
            st.session_state.previous_thread_ids = []
            st.session_state.previous_thread_data = {}
            st.session_state.thread_cache = {}
            st.rerun()
        
        st.subheader("開始新對話")
        new_conversation_name = st.text_input("新對話名稱")
        if st.button("開始新對話") and new_conversation_name:
            st.session_state.messages = []
            st.session_state.conversation_context = []
            st.session_state.previous_thread_ids = []
            st.session_state.previous_thread_data = {}
            st.session_state.thread_cache = {}
            st.session_state.conversation_context.append({"role": "system", "content": f"開始新對話：{new_conversation_name}"})
            st.rerun()
        
        st.subheader("速率限制狀態")
        if st.session_state.rate_limit_until > time.time():
            st.warning(f"速率限制生效中，結束於 {format_timestamp(st.session_state.rate_limit_until)}")
        else:
            st.success("無速率限制")
        st.write(f"當前請求計數：{st.session_state.request_counter['count']}")
        st.write(f"上次重置時間：{format_timestamp(st.session_state.last_reset)}")
    
    # 主聊天介面
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 進度條
    progress_bar = st.progress(0, text="準備就緒")
    
    # 聊天輸入
    prompt = st.chat_input("輸入你的問題（例如：時事台有哪些熱門話題？）")
    if prompt:
        # 添加用戶消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_context.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 分析用戶問題
        try:
            progress_bar.progress(0.05, text="正在分析問題")
            analysis = await analyze_and_screen(
                user_query=prompt,
                cat_name=selected_cat,
                cat_id=cat_id,
                conversation_context=st.session_state.conversation_context
            )
            logger.info(f"Analysis result: {analysis}")
            
            if analysis.get("direct_response", False):
                progress_bar.progress(1.0, text="完成")
                response = f"問題與 LIHKG 討論區無關或過於模糊，請提供更多細節！\n分析原因：{analysis.get('reason', '未知')}"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.conversation_context.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.rerun()
            
            # 處理問題並抓取數據
            progress_bar.progress(0.1, text="正在處理問題")
            result = await process_user_question(
                user_query=prompt,
                selected_cat=selected_cat,
                cat_id=cat_id,
                analysis=analysis,
                request_counter=st.session_state.request_counter,
                last_reset=st.session_state.last_reset,
                rate_limit_until=st.session_state.rate_limit_until,
                is_advanced=analysis.get("needs_advanced_analysis", False),
                previous_thread_ids=st.session_state.previous_thread_ids,
                previous_thread_data=st.session_state.previous_thread_data,
                conversation_context=st.session_state.conversation_context,
                progress_callback=progress_callback
            )
            
            st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
            st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
            st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
            
            if result.get("rate_limit_info"):
                for info in result["rate_limit_info"]:
                    if "until" in info:
                        st.session_state.rate_limit_until = info["until"]
                        progress_bar.progress(1.0, text="速率限制生效")
                        response = f"錯誤：速率限制生效，請等到 {format_timestamp(info['until'])} 後重試。"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.conversation_context.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.rerun()
            
            thread_data_dict = {str(data["thread_id"]): data for data in result["thread_data"]}
            metadata = [
                {
                    "thread_id": data["thread_id"],
                    "title": data["title"],
                    "no_of_reply": data.get("no_of_reply", 0),
                    "last_reply_time": data.get("last_reply_time", "1970-01-01 00:00:00"),
                    "like_count": data.get("like_count", 0),
                    "dislike_count": data.get("dislike_count", 0)
                } for data in result["thread_data"]
            ]
            
            st.session_state.previous_thread_ids = list(thread_data_dict.keys())
            st.session_state.previous_thread_data = thread_data_dict
            
            # 生成回應
            progress_bar.progress(0.8, text="正在生成回應")
            with st.chat_message("assistant"):
                response_container = st.empty()
                response_text = ""
                async for chunk in stream_grok3_response(
                    user_query=prompt,
                    metadata=metadata,
                    thread_data=thread_data_dict,
                    processing=analysis,
                    selected_cat=selected_cat,
                    conversation_context=st.session_state.conversation_context,
                    needs_advanced_analysis=analysis.get("needs_advanced_analysis", False),
                    reason=analysis.get("reason", ""),
                    filters=analysis.get("filters", {}),
                    cat_id=cat_id
                ):
                    response_text += chunk
                    response_container.markdown(response_text)
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.conversation_context.append({"role": "assistant", "content": response_text})
            
            progress_bar.progress(1.0, text="完成")
            st.rerun()
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            progress_bar.progress(1.0, text="錯誤")
            response = f"錯誤：處理請求失敗（{str(e)}）。請稍後重試或聯繫支持。"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.conversation_context.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.rerun()

# 運行應用
if __name__ == "__main__":
    asyncio.run(main())
