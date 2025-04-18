import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
from data_processor import process_user_question
from grok3_client import analyze_question_nature, stream_grok3_response
from lihkg_api import get_category_name

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = st.logger.get_logger(__name__)

def is_new_conversation(current_query, last_query):
    """檢查當前問題是否與上一問題無關，啟動新一輪對話"""
    if not last_query:
        return True
    current_words = set(current_query.split())
    last_words = set(last_query.split())
    return len(current_words.intersection(last_words)) < 2

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
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = None
    if "last_cat_id" not in st.session_state:
        st.session_state.last_cat_id = None
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False
    
    # 分類選單
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
    cat_id = cat_id_map[selected_cat]
    
    # 顯示速率限制狀態
    st.markdown("#### 速率限制狀態")
    st.markdown(f"- 當前請求計數: {st.session_state.request_counter}")
    st.markdown(f"- 最後重置時間: {datetime.fromtimestamp(st.session_state.last_reset, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(
        f"- 速率限制解除時間: "
        f"{datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.rate_limit_until > time.time() else '無限制'}"
    )
    
    # 顯示聊天記錄
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
    
    # 用戶輸入
    user_question = st.chat_input("請輸入您想查詢的 LIHKG 話題（例如：有哪些搞笑話題？）")
    
    if user_question and not st.session_state.awaiting_response:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.awaiting_response = True
        
        with st.spinner("正在處理您的問題..."):
            try:
                # 檢查速率限制
                if time.time() < st.session_state.rate_limit_until:
                    error_message = (
                        f"API 速率限制中，請在 "
                        f"{datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')} 後重試。"
                    )
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": error_message
                    })
                    st.session_state.awaiting_response = False
                    logger.warning(f"速率限制阻止請求: 問題={user_question}")
                    return
                
                # 檢查是否為新一輪對話
                if is_new_conversation(user_question, st.session_state.last_user_query):
                    st.session_state.chat_history = [{"question": user_question, "answer": ""}]
                    st.session_state.topic_list_cache = {}
                    st.session_state.last_user_query = user_question
                    st.session_state.last_cat_id = cat_id
                
                # 分析問題性質
                logger.info(f"開始分析問題: 問題={user_question}, 分類={selected_cat}")
                analysis = await analyze_question_nature(user_question, selected_cat, cat_id)
                
                if not analysis.get("category_ids"):
                    # 無關 LIHKG 的問題
                    response = ""
                    with st.chat_message("assistant"):
                        grok_container = st.empty()
                        async for chunk in stream_grok3_response(user_question, [], {}, "other"):
                            response += chunk
                            grok_container.markdown(response)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.chat_history[-1]["answer"] = response
                    st.session_state.awaiting_response = False
                    logger.info(f"無關 LIHKG 問題: 問題={user_question}, 回應={response[:50]}...")
                    return
                
                # 檢查緩存
                cache_key = f"{analysis['category_ids'][0]}"
                cache_duration = 300
                use_cache = False
                if cache_key in st.session_state.topic_list_cache:
                    cache_data = st.session_state.topic_list_cache[cache_key]
                    if time.time() - cache_data["timestamp"] < cache_duration:
                        result = {
                            "thread_data": cache_data["thread_data"],
                            "rate_limit_info": [],
                            "request_counter": st.session_state.request_counter,
                            "last_reset": st.session_state.last_reset,
                            "rate_limit_until": st.session_state.rate_limit_until,
                            "selected_cat": selected_cat
                        }
                        use_cache = True
                        logger.info(f"使用緩存: cat_id={cache_key}, 帖子數={len(result['thread_data'])}")
                
                if not use_cache:
                    result = await process_user_question(
                        user_question,
                        cat_id_map,
                        selected_cat,
                        analysis,
                        st.session_state.request_counter,
                        st.session_state.last_reset,
                        st.session_state.rate_limit_until
                    )
                    st.session_state.topic_list_cache[cache_key] = {
                        "thread_data": result.get("thread_data", []),
                        "timestamp": time.time()
                    }
                    logger.info(f"更新緩存: cat_id={cache_key}, 帖子數={len(result['thread_data'])}")
                
                # 更新速率限制
                st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
                st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
                st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
                
                thread_data = result.get("thread_data", [])
                rate_limit_info = result.get("rate_limit_info", [])
                question_cat = result.get("selected_cat", selected_cat)
                
                # 檢查數據
                if not thread_data:
                    answer = f"在 {question_cat} 中未找到符合條件的帖子。"
                    if rate_limit_info:
                        answer += "\n#### 調試信息：\n" + "\n".join(f"- {info}" for info in rate_limit_info)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.chat_history[-1]["answer"] = answer
                    st.session_state.awaiting_response = False
                    logger.warning(f"無有效帖子: 問題={user_question}, 分類={question_cat}")
                    return
                
                # 生成回應
                metadata = [
                    {
                        "thread_id": item["thread_id"],
                        "title": item["title"],
                        "no_of_reply": item["no_of_reply"],
                        "last_reply_time": item["last_reply_time"],
                        "like_count": item["like_count"],
                        "dislike_count": item["dislike_count"]
                    }
                    for item in thread_data
                ]
                response = ""
                with st.chat_message("assistant"):
                    grok_container = st.empty()
                    async for chunk in stream_grok3_response(user_question, metadata, {item["thread_id"]: item for item in thread_data}, analysis["processing"]):
                        response += chunk
                        grok_container.markdown(response)
                debug_info = [f"#### 調試信息：\n- 分類: {question_cat}", f"- 帖子數: {len(thread_data)}", f"- 使用緩存: {use_cache}"]
                if rate_limit_info:
                    debug_info.append("- 速率限制或錯誤：")
                    debug_info.extend(f"  - {info}" for info in rate_limit_info)
                final_answer = response + "\n\n" + "\n".join(debug_info)
                
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.chat_history[-1]["answer"] = final_answer
                st.session_state.last_user_query = user_question
                st.session_state.last_cat_id = cat_id
                st.session_state.awaiting_response = False
                
            except Exception as e:
                error_message = f"處理失敗，原因：{str(e)}\n\n#### 調試信息：\n- 錯誤: {str(e)}"
                if 'result' in locals() and result.get("rate_limit_info"):
                    error_message += "\n- 速率限制或錯誤：\n" + "\n".join(f"  - {info}" for info in result["rate_limit_info"])
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.chat_history[-1]["answer"] = error_message
                st.session_state.awaiting_response = False
                logger.error(f"處理錯誤: 問題={user_question}, 錯誤={str(e)}")
    
    # 處理後續交互
    if st.session_state.awaiting_response and st.session_state.chat_history[-1]["answer"]:
        response_input = st.chat_input("請輸入指令（繼續、修改分類、ID 數字、結束）：")
        if response_input:
            response_input = response_input.strip().lower()
            if response_input == "結束":
                final_answer = "分析已結束，感謝您的使用！"
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.chat_history.append({"question": "結束", "answer": final_answer})
                st.session_state.awaiting_response = False
            elif response_input == "繼續":
                analysis = await analyze_question_nature(
                    st.session_state.last_user_query,
                    get_category_name(st.session_state.last_cat_id),
                    st.session_state.last_cat_id,
                    is_advanced=True,
                    metadata=[{k: v for k, v in item.items() if k != "replies"} for item in st.session_state.chat_history[-1]["thread_data"]],
                    thread_data={item["thread_id"]: item for item in st.session_state.chat_history[-1]["thread_data"]},
                    initial_response=st.session_state.chat_history[-1]["answer"]
                )
                if analysis.get("needs_advanced_analysis"):
                    result = await process_user_question(
                        st.session_state.last_user_query,
                        cat_id_map,
                        get_category_name(st.session_state.last_cat_id),
                        analysis["suggestions"],
                        st.session_state.request_counter,
                        st.session_state.last_reset,
                        st.session_state.rate_limit_until
                    )
                    st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
                    st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
                    st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
                    
                    thread_data = result.get("thread_data", [])
                    metadata = [
                        {
                            "thread_id": item["thread_id"],
                            "title": item["title"],
                            "no_of_reply": item["no_of_reply"],
                            "last_reply_time": item["last_reply_time"],
                            "like_count": item["like_count"],
                            "dislike_count": item["dislike_count"]
                        }
                        for item in thread_data
                    ]
                    response = ""
                    with st.chat_message("assistant"):
                        grok_container = st.empty()
                        async for chunk in stream_grok3_response(
                            st.session_state.last_user_query,
                            metadata,
                            {item["thread_id"]: item for item in thread_data},
                            analysis["suggestions"]["processing"]
                        ):
                            response += chunk
                            grok_container.markdown(response)
                    debug_info = [f"#### 調試信息：\n- 分類: {result.get('selected_cat')}", f"- 帖子數: {len(thread_data)}"]
                    if result.get("rate_limit_info"):
                        debug_info.append("- 速率限制或錯誤：")
                        debug_info.extend(f"  - {info}" for info in result["rate_limit_info"])
                    final_answer = response + "\n\n" + "\n".join(debug_info)
                    
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                    st.session_state.chat_history.append({"question": "繼續", "answer": final_answer, "thread_data": thread_data})
                else:
                    final_answer = "數據已充分，無需進一步分析。請輸入「結束」或新問題。"
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                    st.session_state.chat_history.append({"question": "繼續", "answer": final_answer})
                st.session_state.awaiting_response = False
            elif response_input == "修改分類":
                final_answer = "請選擇新分類並輸入問題。"
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.chat_history.append({"question": "修改分類", "answer": final_answer})
                st.session_state.awaiting_response = False
            elif response_input.isdigit():
                thread_id = response_input
                thread_data = st.session_state.chat_history[-1]["thread_data"]
                if thread_id in [item["thread_id"] for item in thread_data]:
                    response = ""
                    with st.chat_message("assistant"):
                        grok_container = st.empty()
                        async for chunk in stream_grok3_response(
                            st.session_state.last_user_query,
                            [item for item in thread_data if item["thread_id"] == thread_id],
                            {thread_id: next(item for item in thread_data if item["thread_id"] == thread_id)},
                            "summarize"
                        ):
                            response += chunk
                            grok_container.markdown(response)
                    debug_info = [f"#### 調試信息：\n- 帖子 ID: {thread_id}"]
                    final_answer = response + "\n\n" + "\n".join(debug_info)
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                    st.session_state.chat_history.append({"question": f"ID {thread_id}", "answer": final_answer})
                else:
                    final_answer = f"無效帖子 ID {thread_id}，請重新輸入。"
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                    st.session_state.chat_history.append({"question": f"ID {thread_id}", "answer": final_answer})
                st.session_state.awaiting_response = False
            else:
                final_answer = "請輸入有效指令：繼續、修改分類、ID 數字、結束"
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.chat_history.append({"question": response_input, "answer": final_answer})
                st.session_state.awaiting_response = True

if __name__ == "__main__":
    asyncio.run(chat_page())
