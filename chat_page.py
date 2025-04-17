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
        st.session_state.topic_list_cache = {}  # {cat_id: {"items": [...], "timestamp": float}}
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
    
    # 顯示速率限制狀態
    st.markdown("#### 速率限制狀態")
    st.markdown(f"- 當前請求計數: {st.session_state.request_counter}")
    st.markdown(f"- 最後重置時間: {datetime.fromtimestamp(st.session_state.last_reset, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(
        f"- 速率限制解除時間: "
        f"{datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.rate_limit_until > time.time() else '無限制'}"
    )
    
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
                    logger.warning(f"速率限制阻止請求: 問題={user_question}, 限制至={st.session_state.rate_limit_until}")
                    return
                
                st.session_state.last_user_query = user_question
                
                logger.info(f"開始處理問題: 問題={user_question}, 分類={selected_cat}")
                
                # 檢查緩存
                cat_id = cat_id_map[selected_cat]
                for cat_name, cat_id_val in cat_id_map.items():
                    if cat_name in user_question:
                        selected_cat = cat_name
                        cat_id = cat_id_val
                        break
                
                cache_key = str(cat_id)
                cache_duration = 300  # 緩存 5 分鐘
                use_cache = False
                if cache_key in st.session_state.topic_list_cache:
                    cache_data = st.session_state.topic_list_cache[cache_key]
                    if time.time() - cache_data["timestamp"] < cache_duration:
                        result = {
                            "items": cache_data["items"],
                            "rate_limit_info": [],
                            "request_counter": st.session_state.request_counter,
                            "last_reset": st.session_state.last_reset,
                            "rate_limit_until": st.session_state.rate_limit_until
                        }
                        use_cache = True
                        logger.info(f"使用緩存: cat_id={cat_id}, 帖子數={len(result['items'])}")
                
                if not use_cache:
                    result = await process_user_question(
                        user_question,
                        cat_id_map=cat_id_map,
                        selected_cat=selected_cat,
                        request_counter=st.session_state.request_counter,
                        last_reset=st.session_state.last_reset,
                        rate_limit_until=st.session_state.rate_limit_until
                    )
                    # 更新緩存
                    st.session_state.topic_list_cache[cache_key] = {
                        "items": result.get("items", []),
                        "timestamp": time.time()
                    }
                    logger.info(f"更新緩存: cat_id={cat_id}, 帖子數={len(result['items'])}")
                
                st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
                st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
                st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
                
                thread_data = result.get("thread_data", [])
                rate_limit_info = result.get("rate_limit_info", [])
                question_cat = result.get("selected_cat", selected_cat)
                
                # 記錄抓取總結
                logger.info(
                    f"抓取完成: 問題={user_question}, 分類={question_cat}, "
                    f"帖子數={len(thread_data)}, 速率限制信息={rate_limit_info if rate_limit_info else '無'}, "
                    f"使用緩存={use_cache}"
                )
                
                # 檢查是否有數據
                if not thread_data:
                    answer = f"在 {question_cat} 中未找到回覆數 ≥ 125 的帖子。"
                    debug_info = []
                    if rate_limit_info:
                        debug_info.append("#### 調試信息：")
                        debug_info.extend(f"- {info}" for info in rate_limit_info)
                    answer += "\n".join(debug_info)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
                    logger.warning(f"無有效帖子數據: 問題={user_question}, 分類={question_cat}, 錯誤={rate_limit_info}")
                    return
                
                # 構建回應並添加調試資訊
                answer = f"### 來自 {question_cat} 的話題（回覆數 ≥ 125）：\n"
                debug_info = ["#### 調試信息："]
                failed_threads = []
                successful_threads = []
                
                for item in thread_data:
                    thread_id = item["thread_id"]
                    if not item["replies"] and item["no_of_reply"] >= 125:
                        failed_threads.append(thread_id)
                        debug_info.append(f"- 帖子 ID {thread_id} 抓取失敗，可能無效或無權訪問（錯誤 998）。")
                    else:
                        successful_threads.append(thread_id)
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
                debug_info.append(f"- 成功抓取帖子數: {len(successful_threads)}，ID: {successful_threads}")
                if failed_threads:
                    debug_info.append(f"- 失敗帖子數: {len(failed_threads)}，ID: {failed_threads}")
                    answer += f"\n警告：以下帖子無法獲取回覆（可能因錯誤 998）：{', '.join(map(str, failed_threads))}。"
                
                if rate_limit_info:
                    debug_info.append("- 速率限制或錯誤記錄：")
                    debug_info.extend(f"  - {info}" for info in rate_limit_info)
                
                debug_info.append(f"- 使用緩存: {use_cache}")
                logger.info("\n".join(debug_info))
                
                # 整合 Grok 3 回應
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
                        debug_info.append(f"- 縮減後上下文: 字元數={len(grok_context)}")
                        logger.info(f"縮減後上下文: 字元數a092a85a7f37388a1a8a1a6a6a1a1a1a1a1a1a1a1a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8 - 0x1f4af
                if not grok_response or grok_response.startswith("錯誤:"):
                    final_answer = (grok_response or "無法獲取 Grok 3 建議。") + "\n\n" + "\n".join(debug_info)
                else:
                    final_answer = grok_response + "\n\n" + "\n".join(debug_page.py" contentType="text/python">
import streamlit as st
import asyncio
import time
from datetime import datetime
import random
import pytz
from lihkg_api import get_lihkg_topic_list
import aiohttp

# 使用 Streamlit 的 logger
logger = st.logger.get_logger(__name__)

# 定義香港時區
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

async def test_page():
    st.title("LIHKG 數據測試頁面")
    
    # 初始化速率限制狀態
    if "rate_limit_until" not in st.session_state:
        st.session_state.rate_limit_until = 0
    if "request_counter" not in st.session_state:
        st.session_state.request_counter = 0
    if "last_reset" not in st.session_state:
        st.session_state.last_reset = time.time()
    
    # 檢查是否處於速率限制中
    if time.time() < st.session_state.rate_limit_until:
        st.error(f"API 速率限制中，請在 {datetime.fromtimestamp(st.session_state.rate_limit_until)} 後重試。")
        return
    
    cat_id_map = {
        "吹水台": 1,
        "熱門台": 2,
        "時事台": 5,
        "上班台": 14,
        "財經台": 15,
        "成人台": 29,
        "創意台": 31
    }
    
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_cat = st.selectbox(
            "選擇分類",
            options=list(cat_id_map.keys()),
            index=0
        )
        cat_id = cat_id_map[selected_cat]
        max_pages = st.slider("抓取頁數", 1, 5, 1)  # 範圍 1-5，預設 1
    
    with col1:
        # 顯示速率限制狀態
        st.markdown("#### 速率限制狀態")
        st.markdown(f"- 當前請求計數: {st.session_state.request_counter}")
        st.markdown(f"- 最後重置時間: {datetime.fromtimestamp(st.session_state.last_reset)}")
        st.markdown(
            f"- 速率限制解除時間: "
            f"{datetime.fromtimestamp(st.session_state.rate_limit_until) if st.session_state.rate_limit_until > time.time() else '無限制'}"
        )
        
        # 抓取數據
        if st.button("抓取數據"):
            with st.spinner("正在抓取數據..."):
                logger.info(f"開始抓取數據: 分類={selected_cat}, cat_id={cat_id}, 頁數={max_pages}")
                
                # 直接調用 API，無快取
                result = await get_lihkg_topic_list(
                    cat_id,
                    sub_cat_id=0,
                    start_page=1,
                    max_pages=max_pages,
                    request_counter=st.session_state.request_counter,
                    last_reset=st.session_state.last_reset,
                    rate_limit_until=st.session_state.rate_limit_until
                )
                items = result["items"]
                rate_limit_info = result["rate_limit_info"]
                st.session_state.request_counter = result["request_counter"]
                st.session_state.last_reset = result["last_reset"]
                st.session_state.rate_limit_until = result["rate_limit_until"]
                
                # 記錄抓取條件總結
                logger.info(
                    f"抓取完成: 分類={selected_cat}, cat_id={cat_id}, 頁數={max_pages}, "
                    f"帖子數={len(items)}, 成功={len(items) > 0}, "
                    f"速率限制信息={rate_limit_info if rate_limit_info else '無'}"
                )
                
                # 檢查抓取是否完整
                expected_items = max_pages * 60
                if len(items) < expected_items * 0.5:
                    st.warning("抓取數據不完整，可能是因為 API 速率限制。請稍後重試，或減少抓取頁數。")
                
                # 顯示速率限制調試信息
                if rate_limit_info:
                    st.markdown("#### 速率限制調試信息")
                    for info in rate_limit_info:
                        st.markdown(f"- {info}")
                
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
                
                # 顯示結果
                st.markdown(f"### 抓取結果（分類：{selected_cat}）")
                st.markdown(f"- 總共抓取 {len(metadata_list)} 篇帖子")
                st.markdown(f"- 回覆數 ≥ {min_replies} 的帖子數：{len(filtered_items)} 篇")
                
                if filtered_items:
                    st.markdown("#### 符合條件的帖子：")
                    for item in filtered_items:
                        st.markdown(f"- 帖子 ID: {item['thread_id']}，標題: {item['title']}，回覆數: {item['no_of_reply']}，最後回覆時間: {item['last_reply_time']}")
                else:
                    st.markdown("無符合條件的帖子。")
                    logger.warning("無符合條件的帖子，可能數據不足或篩選條件過嚴")
