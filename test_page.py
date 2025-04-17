import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
from lihkg_api import get_lihkg_topic_list
import streamlit.logger

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
        st.error(f"API 速率限制中，請在 {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')} 後重試。")
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
        st.markdown(f"- 最後重置時間: {datetime.fromtimestamp(st.session_state.last_reset, tz=HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(
            f"- 速率限制解除時間: "
            f"{datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ).strftime('%Y W-%m-%d %H:%M:%S') if st.session_state.rate_limit_until > time.time() else '無限制'}"
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