import streamlit as st
import asyncio
import streamlit.logger

logger = streamlit.logger.get_logger(__name__)

from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

async def test_page():
    st.title("LIHKG 數據測試頁面")
    
    # 初始化緩存
    if "topic_list_cache" not in st.session_state:
        st.session_state.topic_list_cache = {}
    
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
        max_pages = st.slider("抓取頁數", 1, 20, 10)
    
    with col1:
        # 原有功能：抓取數據
        if st.button("抓取數據"):
            with st.spinner("正在抓取數據..."):
                logger.info(f"開始抓取數據: 分類={selected_cat}, cat_id={cat_id}, 頁數={max_pages}")
                cache_key = f"{cat_id}_{max_pages}"
                if cache_key in st.session_state.topic_list_cache:
                    items = st.session_state.topic_list_cache[cache_key]["items"]
                    rate_limit_info = st.session_state.topic_list_cache[cache_key]["rate_limit_info"]
                    logger.info(f"從緩存中載入數據: cat_id={cat_id}, 頁數={max_pages}, 帖子數={len(items)}")
                else:
                    result = await get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=max_pages)
                    items = result["items"]
                    rate_limit_info = result["rate_limit_info"]
                    st.session_state.topic_list_cache[cache_key] = {"items": items, "rate_limit_info": rate_limit_info}
                    logger.info(f"抓取完成: 總共 {len(items)} 篇帖子")
                
                # 檢查抓取是否完整
                expected_items = max_pages * 60
                if len(items) < expected_items * 0.5:  # 如果抓取的帖子數少於預期的一半
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
                        "last_reply_time": item.get("last_reply_time", ""),
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
                        st.markdown(f"- 帖子 ID: {item['thread_id']}，標題: {item['title']}，回覆數: {item['no_of_reply']}")
                else:
                    st.markdown("無符合條件的帖子。")
                    logger.warning("無符合條件的帖子，可能數據不足或篩選條件過嚴")

        # 新功能：查詢帖子回覆數
        st.markdown("---")
        st.markdown("### 查詢帖子回覆數")
        thread_id_input = st.text_input("輸入帖子 ID", placeholder="例如：123456")
        if st.button("查詢回覆數"):
            if thread_id_input:
                try:
                    thread_id = int(thread_id_input)
                    with st.spinner(f"正在查詢帖子 {thread_id} 的回覆數..."):
                        logger.info(f"查詢帖子回覆數: thread_id={thread_id}")
                        # 獲取帖子內容（不指定分類）
                        thread_data = await get_lihkg_thread_content(thread_id)
                        replies = thread_data["replies"]
                        thread_title = thread_data["title"]
                        api_reply_count = thread_data["total_replies"]
                        rate_limit_info = thread_data["rate_limit_info"]
                        
                        if rate_limit_info:
                            st.markdown("#### 速率限制調試信息")
                            for info in rate_limit_info:
                                st.markdown(f"- {info}")
                        
                        if replies is not None:
                            fetched_reply_count = len(replies)
                            if thread_title is None:
                                # 如果標題仍未找到，嘗試從所有分類中查找
                                logger.warning(f"從 API 回應中未找到標題，嘗試從帖子列表中查找: thread_id={thread_id}")
                                thread_title = "未知標題"
                                for cat_name, cat_id in cat_id_map.items():
                                    cache_key = f"{cat_id}_5"  # 使用 max_pages=5 作為緩存鍵
                                    if cache_key in st.session_state.topic_list_cache:
                                        items = st.session_state.topic_list_cache[cache_key]["items"]
                                        logger.info(f"從緩存中載入數據: cat_id={cat_id}, 頁數=5, 帖子數={len(items)}")
                                    else:
                                        result = await get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=5)
                                        items = result["items"]
                                        rate_limit_info.extend(result["rate_limit_info"])
                                        st.session_state.topic_list_cache[cache_key] = {"items": items, "rate_limit_info": result["rate_limit_info"]}
                                    metadata_list = [
                                        {
                                            "thread_id": item["thread_id"],
                                            "title": item["title"],
                                            "no_of_reply": item["no_of_reply"],
                                        }
                                        for item in items
                                    ]
                                    for item in metadata_list:
                                        if str(item["thread_id"]) == str(thread_id):
                                            thread_title = item["title"]
                                            logger.info(f"找到標題: thread_id={thread_id}, 標題={thread_title}, 分類={cat_name}")
                                            break
                                    if thread_title != "未知標題":
                                        break
                            
                            st.markdown(f"#### 查詢結果")
                            st.markdown(f"- 帖子 ID: {thread_id}")
                            st.markdown(f"- 標題: {thread_title}")
                            st.markdown(f"- 抓取的回覆數: {fetched_reply_count}")
                            if api_reply_count:
                                st.markdown(f"- API 報告的總回覆數: {api_reply_count}")
                            logger.info(f"查詢成功: thread_id={thread_id}, 抓取回覆數={fetched_reply_count}, API 總回覆數={api_reply_count}, 標題={thread_title}")
                        else:
                            st.markdown(f"錯誤：無法獲取帖子 {thread_id} 的回覆數據，可能是帖子不存在或需要登錄權限。")
                            logger.warning(f"查詢失敗: thread_id={thread_id}, 無回覆數據")
                except ValueError:
                    st.markdown("錯誤：請輸入有效的帖子 ID（必須是數字）。")
                    logger.warning(f"無效帖子 ID: {thread_id_input}")
                except Exception as e:
                    st.markdown(f"錯誤：查詢失敗，原因：{str(e)}")
                    logger.error(f"查詢帖子回覆數失敗: thread_id={thread_id_input}, 錯誤={str(e)}")
            else:
                st.markdown("請輸入帖子 ID。")
                logger.warning("未輸入帖子 ID")
