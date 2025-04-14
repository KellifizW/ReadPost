# 前置配置（LIHKG、Grok 3、時區等）保持不變
# 其他函數（clean_html, try_parse_date, chunk_text, async_request, get_lihkg_topic_list, get_lihkg_thread_content, build_post_context, summarize_with_grok3, analyze_lihkg_metadata, select_relevant_threads, summarize_thread）保持不變

# 修改 manual_fetch_and_summarize 以修復無回應問題
async def manual_fetch_and_summarize(cat_id, sub_cat_id, start_page, max_pages, auto_sub_cat):
    st.session_state.is_fetching = True
    st.session_state.lihkg_data = {}
    st.session_state.summaries = {}
    st.session_state.char_counts = {}
    st.session_state.debug_log.append(f"手動抓取: 分類={cat_id}, 子分類={sub_cat_id}, 頁數={start_page}-{start_page+max_pages-1}")
    all_items = []
    
    valid_sub_cat_ids = [0, 1, 2]
    sub_cat_ids = valid_sub_cat_ids if auto_sub_cat else [sub_cat_id]
    
    for sub_id in sub_cat_ids:
        items = await get_lihkg_topic_list(cat_id, sub_id, start_page, max_pages)
        existing_ids = {item["thread_id"] for item in all_items}
        new_items = [item for item in items if item["thread_id"] not in existing_ids]
        all_items.extend(new_items)
    
    filtered_items = [item for item in all_items if item.get("no_of_reply", 0) > 175]
    sorted_items = sorted(filtered_items, key=lambda x: x.get("last_reply_time", ""), reverse=True)
    top_items = sorted_items[:10]
    
    for item in top_items:
        thread_id = item["thread_id"]
        replies = await get_lihkg_thread_content(thread_id)
        st.session_state.lihkg_data[thread_id] = {"post": item, "replies": replies}
        
        context = build_post_context(item, replies)
        chunks = chunk_text([context])
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await summarize_with_grok3(
                f"請將以下討論區帖子和回覆總結為100-200字，聚焦主要主題和關鍵意見，並以繁體中文回覆：\n\n{chunk}",
                call_id=f"{thread_id}_chunk_{i}"
            )
            if summary.startswith("錯誤:"):
                st.session_state.debug_log.append(f"手動總結失敗: 帖子 {thread_id}, 分塊 {i}, 錯誤: {summary}")
                st.error(f"帖子 {thread_id} 分塊 {i} 總結失敗：{summary}")
                continue
            chunk_summaries.append(summary)
        
        if chunk_summaries:
            final_summary = await summarize_with_grok3(
                f"請將以下分塊總結合併為100-200字的最終總結，聚焦主要主題和關鍵意見，並以繁體中文回覆：\n\n{'\n'.join(chunk_summaries)}",
                call_id=f"{thread_id}_final"
            )
            if not final_summary.startswith("錯誤:"):
                st.session_state.summaries[thread_id] = final_summary
    
    st.session_state.debug_log.append(f"抓取完成: 總結數={len(st.session_state.summaries)}")
    st.session_state.is_fetching = False
    if not st.session_state.summaries:
        st.warning("無總結結果，可能無符合條件的帖子，請檢查調錯日誌或調整參數。")
    st.rerun()

# Streamlit 主程式
def main():
    st.title("LIHKG 總結聊天機器人")

    # 初始化狀態
    if "is_fetching" not in st.session_state:
        st.session_state.is_fetching = False

    st.header("與 Grok 3 聊天")
    chat_cat_id = st.selectbox(
        "聊天分類",
        options=[1, 31, 11, 17, 15, 34],
        format_func=lambda x: {1: "吹水台", 31: "創意台", 11: "時事台", 17: "上班台", 15: "財經台", 34: "成人台"}[x],
        key="chat_cat_id"
    )
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("輸入問題（例如「有咩膠post?」或「有咩得意野?」）：", key="chat_input")
        submit_chat = st.form_submit_button("提交問題")
    
    if submit_chat and user_input:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("正在分析 LIHKG 帖子..."):
            analysis_result = asyncio.run(analyze_lihkg_metadata(user_input, cat_id=chat_cat_id))
        st.session_state.chat_history.append({"role": "assistant", "content": analysis_result})
        
        with st.spinner("正在選擇並總結相關帖子..."):
            thread_ids = asyncio.run(select_relevant_threads(analysis_result))
            if thread_ids:
                for thread_id in thread_ids:
                    if thread_id not in st.session_state.summaries:
                        summary = asyncio.run(summarize_thread(thread_id))
                        if not summary.startswith("錯誤:"):
                            st.session_state.summaries[thread_id] = summary
    
    if st.session_state.chat_history:
        st.subheader("聊天記錄")
        for i, chat in enumerate(st.session_state.chat_history):
            role = "你" if chat["role"] == "user" else "Grok 3"
            st.markdown(f"**{role}**：{chat['content']}")
            if chat["role"] == "assistant":
                call_id = f"metadata_{i//2}"
                char_count = st.session_state.char_counts.get(call_id, 0)
                st.write(f"**處理字元數**：{char_count} 字元")
            st.write("---")
        
        if st.session_state.debug_log:
            st.subheader("調錯日誌")
            for log in st.session_state.debug_log[-5:]:
                st.write(log)

    st.header("帖子總結")
    if st.session_state.summaries:
        for thread_id, summary in st.session_state.summaries.items():
            post = st.session_state.lihkg_data[thread_id]["post"]
            st.write(f"**標題**: {post['title']} (ID: {thread_id})")
            st.write(f"**總結**: {summary}")
            chunk_counts = [st.session_state.char_counts.get(f"{thread_id}_chunk_{i}", 0) for i in range(len(chunk_text([build_post_context(post, st.session_state.lihkg_data[thread_id]["replies"])])))]
            final_count = st.session_state.char_counts.get(f"{thread_id}_final", 0)
            st.write(f"**處理字元數**：分塊總結 {sum(chunk_counts)} 字元，最終總結 {final_count} 字元")
            if st.button(f"查看詳情 {thread_id}", key=f"detail_{thread_id}"):
                st.write("**回覆內容**：")
                for reply in st.session_state.lihkg_data[thread_id]["replies"]:
                    st.write(f"- {clean_html(reply['msg'])}")
            st.write("---")
    elif not st.session_state.is_fetching and not st.session_state.summaries:
        st.info("尚無總結內容，請提交問題或手動抓取帖子。")

    st.header("手動抓取 LIHKG 帖子")
    with st.form("manual_fetch_form"):
        cat_id = st.text_input("分類 ID (如 1 為吹水台)", "1")
        sub_cat_id = st.number_input("子分類 ID", min_value=0, value=0)
        start_page = st.number_input("開始頁數", min_value=1, value=1)
        max_pages = st.number_input("最大頁數", min_value=1, value=5)
        auto_sub_cat = st.checkbox("自動遍歷子分類 (0-2)", value=True)
        submit_fetch = st.form_submit_button("抓取並總結", disabled=st.session_state.is_fetching)
    
    if submit_fetch:
        with st.spinner("正在抓取並總結..."):
            asyncio.run(manual_fetch_and_summarize(cat_id, sub_cat_id, start_page, max_pages, auto_sub_cat))

if __name__ == "__main__":
    main()
