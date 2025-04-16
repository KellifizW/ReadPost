import streamlit as st
import asyncio
import re
from data_processor import analyze_lihkg_metadata
from grok3_client import stream_grok3_response
from lihkg_api import get_lihkg_thread_content
from utils import build_post_context

st.set_page_config(page_title="LIHKG 討論區分析", layout="wide")

async def select_relevant_threads(user_query, metadata, max_threads=3):
    if not metadata:
        st.warning("無元數據可選擇帖子")
        return []
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}, 正評: {item['like_count']}, 負評: {item['dislike_count']}"
        for item in metadata
    ])
    
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區今日（2025-04-16）已排序的帖子元數據：
{metadata_text}

基於標題，選擇與問題最相關的帖子 ID（每行一個數字），最多 {max_threads} 個。僅返回 ID，無其他內容。若無相關帖子，返回空列表。
"""
    
    call_id = f"select_{int(time.time())}"
    response = ""
    async for chunk in stream_grok3_response(prompt, call_id=call_id):
        response += chunk
    
    thread_ids = re.findall(r'\b(\d{5,})\b', response, re.MULTILINE)
    valid_ids = [str(item["thread_id"]) for item in metadata]
    return [tid for tid in thread_ids if tid in valid_ids][:max_threads]

async def summarize_thread(thread_id, cat_id, metadata, user_query, lihkg_data):
    post = next((item for item in metadata if str(item["thread_id"]) == str(thread_id)), None)
    if not post:
        yield f"錯誤: 找不到帖子 {thread_id}"
        return
    
    replies = await get_lihkg_thread_content(thread_id, cat_id=cat_id)
    like_count = post.get("like_count", 0)
    dislike_count = post.get("dislike_count", 0)
    lihkg_data[thread_id] = {"post": post, "replies": replies}
    
    if len(replies) < 50:
        yield f"標題: {post['title']}\n總結: 討論參與度低，網民回應不足，話題未見熱烈討論。（回覆數: {len(replies)}）\n評分: 正評 {like_count}, 負評 {dislike_count}"
        return
    
    context = build_post_context(post, replies)
    chunks = context.split("\n", 1)  # 簡單分塊，實際應使用 utils.chunk_text
    
    chunk_summaries = []
    cat_name = {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(cat_id, "未知分類")
    for i, chunk in enumerate(chunks):
        prompt = f"""
請將帖子（ID: {thread_id}）總結為 100-200 字，僅基於以下內容，聚焦標題與回覆，作為問題的補充細節，禁止引入無關話題。以繁體中文回覆。

帖子內容：
{chunk}

參考使用者問題「{user_query}」與分類（cat_id={cat_id}，{cat_name}），執行以下步驟：
1. 識別問題意圖（如搞笑、爭議、時事、財經）。
2. 總結網民觀點，回答問題，適配分類語氣：
   - 吹水台：提取搞笑、輕鬆觀點。
   - 其他：適配主題。
3. 若內容與問題無關，返回：「內容與問題不符，無法回答。」
4. 若數據不足，返回：「內容不足，無法生成總結。」

輸出格式：
- 標題：<標題>
- 總結：100-200 字，反映網民觀點。
"""
        summary = ""
        async for chunk in stream_grok3_response(prompt, call_id=f"{thread_id}_chunk_{i}"):
            summary += chunk
        chunk_summaries.append(summary)
    
    final_prompt = f"""
請將帖子（ID: {thread_id}）的總結合併為 150-200 字，僅基於以下內容，聚焦標題與回覆，作為問題的補充細節，禁止引入無關話題。以繁體中文回覆。

分塊總結：
{'\n'.join(chunk_summaries)}

參考使用者問題「{user_query}」與分類（cat_id={cat_id}，{cat_name}），執行以下步驟：
1. 識別問題意圖（如搞笑、爭議、時事、財經）。
2. 總結網民觀點，回答問題，適配分類語氣：
   - 吹水台：提取搞笑、輕鬆觀點。
   - 其他：適配主題。
3. 若內容與問題無關，返回：「內容與問題不符，無法回答。」
4. 若數據不足，返回：「內容不足，無法生成總結。」

輸出格式：
- 標題：<標題>
- 總結：150-200 字，反映網民觀點。
- 評分：正評 {like_count}, 負評 {dislike_count}
"""
    async for chunk in stream_grok3_response(final_prompt, call_id=f"{thread_id}_final"):
        yield chunk

def chat_page():
    # 初始化 session state
    if "lihkg_data" not in st.session_state:
        st.session_state.lihkg_data = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "char_counts" not in st.session_state:
        st.session_state.char_counts = {}
    if "metadata" not in st.session_state:
        st.session_state.metadata = []
    if "is_fetching" not in st.session_state:
        st.session_state.is_fetching = False
    if "last_call_id" not in st.session_state:
        st.session_state.last_call_id = None
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = ""
    if "waiting_for_summary" not in st.session_state:
        st.session_state.waiting_for_summary = False
    if "last_cat_id" not in st.session_state:
        st.session_state.last_cat_id = 1

    st.title("LIHKG 討論區分析")
    
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
            index=list(cat_id_map.keys()).index(
                {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(st.session_state.last_cat_id, "吹水台")
            )
        )
        st.session_state.last_cat_id = cat_id_map[selected_cat]
    
    with col1:
        user_query = st.chat_input("輸入問題（例如：吹水台有哪些搞笑話題？）或回應（『需要』、『ID 數字』、『不需要』）")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if user_query and not st.session_state.is_fetching:
        st.session_state.is_fetching = True
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
            
            if st.session_state.waiting_for_summary:
                prompt_lower = user_query.lower().strip()
                
                if prompt_lower == "不需要":
                    st.session_state.waiting_for_summary = False
                    response = "好的，已結束深入分析。你可以提出新問題！"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                
                elif prompt_lower == "需要":
                    with st.spinner("正在選擇並總結相關帖子..."):
                        thread_ids = await select_relevant_threads(st.session_state.last_user_query, st.session_state.metadata)
                        if thread_ids:
                            summaries = []
                            for thread_id in thread_ids:
                                with st.chat_message("assistant"):
                                    full_summary = st.write_stream(summarize_thread(
                                        thread_id, st.session_state.last_cat_id, st.session_state.metadata,
                                        st.session_state.last_user_query, st.session_state.lihkg_data
                                    ))
                                    if not full_summary.startswith("錯誤:"):
                                        summaries.append(full_summary)
                                        st.session_state.messages.append({"role": "assistant", "content": full_summary})
                            if summaries:
                                response = "以上是相關帖子總結。你需要進一步分析其他帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                            else:
                                response = "無法生成帖子總結，可能數據不足。你需要我嘗試其他分析嗎？（輸入『需要』或『不需要』）"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
                        else:
                            response = "無相關帖子可總結。你需要我嘗試其他分析嗎？（輸入『需要』或『不需要』）"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
                
                elif re.match(r'^(id\s*)?\d{5,}$', prompt_lower.replace(" ", "")):
                    thread_id = re.search(r'\d{5,}', prompt_lower).group()
                    valid_ids = [str(item["thread_id"]) for item in st.session_state.metadata]
                    if thread_id in valid_ids:
                        with st.spinner(f"正在總結帖子 {thread_id}..."):
                            with st.chat_message("assistant"):
                                full_summary = st.write_stream(summarize_thread(
                                    thread_id, st.session_state.last_cat_id, st.session_state.metadata,
                                    st.session_state.last_user_query, st.session_state.lihkg_data
                                ))
                                if not full_summary.startswith("錯誤:"):
                                    response = f"帖子 {thread_id} 的總結：\n\n{full_summary}\n\n你需要進一步分析其他帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                                else:
                                    response = f"帖子 {thread_id} 總結失敗：{full_summary}。你需要嘗試其他帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.markdown(response)
                    else:
                        response = f"無效帖子 ID {thread_id}，請確認 ID 是否正確。你需要我自動選擇帖子嗎？（輸入『需要』、『ID 數字』或『不需要』）"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                
                else:
                    response = "請輸入『需要』以自動選擇帖子、『ID 數字』以指定帖子，或『不需要』以結束。"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
            
            else:
                st.session_state.metadata = []
                st.session_state.char_counts = {}
                st.session_state.waiting_for_summary = False
                st.session_state.last_user_query = user_query
                
                with st.spinner("正在分析討論區數據..."):
                    try:
                        result = await analyze_lihkg_metadata(user_query, cat_id=st.session_state.last_cat_id, max_pages=10)
                        if isinstance(result, str):  # 如果結果是字符串，直接顯示
                            full_response = result
                            with st.chat_message("assistant"):
                                st.markdown(full_response)
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:  # 如果結果是異步生成器，流式顯示
                            with st.chat_message("assistant"):
                                full_response = st.write_stream(result)
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        if not full_response.startswith("今日") and not full_response.startswith("錯誤") and not full_response.startswith("以下是分類"):
                            response = "你需要我對某個帖子生成更深入的總結嗎？請輸入『需要』以自動選擇帖子、『ID 數字』以指定帖子，或『不需要』以結束。"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
                            st.session_state.waiting_for_summary = True
                    except Exception as e:
                        response = f"錯誤: {str(e)}。請重試或檢查網路連線。"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
        
        st.session_state.is_fetching = False
