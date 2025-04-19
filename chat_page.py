import streamlit as st
import asyncio
import logging
from grok3_client import analyze_question_nature, stream_grok3_response
from data_processor import process_user_question
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)

async def main():
    st.title("LIHKG Post Analyzer")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "used_thread_ids" not in st.session_state:
        st.session_state.used_thread_ids = set()
    if "request_counter" not in st.session_state:
        st.session_state.request_counter = 0
    if "last_reset" not in st.session_state:
        st.session_state.last_reset = 0
    if "rate_limit_until" not in st.session_state:
        st.session_state.rate_limit_until = None
    if "cached_thread_data" not in st.session_state:
        st.session_state.cached_thread_data = {}
    
    # User input
    user_question = st.text_input("輸入你的問題：", placeholder="例如：分享5個最on9既 post?")
    question_cat = st.selectbox("選擇分類：", ["吹水台", "熱門台"], index=0)
    
    cat_id = 1 if question_cat == "吹水台" else 2
    
    if st.button("提交"):
        if not user_question:
            st.error("請輸入問題！")
            return
        
        is_new_conversation = True  # Simplified for this example
        
        # Clear previous results if new conversation
        if is_new_conversation:
            st.session_state.chat_history = []
            st.session_state.used_thread_ids = set()
        
        # Log question analysis
        logger.info(f"開始分析問題: 問題={user_question}, 分類={question_cat}")
        
        # Analyze question nature
        analysis = await analyze_question_nature(
            user_query=user_question,
            cat_name=question_cat,
            cat_id=cat_id,
            is_advanced=False,
            metadata={}
        )
        
        # Process question and fetch data
        thread_data, used_thread_ids = await process_user_question(
            user_question=user_question,
            cat_id=cat_id,
            cat_name=question_cat,
            analysis=analysis,
            used_thread_ids=st.session_state.used_thread_ids,
            request_counter=st.session_state.request_counter,
            last_reset=st.session_state.last_reset,
            rate_limit_until=st.session_state.rate_limit_until,
            fetch_last_pages=1,
            max_replies=60
        )
        
        # Update session state
        st.session_state.used_thread_ids.update(used_thread_ids)
        st.session_state.cached_thread_data[cat_id] = thread_data
        
        # Log initial thread data
        total_replies = sum(len(item.get("replies", [])) for item in thread_data)
        logger.info(f"初次分析詳情: 帖子數={len(thread_data)}, 總回覆數={total_replies}")
        
        # Generate initial response
        response = await stream_grok3_response(
            user_query=user_question,
            thread_data=thread_data,
            prompt_template="""
            你是一個智能助手，任務是基於 LIHKG 數據總結與幽默或搞笑相關的帖子內容，回答用戶問題。
            以繁體中文回覆，300-500 字，詳細總結每個帖子的搞笑內容，包括標題、關鍵回覆和幽默元素，僅使用提供數據。
            使用者問題：{user_query}
            分類：{thread_data}
            """
        )
        
        # Display initial response
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Advanced analysis check
        post_limit = analysis.get("post_limit", 5)
        if post_limit > 3:
            analysis_advanced = await analyze_question_nature(
                user_query=user_question,
                cat_name=question_cat,
                cat_id=cat_id,
                is_advanced=True,
                metadata={},
                thread_data={item["thread_id"]: item for item in thread_data},
                initial_response=response
            )
            
            if analysis_advanced.get("needs_advanced_analysis"):
                reason = analysis_advanced.get("suggestions", {}).get("reason", "未知")
                logger.info(f"觸發進階分析: 原因={reason}")
                
                # Reuse initial thread_data for advanced analysis
                thread_data_advanced = thread_data
                used_thread_ids = set(str(item["thread_id"]) for item in thread_data_advanced)
                
                # Fetch remaining replies for initial threads
                thread_data_advanced, _ = await process_user_question(
                    user_question=user_question,
                    cat_id=cat_id,
                    cat_name=question_cat,
                    analysis=analysis_advanced,
                    used_thread_ids=used_thread_ids,
                    request_counter=st.session_state.request_counter,
                    last_reset=st.session_state.last_reset,
                    rate_limit_until=st.session_state.rate_limit_until,
                    fetch_last_pages=0,
                    max_replies=100,
                    fetch_remaining_pages=True
                )
                
                # Generate advanced response
                response_advanced = await stream_grok3_response(
                    user_query=user_question,
                    thread_data=thread_data_advanced,
                    prompt_template="""
                    你是一個智能助手，任務是基於 LIHKG 數據總結與幽默或搞笑相關的帖子內容，回答用戶問題。
                    以繁體中文回覆，500-800 字，詳細總結5個帖子的搞笑內容，基於初次分析的帖子，重點分析新抓取的回覆，引用具體回覆內容。
                    回應以「**進階分析補充：**」開頭，僅使用提供數據。
                    使用者問題：{user_query}
                    分類：{thread_data}
                    """
                )
                
                # Display advanced response
                st.markdown(response_advanced)
                st.session_state.chat_history.append({"role": "assistant", "content": response_advanced})
            else:
                logger.info("跳過進階分析：帖子數量或回應充分")
        else:
            logger.info(f"跳過進階分析：post_limit={post_limit} 小於或等於 3")

if __name__ == "__main__":
    asyncio.run(main())
