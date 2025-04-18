import streamlit as st
from chat_page import chat_page
from test_page import test_page
import asyncio
import nest_asyncio

# 應用 nest_asyncio 來允許在 Streamlit 中嵌套運行事件迴圈
nest_asyncio.apply()

async def main():
    st.sidebar.title("導航")
    page = st.sidebar.selectbox("選擇頁面", ["聊天介面", "測試頁面"])
    
    if page == "聊天介面":
        await chat_page()
    elif page == "測試頁面":
        await test_page()  # 添加 await

if __name__ == "__main__":
    # 在 Streamlit 中運行異步主函數
    asyncio.run(main())
