import streamlit as st
from chat_page import chat_page
from test_page import test_page

def main():
    st.sidebar.title("導航")
    page = st.sidebar.selectbox("選擇頁面", ["聊天介面", "測試頁面"])
    
    if page == "聊天介面":
        chat_page()
    elif page == "測試頁面":
        test_page()

if __name__ == "__main__":
    main()
