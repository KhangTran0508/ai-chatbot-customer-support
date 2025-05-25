import streamlit as st
from agent import ask_ai

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="AI Tư vấn Khách Hàng", layout="wide")

# Tiêu đề trang
st.title("🤖 AI Tư Vấn Khách Hàng Doanh Nghiệp")

# Ô nhập câu hỏi
user_input = st.text_input("💬 Nhập câu hỏi của bạn:")

# Khi người dùng nhập câu hỏi
if user_input:
    with st.spinner("🤔 Đang tìm câu trả lời..."):
        answer = ask_ai(user_input)
        st.success(f"💡 Trả lời: {answer}")
