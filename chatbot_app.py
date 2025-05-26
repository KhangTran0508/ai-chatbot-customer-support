import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import streamlit as st

# Kiểm tra xem API key đã được nạp từ biến môi trường chưa
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    st.info(f"🔐 API Key đã nạp: {api_key[:8]}..." + "[ĐÃ ẨN PHẦN CÒN LẠI]")
else:
    st.error("❌ CHƯA nạp được API Key từ biến môi trường hoặc secrets")

from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load biến môi trường
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Kiểm tra API key
if not openai_api_key:
    st.error("❌ Không tìm thấy OpenAI API key. Vui lòng kiểm tra file .env.")
    st.stop()

# Giao diện
st.set_page_config(page_title="AI Tư vấn Tài liệu Khách hàng", page_icon="📄")
st.title("📄 Tải tài liệu riêng và hỏi AI")

# Sidebar - tên khách hàng
st.sidebar.header("👤 Thông tin khách hàng")
if "customer_name" not in st.session_state:
    st.session_state.customer_name = ""

name_input = st.sidebar.text_input("Nhập tên khách hàng", value=st.session_state.customer_name)
if name_input:
    st.session_state.customer_name = name_input

# Giao diện chính
if st.session_state.customer_name:
    st.success(f"Xin chào {st.session_state.customer_name}! Bạn muốn hỏi gì?")
else:
    st.info("Vui lòng nhập tên để bắt đầu.")

# Lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Nhập tên khách hàng tùy chọn
name = st.text_input("👤 Tên khách hàng (tuỳ chọn):")

# Tải tài liệu
uploaded_file = st.file_uploader("📎 Tải file tài liệu (PDF hoặc TXT)", type=["txt", "pdf"])

# Câu hỏi
question = st.text_input(f"{st.session_state.customer_name} hỏi AI:" if st.session_state.customer_name else "Bạn muốn hỏi gì?")

# Nếu đủ thông tin
if uploaded_file and question:
    try:
        # Lưu file tạm
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Đọc file
        loader = PDFMinerLoader(temp_path) if uploaded_file.name.endswith(".pdf") else TextLoader(temp_path, encoding="utf-8")
        documents = loader.load()

        # Tách văn bản
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        # Tạo vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(docs, embeddings)

        # Khởi tạo LLM và chain
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")

        # Tạo prompt cá nhân hóa
        context = f"Khách hàng: {name}. " if name else ""
        answer = chain.run(input_documents=db.similarity_search(question), question=context + question)

        # Hiển thị
        st.markdown("#### 🤖 Trả lời:")
        st.write(answer)

        # Lưu lịch sử
        st.session_state.chat_history.append(("Khách hàng", question))
        st.session_state.chat_history.append(("AI", answer))

        # Xóa file tạm
        os.remove(temp_path)

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý: {e}")

# Lưu lịch sử
st.markdown("---")
if st.button("💾 Lưu lịch sử chat"):
    if st.session_state.chat_history:
        df = pd.DataFrame([
            {"Người gửi": sender, "Nội dung": msg}
            for sender, msg in st.session_state.chat_history
        ])
        df.to_csv("lich_su_chat.csv", index=False, encoding="utf-8-sig")
        st.success("✅ Đã lưu vào file lich_su_chat.csv")
    else:
        st.warning("❗ Chưa có cuộc trò chuyện nào để lưu.")

