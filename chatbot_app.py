
import os
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI


# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Giao diện
st.set_page_config(page_title="AI Tư vấn Tài liệu Khách hàng", page_icon="📄")
st.title("📄 Tải tài liệu riêng và hỏi AI")
# Nhập tên khách hàng (chỉ hiển thị khi chưa nhập)
if "customer_name" not in st.session_state:
    st.session_state.customer_name = ""
# Khởi tạo chat_history nếu chưa tồn tại
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if question:
    st.session_state.chat_history.append(("Khách hàng", question))
    st.session_state.chat_history.append(("AI", response))

st.sidebar.header("👤 Thông tin khách hàng")
name_input = st.sidebar.text_input("Nhập tên khách hàng", value=st.session_state.customer_name)

# Gán lại khi có nhập
if name_input:
    st.session_state.customer_name = name_input

# Hiển thị lời chào
if st.session_state.customer_name:
    st.success(f"Xin chào {st.session_state.customer_name}! Bạn muốn hỏi gì?")
else:
    st.info("Vui lòng nhập tên để bắt đầu.")

# B1: Nhập tên khách hàng
name = st.text_input("👤 Tên khách hàng (tuỳ chọn):")

# B2: Upload tài liệu
uploaded_file = st.file_uploader("📎 Tải file tài liệu (PDF hoặc TXT)", type=["txt", "pdf"])

# B3: Nhập câu hỏi
if st.session_state.customer_name:
    question = st.text_input(f"{st.session_state.customer_name} hỏi AI:")
else:
    question = st.text_input("Bạn muốn hỏi gì?")
personalized_question = f"Khách hàng tên {st.session_state.customer_name} hỏi: {question}"
response = llm.invoke(personalized_question)


if uploaded_file and question:
    # Đọc file tạm thời
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Tải tài liệu
    if uploaded_file.name.endswith(".pdf"):
        loader = PDFMinerLoader(temp_path)
    else:
        loader = TextLoader(temp_path, encoding="utf-8")

    documents = loader.load()

    # Tách đoạn
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embedding tạm thời
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(docs, embeddings)

    # Hỏi
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    context = f"Khách hàng: {name}. " if name else ""
    answer = chain.run(input_documents=db.similarity_search(question), question=context + question)

    st.markdown("#### 🤖 Trả lời:")
    st.write(answer)

    os.remove(temp_path)
import pandas as pd

import pandas as pd

st.markdown("---")
if st.button("💾 Lưu lịch sử chat"):
    if st.session_state.chat_history:
        chat_data = [
            {"Người gửi": sender, "Nội dung": message}
            for sender, message in st.session_state.chat_history
        ]
        df = pd.DataFrame(chat_data)
        df.to_csv("lich_su_chat.csv", index=False, encoding="utf-8-sig")
        st.success("✅ Đã lưu vào file lich_su_chat.csv")
    else:
        st.warning("❗ Chưa có cuộc trò chuyện nào để lưu.")

