
import os
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI


# Load biến môi trường từ .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
question = st.text_input("💬 Câu hỏi của bạn:")
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
if question:
    try:
        prompt = f"Câu hỏi: {question}"
        st.code(prompt)

        response = llm.invoke(prompt)
        st.success(response)
    except Exception as e:
        st.error(f"Lỗi khi gọi AI: {e}")
# Giao diện ứng dụng
st.set_page_config(page_title="AI Tư vấn Tài liệu Khách hàng", page_icon="📄")
st.title("📄 Tải tài liệu riêng và hỏi AI")

# Sidebar - Nhập tên khách hàng
st.sidebar.header("👤 Thông tin khách hàng")
if "customer_name" not in st.session_state:
    st.session_state.customer_name = ""

name_input = st.sidebar.text_input("Nhập tên khách hàng", value=st.session_state.customer_name)
if name_input:
    st.session_state.customer_name = name_input

# Hiển thị chào mừng
if st.session_state.customer_name:
    st.success(f"Xin chào {st.session_state.customer_name}! Bạn muốn hỏi gì?")
else:
    st.info("Vui lòng nhập tên để bắt đầu.")

# Khởi tạo lịch sử chat nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# B1: Nhập tên (tùy chọn, lại lần nữa cho giao diện chính)
name = st.text_input("👤 Tên khách hàng (tuỳ chọn):")

# B2: Upload tài liệu
uploaded_file = st.file_uploader("📎 Tải file tài liệu (PDF hoặc TXT)", type=["txt", "pdf"])

# B3: Nhập câu hỏi
if st.session_state.customer_name:
    question = st.text_input(f"{st.session_state.customer_name} hỏi AI:")
else:
    question = st.text_input("Bạn muốn hỏi gì?")

# B4: Xử lý khi đủ thông tin
if uploaded_file and question:
    # Lưu tạm file
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Tải tài liệu
    if uploaded_file.name.endswith(".pdf"):
        loader = PDFMinerLoader(temp_path)
    else:
        loader = TextLoader(temp_path, encoding="utf-8")
    documents = loader.load()

    # Tách tài liệu
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Khởi tạo vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(docs, embeddings)

    # Tạo mô hình và chuỗi trả lời
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Tạo context cá nhân hóa
    context = f"Khách hàng: {name}. " if name else ""
    answer = chain.run(input_documents=db.similarity_search(question), question=context + question)

    # Hiển thị câu trả lời
    st.markdown("#### 🤖 Trả lời:")
    st.write(answer)

    # Lưu vào lịch sử chat
    st.session_state.chat_history.append(("Khách hàng", question))
    st.session_state.chat_history.append(("AI", answer))

    # Xóa file tạm
    os.remove(temp_path)

# B5: Lưu lịch sử chat
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

