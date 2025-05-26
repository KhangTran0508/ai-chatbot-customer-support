
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Giao diện
st.set_page_config(page_title="AI Tư vấn Tài liệu Khách hàng", page_icon="📄")
st.title("📄 Tải tài liệu riêng và hỏi AI")

# B1: Nhập tên khách hàng
name = st.text_input("👤 Tên khách hàng (tuỳ chọn):")

# B2: Upload tài liệu
uploaded_file = st.file_uploader("📎 Tải file tài liệu (PDF hoặc TXT)", type=["txt", "pdf"])

# B3: Nhập câu hỏi
question = st.text_input("💬 Câu hỏi của bạn:")

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

# Nút lưu lịch sử
if st.button("💾 Lưu lịch sử chat"):
    chat_data = [{"Người gửi": sender, "Nội dung": message} for sender, message in st.session_state.chat_history]
    df = pd.DataFrame(chat_data)
    df.to_csv("lich_su_chat.csv", index=False, encoding="utf-8-sig")
    st.success("✅ Đã lưu vào file lich_su_chat.csv")
