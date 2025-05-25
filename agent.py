import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# 1. Tải API Key từ file .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. Tải nội dung tài liệu công ty từ file
loader = TextLoader("company_info.txt", encoding="utf-8")
documents = loader.load()

# 3. Cắt nhỏ nội dung cho AI dễ xử lý
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 4. Tạo vector database từ nội dung
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(docs, embeddings)

# 5. Hàm AI trả lời câu hỏi
def ask_ai(question):
    relevant_docs = db.similarity_search(question)
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=relevant_docs, question=question)

# ✅ Tùy chọn: Kiểm thử khi chạy file trực tiếp
if __name__ == "__main__":
    print(ask_ai("Xin chào!"))
