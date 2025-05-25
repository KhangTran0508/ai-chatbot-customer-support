import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# B1: Nạp API Key từ .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# B2: Load file văn bản (TXT)
file_path = "company_info.txt"  # đặt đúng tên file
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# B3: Tách nhỏ nội dung để AI dễ hiểu
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# B4: Tạo Embedding và lưu vào Chroma DB
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
db.persist()

print("✅ Đã nạp và lưu dữ liệu tài liệu vào Chroma!")
