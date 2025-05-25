import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# B1: Nạp API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# B2: Kết nối lại vector database đã lưu
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# B3: Hàm hỏi AI
def ask_ai(question):
    # Tìm những đoạn liên quan nhất từ tài liệu
    relevant_docs = db.similarity_search(question)

    # Tạo LLM (GPT)
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Tạo chuỗi hỏi-đáp
    chain = load_qa_chain(llm, chain_type="stuff")

    # Trả lời dựa vào tài liệu
    answer = chain.run(input_documents=relevant_docs, question=question)
    return answer

# ✅ Dùng thử
if __name__ == "__main__":
    while True:
        question = input("💬 Hỏi AI: ")
        if question.lower() in ["exit", "quit"]:
            break
        print("🤖 Trả lời:", ask_ai(question))
