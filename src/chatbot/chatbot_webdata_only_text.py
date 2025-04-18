import json
import os
from dotenv import load_dotenv
import bs4
from langchain import hub, requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores import FAISS

# === Vannhk ===
import chatbot_crawl_data as crawl_data

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# === CHECK IF VECTOR DB EXISTED ===
if os.path.exists(VECTOR_DB_PATH):
    print("[INFO] Đang tải dữ liệu từ Vector DB đã lưu...")
    with open(VECTOR_DB_PATH, "rb") as f:
        vector_store = pickle.load(f)
else:
    print("[INFO] Lần đầu tiên chạy - Đang tải và embedding dữ liệu...")

    # Home page cua Sapo Retail
    base_url = "https://support.sapo.vn/sapo-retail"
    article_links = crawl_data.get_article_links(base_url)

    docs = crawl_data.load_articles(article_links)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # === DATA EMBEDDING + VECTOR DB ===
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPEN_API_KEY,base_url="https://models.inference.ai.azure.com")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # ******embedidng_anh =

    vector_store = FAISS.from_documents(all_splits, embeddings)  # Sử dụng FAISS để lưu trữ vector
    # ****** vector_store cho ảnh =

    # === Lưu Vector DB để sử dụng sau ===
    with open(VECTOR_DB_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    print("[INFO] Lưu trữ Vector DB thành công!")



# === Define prompt for question-answering ===
# # C1
# prompt = hub.pull("rlm/rag-prompt")
# C2: customized
template = """ 
Hướng dẫn chi tiết theo yêu cầu người hỏi. Nếu trong văn bản có đường dẫn hình ảnh ("image link:"), hãy đưa chúng vào câu trả lời để minh họa rõ hơn.

Câu hỏi gốc: {question}
Thông tin chi tiết từ hệ thống: {context}

Trong câu trả lời của bạn:
1. Nếu có thông tin về các bước, hãy trình bày theo từng bước rõ ràng
2. Đối với mỗi bước mà có hình ảnh liên quan (có dòng "image link:"), hãy đưa link ảnh đó vào cuối phần giải thích của bước đó
3. Định dạng câu trả lời giống như ví dụ minh họa sau:

**Bước 1: [Tên bước]**
[Giải thích chi tiết]
[image link nếu có]

**Bước 2: [Tên bước]**
[Giải thích chi tiết]
[image link nếu có]

Chỉ trả lời dựa trên thông tin được cung cấp, không thêm thông tin không có trong văn bản gốc.

Câu hỏi gốc: {question}
Thông tin chi tiết từ hệ thống: {context}
"""

prompt = PromptTemplate.from_template(template)

# === Choose LLM model ===
llm = ChatOpenAI(model_name="gpt-4o",
                 api_key=OPEN_API_KEY,
                 base_url="https://models.inference.ai.azure.com",
                 temperature=0.0)

# === Define state for application ===
class State(TypedDict):
    question: str
    context: List[Document]
    answer:str


# === Define application steps (NODE) ===
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    # retrieved_docs = vector_store.similarity_search("question")
    return {"context": retrieved_docs}

def generate(state: State):

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = prompt.invoke({"question": state["question"], "context":docs_content})
    response = llm.invoke(messages)

    return {"answer": response.content}

# Compile application and test (CONTROL FLOW)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "Hướng dẫn tôi xử lý đơn hàng"})
response = dict(response)

print(response["answer"])






