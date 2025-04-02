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
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Milvus

# === Vannhk ===
import chatbot_crawl_data as crawl_data

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# connect Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# === CHECK IF VECTOR DB EXISTED ===
if utility.has_collection(COLLECTION_NAME):
    print(f"[Collection {COLLECTION_NAME} already exists] Loading DATA")
    vector_store = Milvus(collection_name=COLLECTION_NAME)
else:
    print(f"[Collection {COLLECTION_NAME} did not exist] Processing DATA")

    # crawl data
    base_url = "https://support.sapo.vn/sapo-retail"
    article_links = crawl_data.get_article_links(base_url)
    docs = crawl_data.load_articles(article_links)

    # chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # === DATA EMBEDDING + VECTOR DB ===
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPEN_API_KEY,base_url="https://models.inference.ai.azure.com")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # vector_store = FAISS.from_documents(all_splits, embeddings)  # Sử dụng FAISS để lưu trữ vector

    # Initialize Milvus collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="Collection contains vector embeddings of Sapo data")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Indexing and store in Milvus
    vector_store = Milvus.from_documents(all_splits, embeddings, collection_name=COLLECTION_NAME)
    print("[INFO] Lưu trữ Vector DB thành công!")


template = """ 
Hướng dẫn chi tiết theo yêu cầu người hỏi
Ví dụ minh họa:
**Bước 1: Truy cập màn hình thêm mới sản phẩm**
Bạn thực hiện các bước sau:
1. Đăng nhập vào trang quản lý phần mềm Sapo
2. Chọn mục **Sản phẩm**
3. Vào **Danh sách sản phẩm**
4. Nhấn nút **Tạo sản phẩm**

**Bước 2: Phân loại sản phẩm**
a. **Lựa chọn phân loại sản phẩm cần thêm mới:**
   - Chọn danh mục phù hợp với sản phẩm của bạn
   - Có thể tạo danh mục mới nếu chưa tồn tại

b. **Cập nhật loại / nhãn hiệu / tags sản phẩm:**
   - Chọn loại sản phẩm từ danh sách có sẵn
   - Thêm nhãn hiệu (nếu có)
   - Gắn tags để dễ dàng tìm kiếm

**Bước 3: Nhập thông tin chi tiết sản phẩm**
1. Điền tên sản phẩm
2. Nhập mã sản phẩm
3. Cập nhật giá bán
4. Quản lý tồn kho

Ghi chú: Luôn kiểm tra kỹ thông tin trước khi lưu sản phẩm.

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
