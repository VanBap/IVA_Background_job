import os
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
# === Vannhk ===
import time
import vannhk_template

from splitter import HierarchicalPDFChunker

VECTOR_DB_DEMO_PATH = os.getenv("VECTOR_DB_DEMO_PATH_PDF")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
collection_name = "VBD_documents"


# === CHECK IF VECTOR DB EXISTED ===
# Kiểm tra xem collection đã tồn tại trong Milvus chưa
from pymilvus import utility, connections

# Kết nối đến Milvus
connections.connect(
    alias="default",
    uri=VECTOR_DB_DEMO_PATH
)

collection_exists = utility.has_collection(collection_name)

# # Sau đoạn code kiểm tra collection_exists
# if collection_exists:
#     print("[INFO] Collection đã tồn tại. Đang xóa và tạo lại...")
#     utility.drop_collection(collection_name)
#     collection_exists = False

if not collection_exists:
    print("[INFO] ================ chatbot_TEST =======================")
    print("[INFO] Lần đầu tiên chạy - Đang tải và embedding dữ liệu...")
    start = time.time()


    # ============== LOAD DATA FROM PDF =============================
    pdf_path = "/home/vbd-vanhk-l1-ubuntu/PycharmProjects/PythonProject/data/VBD_IVA_HDSD.pdf"
    chunker = HierarchicalPDFChunker()
    documents = chunker.process_pdf(pdf_path, start_page=10)

    print(f"Tạo {len(documents)} chunks theo cấu trúc phân cấp")

    # Hiển thị thông tin về documents
    print("\nThông tin về documents:")
    for i, doc in enumerate(documents):
        print("==========================================================")
        print(f"[Document {i + 1}]: section_level: {doc.metadata['section_level']} | section_title: {doc.metadata['section_title']}")
        print(f"[Page content]:  {doc.page_content}")
        # print(f"  - Tokens: {doc.metadata['tokens_count']}")
        # if "chunk_id" in doc.metadata:
        #     print(f"  - Chunk: {doc.metadata['chunk_id'] + 1}/{doc.metadata['total_chunks']}")
        # print(f"  - Đường dẫn đầy đủ: {doc.metadata['full_path']}")
        print()

    # ============ TEST MILVUS as VECTOR STORE ==============

    # Khi tạo mới vector store
    # Phiên bản đơn giản hơn không sử dụng hybrid search
    vector_store = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        connection_args={"uri": VECTOR_DB_DEMO_PATH},
        collection_name=collection_name,
        text_field="page_content",
        vector_field="embedding",
        consistency_level="Strong",
    )

    end = time.time()
    print(f"[SPLIT + EMBEDDING DATA] {end-start} seconds")
    print("[INFO] Lưu trữ Vector DB thành công!")

else:
    print("[INFO] ================ chatbot_TEST =======================")
    print("[INFO] Đang load vector store đã tồn tại...")

    # Khi load vector store đã tồn tại
    vector_store = Milvus(
        collection_name=collection_name,
        embedding_function=embeddings,
        connection_args={"uri": VECTOR_DB_DEMO_PATH},
        text_field="page_content",
        vector_field="embedding",
        enable_dynamic_field=True,

    )




from vannhk_template import template1, template2
prompt = PromptTemplate.from_template(template1)

# === Choose LLM model ===
llm = ChatOpenAI(model_name="gpt-4o-mini",
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
    start = time.time()
    retrieved_docs = vector_store.similarity_search(state["question"], k=2) #default k = 4

    end = time.time()
    print(f"[RETRIEVE TIME] {end-start} senconds")
    return {"context": retrieved_docs}

def generate(state: State):
    start = time.time()
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = prompt.invoke({"question": state["question"], "context":docs_content})
    response = llm.invoke(messages)


    end = time.time()
    print(f"[GENERATE TIME] {end - start} seconds")
    return {"answer": response.content}

# Compile application and test (CONTROL FLOW)
graph_builder = StateGraph(State).add_sequence([retrieve, generate]) # Dam bao "ket noi": "retrieve" chay truoc, "generate" chay sau.
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


response = graph.invoke({"question": "Hướng dẫn tôi đặt lại mật khẩu người dùng"})
response = dict(response)

# print(f"CONTEXT: {response['context']}")
print(f":ANSWER {response['answer']}")
# print(response)







