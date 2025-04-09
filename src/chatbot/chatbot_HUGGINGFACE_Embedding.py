import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from sentence_transformers import CrossEncoder
from typing_extensions import List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores import FAISS

# === Vannhk ===
import chatbot_crawl_data as crawl_data
from splitter import SapoSupportChunker
import time
# import chatbot_crawl_data as crawl_data
#
# from splitter import SapoSupportChunker

VECTOR_DB_DEMO_PATH = os.getenv("VECTOR_DB_DEMO_PATH_HUGGINGFACE")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# === CHECK IF VECTOR DB EXISTED ===
if os.path.exists(VECTOR_DB_DEMO_PATH):
    print("[INFO] Đang tải dữ liệu từ Vector DB đã lưu...")
    with open(VECTOR_DB_DEMO_PATH, "rb") as f:
        vector_store = pickle.load(f)
else:
    print("[INFO] Lần đầu tiên chạy - Đang tải và embedding dữ liệu...")
    start = time.time()
    # url = "https://support.sapo.vn/tim-hieu-ve-don-hang-1"
    # url = "https://support.sapo.vn/ket-noi-kenh-facebook-tren-app-sapo"
    # loader = crawl_data.WebWithImageLoader(web_paths=(url,))
    # documents = loader.load()

    base_url = "https://support.sapo.vn/sapo-retail"
    get_article_links = crawl_data.get_article_links(base_url)
    documents = crawl_data.load_articles(get_article_links)

    print(f"[INFO] Đã tải {len(documents)} tài liệu từ URL")

    chunker = SapoSupportChunker(chunk_size=2000, chunk_overlap=200)
    all_splits = chunker.split_documents(documents)

    print(f"[INFO] Đã tách thành {len(all_splits)} đoạn")

    # In thông tin về một số đoạn đầu tiên để kiểm tra
    for i in range(len(all_splits)):
        doc = all_splits[i]
        print(f"\n--- Đoạn {i + 1} ---")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content)
        has_images = "image link:" in doc.page_content
        print(f"Có hình ảnh: {has_images}")
        if has_images:
            image_links = [line.strip() for line in doc.page_content.split("\n") if "image link:" in line]
            print(f"image_links: {image_links}")
            print(f"Số lượng hình ảnh: {len(image_links)}")


    # === DATA EMBEDDING + VECTOR DB ===
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPEN_API_KEY,base_url="https://models.inference.ai.azure.com")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # ******embedidng_anh =

    vector_store = FAISS.from_documents(all_splits, embeddings)  # Sử dụng FAISS để lưu trữ vector
    # ****** vector_store cho ảnh =

    # === Lưu Vector DB để sử dụng sau ===
    with open(VECTOR_DB_DEMO_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    end = time.time()
    print(f"[CRAWL + SPLIT + EMBEDDING DATA] {end-start} seconds")
    print("[INFO] Lưu trữ Vector DB thành công!")


template = """ 
Hướng dẫn chi tiết theo yêu cầu người hỏi. Nếu trong văn bản có đường dẫn hình ảnh (image_link), hãy đưa chúng vào câu trả lời để minh họa rõ hơn.

Câu hỏi gốc: {question}
Thông tin chi tiết từ hệ thống: {context}

Trong câu trả lời của bạn:
1. Nếu có thông tin về các bước, hãy trình bày theo từng bước rõ ràng
2. Đối với mỗi bước mà có hình ảnh liên quan (có dòng "image link:"), hãy đưa link ảnh đó vào cuối phần giải thích của bước đó
3. Định dạng câu trả lời giống như ví dụ minh họa sau:

**Bước 1: [Tên bước]**
[Giải thích chi tiết]
![Alt text](image_link) (nếu có)

**Bước 2: [Tên bước]**
[Giải thích chi tiết]
![Alt text](image_link) (nếu có)

Chỉ trả lời dựa trên thông tin được cung cấp, không thêm thông tin không có trong văn bản gốc.
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
    start = time.time()
    retrieved_docs = vector_store.similarity_search(state["question"], k=7) #default k = 4

    # Check co Image hay khong
    # has_images = any("image_link:" in doc.page_content for doc in retrieved_docs)
    # if has_images:
    #     print("[DEBUG] Kết quả retrieved có chứa image links")

    # # === Rerank với cross-encoder ===
    # reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # doc_scores = reranker.predict([(state["question"], doc.page_content) for doc in retrieved_docs])
    #
    # # Lấy top documents sau reranking
    # reranked_docs = [doc for _, doc in sorted(zip(doc_scores, retrieved_docs), key=lambda x: x[0], reverse=True)][:4]
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


response = graph.invoke({"question": "Hướng dẫn kết nối kênh Facebook trên App Sapo"})
response = dict(response)

# print(f"CONTEXT: {response['context']}")
print(f":ANSWER {response['answer']}")
# print(response)







