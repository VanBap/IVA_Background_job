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

# === Vannhk ===
import chatbot_crawl_data as crawl_data

OPEN_API_KEY = "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"

# === DATA EMBEDDING + VECTOR DB ===
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPEN_API_KEY,base_url="https://models.inference.ai.azure.com")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# URL của trang hỗ trợ Sapo Retail
base_url = "https://support.sapo.vn/sapo-retail"
article_links = crawl_data.get_article_links(base_url)
docs = crawl_data.load_articles(article_links)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"[SPLIT] All splits: {len(all_splits)}")
print("=============================")

# === Index chunks ===
_ = vector_store.add_documents(documents=all_splits)
print("[Store in Vector DB] Done")
# === Define prompt for question-answering ===
# # C1
# prompt = hub.pull("rlm/rag-prompt")



# === Choose LLM model ===
llm = ChatOpenAI(model_name="gpt-4o",
                 api_key=OPEN_API_KEY,
                 base_url="https://models.inference.ai.azure.com",
                 temperature=0.0)

# # Custom function to extract image URLs from loaded documents
# def extract_image_urls(docs):
#     image_urls = {}
#     for doc in docs:
#         # Use BeautifulSoup to find image URLs in the document
#         soup = BeautifulSoup(doc.page_content, 'html.parser')
#         images = soup.find_all('img')
#
#         # Store image URLs with a key that helps identify context
#         for i, img in enumerate(images):
#             key = f"img_{len(image_urls) + 1}"
#             image_urls[key] = img.get('src', '')
#
#     return image_urls

# Enhanced function to extract and classify image URLs
def extract_structured_image_urls(docs):
    image_categories = {
        "access_screen": [],
        "product_classification": [],
        "product_types": [],
        "additional_details": []
    }

    for doc in docs:
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        images = soup.find_all('img')

        # Categorize images based on surrounding context
        for img in images:
            # Basic categorization logic (can be enhanced)
            alt_text = img.get('alt', '').lower()
            src = img.get('src', '')

            if "login" in alt_text or "truy cập" in alt_text:
                image_categories["access_screen"].append(src)
            elif "phân loại" in alt_text or "classification" in alt_text:
                image_categories["product_classification"].append(src)
            elif "loại" in alt_text or "tags" in alt_text or "nhãn hiệu" in alt_text:
                image_categories["product_types"].append(src)
            else:
                image_categories["additional_details"].append(src)

    return image_categories


# Modified prompt template to include image references
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible.
#
# Context Images Available: {image_urls}
#
# {context}
#
# Question: {question}
#
# Helpful Answer with Image References:"""
# prompt = PromptTemplate.from_template(template)

# Enhanced prompt template for detailed, step-by-step guidance
detailed_template = """Hướng dẫn chi tiết thêm mới sản phẩm trên Sapo:

**Bước 1: Truy cập màn hình thêm mới sản phẩm**
Bạn thực hiện các bước sau:
1. Đăng nhập vào trang quản lý phần mềm Sapo
2. Chọn mục **Sản phẩm**
3. Vào **Danh sách sản phẩm**
4. Nhấn nút **Tạo sản phẩm**
{access_screen_images}

**Bước 2: Phân loại sản phẩm**
a. **Lựa chọn phân loại sản phẩm cần thêm mới:**
   - Chọn danh mục phù hợp với sản phẩm của bạn
   - Có thể tạo danh mục mới nếu chưa tồn tại
{product_classification_images}

b. **Cập nhật loại / nhãn hiệu / tags sản phẩm:**
   - Chọn loại sản phẩm từ danh sách có sẵn
   - Thêm nhãn hiệu (nếu có)
   - Gắn tags để dễ dàng tìm kiếm
{product_types_images}

**Bước 3: Nhập thông tin chi tiết sản phẩm**
1. Điền tên sản phẩm
2. Nhập mã sản phẩm
3. Cập nhật giá bán
4. Quản lý tồn kho
{additional_details_images}

Ghi chú: Luôn kiểm tra kỹ thông tin trước khi lưu sản phẩm.

Câu hỏi gốc: {question}
Thông tin chi tiết từ hệ thống: {context}
"""


# Modify the State to include image URLs
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    image_urls: dict


# Modified retrieve function to capture image URLs
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    image_urls = extract_structured_image_urls(retrieved_docs)
    return {"context": retrieved_docs, "image_urls": image_urls}


# Updated generate function
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    image_urls = state["image_urls"]

    # Format image URLs for each section
    access_screen_images = "\n".join([f"- Ảnh hướng dẫn: {img}" for img in image_urls.get("access_screen", [])])
    product_classification_images = "\n".join(
        [f"- Ảnh phân loại: {img}" for img in image_urls.get("product_classification", [])])
    product_types_images = "\n".join([f"- Ảnh loại sản phẩm: {img}" for img in image_urls.get("product_types", [])])
    additional_details_images = "\n".join(
        [f"- Ảnh chi tiết: {img}" for img in image_urls.get("additional_details", [])])

    # Create prompt with structured images
    prompt_with_images = PromptTemplate.from_template(detailed_template)
    messages = prompt_with_images.invoke({
        "question": state["question"],
        "context": docs_content,
        "access_screen_images": access_screen_images,
        "product_classification_images": product_classification_images,
        "product_types_images": product_types_images,
        "additional_details_images": additional_details_images
    })

    response = llm.invoke(messages)

    return {"answer": response.content}
# Rest of the code remains the same...
# Compile application and test (CONTROL FLOW)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "Hướng dẫn tôi xử lý đơn hàng"})
# print(f"response: {response}")
# print(f"Context: {response['context']}")
print(f"answer: {response['answer']}")