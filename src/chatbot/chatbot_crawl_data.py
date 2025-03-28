import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


# URL của trang hỗ trợ Sapo Retail
base_url = "https://support.sapo.vn/sapo-retail"

def get_article_links(url):
    # Gửi request và parse HTML
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Trích xuất tất cả các link bài viết
    article_links = []
    for link in soup.find_all("a", href=True, title=True):  # Chỉ lấy các thẻ <a> có href và title
        href = link["href"]
        if "sapo" in href:  # Lọc ra các bài viết liên quan đến Sapo Retail
            full_url = f"https://support.sapo.vn{href}" if href.startswith("/") else href
            article_links.append(full_url)

    # Kiểm tra danh sách URL bài viết thu thập được
    print("Danh sách bài viết:")
    for url in article_links:
        print(url)

    return article_links

# === Load nội dung từ các trang con ===
def load_articles(urls):
    loaders = [WebBaseLoader(web_paths=(url,)) for url in urls]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs