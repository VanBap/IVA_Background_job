import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
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

# === WebWithImageLoader ===
class WebWithImageLoader(WebBaseLoader):
    def _scrape(self, url: str, bs_kwargs=None, **kwargs):
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))

        # Lấy nội dung chỉ trong class="page-detail page-detail-guide"
        # content_div = soup.find("div", class_="page-detail page-detail-guide")
        # if not content_div:
        #     return soup, ""  # Trả về rỗng nếu không tìm thấy

        # Chuyển đổi hình ảnh thành định dạng đặc biệt "image link: URL"
        for img in soup.find_all("img"):
            img_url = img.get("src")
            if img_url:
                # Chuyển đường dẫn tương đối thành tuyệt đối nếu cần
                if img_url.startswith('/'):
                    base_url = '/'.join(url.split('/')[:3])  # http://domain.com
                    img_url = base_url + img_url

                # Tạo text mới với định dạng "image link: URL"
                img_text = soup.new_string(f'\nimage link: "{img_url}"\n')
                img.replace_with(img_text)

        # Extract text with inline image links
        text = soup.get_text(separator="\n", strip=True)

        return soup, text  # Return both soup and modified text

    def lazy_load(self):
        for path in self.web_paths:
            soup, text = self._scrape(path, bs_kwargs=self.bs_kwargs)
            yield Document(page_content=text, metadata={"source": path})

