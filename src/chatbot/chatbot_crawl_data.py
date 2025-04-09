import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# URL của trang hỗ trợ Sapo Retail
base_url = "https://support.sapo.vn/sapo-retail"


def get_article_links(base_url):
    # Gửi request và parse HTML
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Trích xuất tất cả các link bài viết
    article_links = []
    for link in soup.find_all("a", href=True, title=True):  # Chỉ lấy các thẻ <a> có href và title
        href = link["href"]
        if href == "/":
            continue

        full_url = f"https://support.sapo.vn{href}" if href.startswith("/") else href
        article_links.append(full_url)

    article_links.append("https://support.sapo.vn/ket-noi-kenh-facebook-tren-app-sapo")
    article_links.append("https://support.sapo.vn/shopee-chat")

    print("Danh sách url:")
    for url in article_links:
        print(url)

    return article_links


# === WebWithImageLoader ===
class WebWithImageLoader(WebBaseLoader):
    def _scrape(self, url: str, bs_kwargs=None, **kwargs):
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))

        # Lấy nội dung chỉ trong class="page-detail page-detail-guide"
        content_div = soup.find("div", class_="col-xl-9 col-lg-8 col-12 detail-content")
        if not content_div:
            return soup, ""  # Trả về rỗng nếu không tìm thấy

        # Chuyển đổi hình ảnh thành định dạng đặc biệt "image link: URL"
        for img in content_div.find_all("img"):
            img_url = img.get("src")
            print(f"img_url la: {img_url}")
            if img_url:
                # Chuyển đường dẫn tương đối thành tuyệt đối nếu cần
                if img_url.startswith('/'):
                    base_url = '/'.join(url.split('/')[:1])  # http://domain.com
                    img_url = base_url + img_url

                # Tạo text mới với định dạng <image_link> URL"
                img_text = soup.new_string(f'\n image_link: "{img_url}"\n')
                print(f"img_text tuong ung la: {img_text}")
                img.replace_with(img_text)

        # Extract text with inline image links
        text = content_div.get_text(separator="\n", strip=True)
        print(f"=== Text: {text}")
        return soup, text  # Return both soup and modified text

    def lazy_load(self):
        for path in self.web_paths:
            soup, text = self._scrape(path, bs_kwargs=self.bs_kwargs)
            yield Document(page_content=text, metadata={"source": path})


# === Load nội dung từ các trang con ===
def load_articles(urls):
    loaders = [WebWithImageLoader(web_paths=(url,)) for url in urls]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


get_article_links(base_url)


