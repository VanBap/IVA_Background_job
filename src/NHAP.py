import requests
from bs4 import BeautifulSoup

# URL của trang hỗ trợ Sapo Retail
base_url = "https://support.sapo.vn/sapo-retail"

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
