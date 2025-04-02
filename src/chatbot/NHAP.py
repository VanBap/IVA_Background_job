from langchain.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import requests


class WebWithImageLoader(WebBaseLoader):
    def _scrape(self, url: str, bs_kwargs=None, **kwargs):
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))

        sections = []
        current_section = {"title": "Giới thiệu", "content": ""}

        # Duyệt qua tất cả các phần tử trên trang
        for element in soup.find_all(["h2", "h3", "p", "ol", "ul", "img"]):
            if element.name in ["h2", "h3"]:  # Nếu gặp tiêu đề mới
                if current_section["content"]:  # Lưu lại phần trước nếu có nội dung
                    sections.append(current_section)
                current_section = {"title": element.text.strip(), "content": ""}

            elif element.name in ["p", "ol", "ul"]:  # Nếu là đoạn văn bản hoặc danh sách
                current_section["content"] += "\n" + element.get_text(separator="\n", strip=True)

            elif element.name == "img":  # Nếu gặp ảnh, thêm vào nội dung dưới dạng Markdown
                img_url = element.get("src")
                alt_text = element.get("alt", "image")
                if img_url:
                    current_section["content"] += f"\n![{alt_text}]({img_url})"

        if current_section["content"]:  # Thêm phần cuối cùng vào danh sách
            sections.append(current_section)

        return sections

    def lazy_load(self):
        """Tải nội dung trang web theo từng mục lục (h2, h3, ảnh)."""
        for path in self.web_paths:
            sections = self._scrape(path)
            for section in sections:
                yield Document(
                    page_content=section["content"],
                    metadata={"source": path, "title": section["title"]}
                )
