from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


class SapoSupportChunker:
    """
    Bộ chia đoạn văn bản chuyên biệt cho trang Sapo Support,
    tập trung vào việc giữ nguyên các hướng dẫn theo bước và link hình ảnh
    """

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Pattern cho các header và section
        self.header_patterns = [
            r"\*\*[^*]+\*\*",  # **Header**
            r"\d+\.\s+[A-Z]",  # 1. Header
            r"Bước \d+:",  # Bước 1:
            r"Bước \d+\.",  # Bước 1.
            r"#+ ",  # Markdown headers
        ]

    def extract_images_from_text(self, text):
        """Trích xuất tất cả các link hình ảnh từ text"""
        image_links = []
        for line in text.split("\n"):
            if "image link:" in line:
                image_links.append(line.strip())
        return image_links

    def split_by_headers(self, text):
        """Tách text thành các đoạn dựa vào các header"""
        # Chuẩn bị regex pattern để nhận diện các header
        combined_pattern = "|".join(self.header_patterns)
        pattern = f"({combined_pattern})"

        # Tách văn bản bằng regex
        import re
        sections = []
        current_section = ""

        lines = text.split("\n")
        in_section = False

        for line in lines:
            # Kiểm tra xem dòng hiện tại có phải là header không
            is_header = any(re.search(pattern, line) for pattern in self.header_patterns)

            if is_header and current_section:
                # Lưu section hiện tại và bắt đầu section mới
                sections.append(current_section)
                current_section = line + "\n"
            else:
                # Thêm dòng vào section hiện tại
                current_section += line + "\n"

        # Thêm section cuối cùng
        if current_section:
            sections.append(current_section)

        return sections

    def ensure_images_in_chunks(self, chunks):
        """
        Đảm bảo mỗi chunk chứa đúng link hình ảnh liên quan đến nội dung đó
        """
        processed_chunks = []
        header_to_images = {}

        # Lần đầu, tạo mapping giữa header và images
        for chunk in chunks:
            lines = chunk.split("\n")
            current_header = None

            for line in lines:
                # Xác định header
                is_header = any(re.search(pattern, line) for pattern in self.header_patterns)
                if is_header:
                    current_header = line.strip()

                # Thu thập link hình ảnh cho header hiện tại
                if current_header and "image link:" in line:
                    if current_header not in header_to_images:
                        header_to_images[current_header] = []
                    header_to_images[current_header].append(line.strip())

        # Lần thứ hai, thêm link hình ảnh vào mỗi chunk
        for chunk in chunks:
            chunk_header = None
            chunk_lines = chunk.split("\n")

            for line in chunk_lines:
                is_header = any(re.search(pattern, line) for pattern in self.header_patterns)
                if is_header:
                    chunk_header = line.strip()
                    break

            # Kiểm tra xem chunk đã có link hình ảnh chưa
            has_images = any("image link:" in line for line in chunk_lines)

            # Nếu chunk có header nhưng không có hình ảnh, thêm vào
            if chunk_header and not has_images and chunk_header in header_to_images:
                chunk += "\n" + "\n".join(header_to_images[chunk_header])

            processed_chunks.append(chunk)

        return processed_chunks

    def split_text(self, text):
        """
        Tách text thành các đoạn, đảm bảo giữ nguyên cấu trúc và link hình ảnh
        """
        # Bước 1: Tách theo headers
        header_sections = self.split_by_headers(text)

        # Bước 2: Tách tiếp các section dài
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        all_chunks = []
        for section in header_sections:
            # Kiểm tra độ dài của section
            if len(section) > self.chunk_size:
                # Thu thập hình ảnh trước khi tách
                images = self.extract_images_from_text(section)

                # Tách section thành các đoạn nhỏ hơn
                sub_chunks = splitter.split_text(section)

                # Đảm bảo link hình ảnh được giữ nguyên trong các chunk con
                for i, chunk in enumerate(sub_chunks):
                    # Nếu đây là chunk cuối và chưa có hình ảnh, thêm vào
                    if i == len(sub_chunks) - 1 and not any("image link:" in line for line in chunk.split("\n")):
                        for img in images:
                            if img not in chunk:
                                chunk += f"\n{img}"

                    all_chunks.append(chunk)
            else:
                all_chunks.append(section)

        # Bước 3: Đảm bảo mỗi chunk chứa đúng link hình ảnh
        processed_chunks = self.ensure_images_in_chunks(all_chunks)

        return processed_chunks

    def create_documents(self, text, metadata=None):
        """Tạo danh sách Document từ text"""
        chunks = self.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            # Tạo metadata mới cho mỗi chunk
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk"] = i
            chunk_metadata["has_images"] = "image link:" in chunk

            # Đếm số lượng hình ảnh trong chunk
            image_count = chunk.count("image link:")
            chunk_metadata["image_count"] = image_count

            documents.append(Document(page_content=chunk, metadata=chunk_metadata))

        return documents

    def split_documents(self, documents):
        """Tách danh sách Document thành các chunks nhỏ hơn"""
        all_docs = []

        for doc in documents:
            chunks = self.create_documents(doc.page_content, doc.metadata)
            all_docs.extend(chunks)

        return all_docs