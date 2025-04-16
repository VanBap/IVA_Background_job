import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional, Dict, Any
import tiktoken

# ============================== Option 1: Chunk by Header ==================================
class SapoSupportChunker:

    def __init__(self, chunk_size=2000, chunk_overlap=200):
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
        image_links = []
        for line in text.split("\n"):
            if "image_link" in line:
                image_links.append(line.strip())
        return image_links

    def split_by_headers(self, text):
        """Tách text thành các đoạn dựa vào các header"""

        # # Chuẩn bị regex pattern để nhận diện các header
        # combined_pattern = "|".join(self.header_patterns)
        # pattern = f"({combined_pattern})"

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
        header_to_images = {} # Dict: mapping header -> img_urls

        # Lần đầu, tạo mapping giữa header và images
        for chunk in chunks:
            lines = chunk.split("\n")
            current_header = None

            for line in lines:
                # Xác định header
                is_header = any(re.search(pattern, line) for pattern in self.header_patterns)
                if is_header:
                    current_header = line.strip()
                    print(f"=== CURRENT HEADER: {current_header}")

                # Thu thập link hình ảnh cho header hiện tại
                if current_header and "image_link" in line:
                    if current_header not in header_to_images:
                        header_to_images[current_header] = []
                    header_to_images[current_header].append(line.strip())
                    print(f"=== header_to_images: {header_to_images}")

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
            has_images = any("image_link" in line for line in chunk_lines)

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

        # Bước 2: Tách tiếp các section dài hơn chunksize
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
                    if i == len(sub_chunks) - 1 and not any("image_link" in line for line in chunk.split("\n")):
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
            chunk_metadata["has_images"] = "image_link" in chunk

            # Đếm số lượng hình ảnh trong chunk
            image_count = chunk.count("image_link")
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

# ================================= Option 2: Chunk by entire website ===================================
class UrlBasedChunker:
    def __init__(self, chunk_size=6000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        # Tokenizer de dem Tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        """ Dem so luong tokens trong 1 doan text"""
        return len(self.tokenizer.encode(text))

    def create_documents(self, docs: List[Document]) -> List[Document]:
        processed_docs = []
        for doc in docs:
            content = doc.page_content
            source_url = doc.metadata.get("source", "unknown")
            tokens_count = self.count_tokens(content)

            # Kiem tra xem noi dung co vuot qua gioi han khong?
            if tokens_count <= self.chunk_size:
                metadata = doc.metadata.copy()
                metadata["has_images"] = "image_link" in content
                metadata["image_count"] = content.count("image_link")
                metadata["is_split"] = False
                metadata["original_url"] = source_url
                metadata["tokens_count"] = tokens_count

                processed_docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
                print(f"URL {source_url} - {tokens_count} tokens - giữ nguyên")
            else:
                chunks = self.text_splitter.split_text(content)

                for i, chunk in enumerate(chunks):
                    chunk_tokens = self.count_tokens(chunk)

                    metadata = doc.metadata.copy()
                    metadata["chunk_id"] = i
                    metadata["total_chunks"] = len(chunks)
                    metadata["has_images"] = "image_link" in chunk
                    metadata["image_count"] = chunk.count("image_link")
                    metadata["is_split"] = True
                    metadata["original_url"] = source_url
                    metadata["tokens_count"] = chunk_tokens

                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))

                print(f"URL {source_url} - {tokens_count} tokens - được chia thành {len(chunks)} chunks")

        return processed_docs

# ================================= Option 3: Chunk by HierarchicalPDF ===================================
import fitz  # PyMuPDF

class HierarchicalPDFChunker:
    """ Chunk PDF theo cau truc phan cap """

    def __init__(self, max_tokens=8000, overlap_tokens=200):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Mở rộng các mẫu nhận diện tiêu đề phân cấp để nhận diện đúng các tiêu đề con
        self.header_patterns = [
            # Tiêu đề dạng "3.7.1. Xem danh sách event đếm theo quy tắc"
            r'^(\d+\.\d+\.\d+\.)\s+(.+)$',
            # Tiêu đề dạng "3.7. Event đếm theo quy tắc"
            r'^(\d+\.\d+\.)\s+(.+)$',
            # Tiêu đề dạng "3. Hướng dẫn sử dụng"
            r'^(\d+\.)\s+(.+)$',
            # Mẫu: "a. Tiêu đề" - Tiêu đề phụ
            r'^([a-z])[\.|\)]\s+(.+)$',

            # # Mẫu: "• Tiêu đề" hoặc "- Tiêu đề" - Đánh dấu điểm
            # r'^([-•])\s+(.+)$',
            # # Tiêu đề "Bước 1:" hoặc "Bước 1."
            # r'^(Bước\s+\d+)[:\.](.+)$',
            # # Tiêu đề "Lưu ý:" hoặc các tiêu đề tương tự
            # r'^(Lưu\s+ý:)(.+)$'
        ]

        # === Cai dat cho xu ly anh


    def count_tokens(self, text: str) -> int:
        """ Đếm số lượng tokens trong một đoạn text"""
        return len(self.tokenizer.encode(text))

    def extract_hierarchical_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Phân tích văn bản và trích xuất cấu trúc phân cấp.
        Cải tiến để nhận diện tốt hơn các tiêu đề có mức độ phân cấp sâu.
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                # Thêm dòng trống vào nội dung hiện tại nếu có
                if current_content:
                    current_content.append("")
                continue

            # Kiểm tra xem dòng hiện tại có phải là một tiêu đề mới không
            match = None
            matched_pattern = None
            for pattern in self.header_patterns:
                match = re.match(pattern, line)
                if match:
                    matched_pattern = pattern
                    break

            if match:
                # Lưu section hiện tại trước khi bắt đầu section mới
                if current_section and current_content:
                    sections.append({
                        "level_id": current_section["level_id"],
                        "level_type": current_section["level_type"],
                        "depth": current_section["depth"],
                        "title": current_section["title"],
                        "content": "\n".join(current_content)
                    })

                # Phân tích level ID và xác định loại cùng độ sâu
                if re.match(r'^\d+\.\d+\.\d+\.$', match.group(1)):
                    # Kiểu "3.7.1."
                    level_id = match.group(1).rstrip('.')
                    level_type = "numeric"
                    depth = 3  # Độ sâu mức 3
                    title = match.group(2)
                elif re.match(r'^\d+\.\d+\.$', match.group(1)):
                    # Kiểu "3.7."
                    level_id = match.group(1).rstrip('.')
                    level_type = "numeric"
                    depth = 2  # Độ sâu mức 2
                    title = match.group(2)
                elif re.match(r'^\d+\.$', match.group(1)):
                    # Kiểu "3."
                    level_id = match.group(1).rstrip('.')
                    level_type = "numeric"
                    depth = 1  # Độ sâu mức 1
                    title = match.group(2)
                elif re.match(r'^[a-z][\.|\)]$', match.group(1)):
                    # Kiểu chữ cái: "a.", "b."
                    level_id = match.group(1).rstrip('.')
                    level_type = "alphabetic"
                    depth = 10  # Giả định mức thấp nhất
                    title = match.group(2)
                elif re.match(r'^Bước\s+\d+', match.group(1)):
                    # Kiểu "Bước 1:"
                    level_id = match.group(1)
                    level_type = "step"
                    depth = 11  # Giả định thấp hơn alphabetic
                    title = match.group(2) if len(match.groups()) > 1 else ""
                elif re.match(r'^Lưu\s+ý:', match.group(1)):
                    # Kiểu "Lưu ý:"
                    level_id = match.group(1)
                    level_type = "note"
                    depth = 12  # Giả định thấp hơn step
                    title = match.group(2) if len(match.groups()) > 1 else ""
                else:
                    # Kiểu bullet: "•", "-"
                    level_id = match.group(1)
                    level_type = "bullet"
                    depth = 13  # Giả định thấp nhất
                    title = match.group(2) if len(match.groups()) > 1 else ""

                # Tạo section mới
                current_section = {
                    "level_id": level_id,
                    "level_type": level_type,
                    "depth": depth,
                    "title": title,
                }
                current_content = []
            elif current_section:
                # Thêm dòng vào nội dung của section hiện tại
                current_content.append(line)
            else:
                # Dòng không thuộc section nào, có thể là phần mở đầu
                pass

        # Thêm section cuối cùng nếu có
        if current_section and current_content:
            sections.append({
                "level_id": current_section["level_id"],
                "level_type": current_section["level_type"],
                "depth": current_section["depth"],
                "title": current_section["title"],
                "content": "\n".join(current_content)
            })

        return sections

    def build_hierarchy_tree(self, sections: List[Dict[str, Any]]) -> Dict:
        """
        Xây dựng một cây phân cấp từ danh sách các section.
        Cải tiến để xử lý tốt hơn các tiêu đề phân cấp sâu.
        """
        # Sắp xếp sections theo chiều sâu để đảm bảo xử lý parent trước children
        sections = sorted(sections, key=lambda x: (x.get("depth", 0), x.get("level_id", "")))

        # Khởi tạo cây với nút gốc ảo
        root = {"children": {}, "sections": []}

        # Dictionary lưu trữ tham chiếu đến các nút trong cây
        nodes = {"root": root}

        # Xác định parent path cho mỗi section
        for section in sections:
            # Xử lý theo loại level
            if section["level_type"] == "numeric":
                # Xác định parent dựa vào level_id
                level_id = section["level_id"]
                parts = level_id.split('.')

                # Node hiện tại
                current_id = level_id

                # Parent ID (bỏ phần tử cuối cùng)
                if len(parts) > 1:
                    parent_parts = parts[:-1]
                    parent_id = '.'.join(parent_parts)
                else:
                    parent_id = "root"
            else:
                # Đối với các loại không phải numeric, tìm parent là section numeric gần nhất
                numeric_sections = [s for s in sections if s["level_type"] == "numeric"]
                numeric_sections = sorted(numeric_sections, key=lambda x: x["depth"], reverse=True)

                # Tìm parent là section numeric có độ sâu nhỏ nhất nhưng lớn hơn section hiện tại
                parent = next((s for s in numeric_sections if s["depth"] < section["depth"]), None)
                parent_id = parent["level_id"] if parent else "root"
                current_id = f"{parent_id}_{section['level_id']}" if parent else section["level_id"]

            # Thêm nút mới vào cây
            if current_id not in nodes:
                nodes[current_id] = {"children": {}, "sections": []}

            # Thêm section vào nút tương ứng
            nodes[current_id]["sections"].append(section)

            # Kết nối nút con với nút cha
            if parent_id in nodes:
                nodes[parent_id]["children"][current_id] = nodes[current_id]
            else:
                # Nếu không tìm thấy parent, gán cho root
                nodes["root"]["children"][current_id] = nodes[current_id]

        return root

    def find_leaf_nodes(self, tree: Dict) -> List[Dict]:
        """
        Tìm tất cả các nút lá (không có con) trong cây phân cấp.
        Đây chính là các section ở mức thấp nhất.
        """
        result = []

        def traverse(node, path=""):
            if not node["children"]:  # Nút lá
                # Kết hợp tất cả các section trong nút
                if node["sections"]:
                    for section in node["sections"]:
                        section_copy = section.copy()
                        section_copy["full_path"] = path
                        result.append(section_copy)
            else:
                # Duyệt qua các nút con
                for child_id, child_node in node["children"].items():
                    new_path = f"{path}.{child_id}" if path else child_id
                    traverse(child_node, new_path)

        traverse(tree)
        return result

    def split_large_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chia nhỏ các phần lớn thành các chunks nhỏ hơn theo giới hạn tokens.
        """
        split_sections = []

        for section in sections:
            content = section["content"]
            token_count = self.count_tokens(content)

            if token_count <= self.max_tokens:
                # Nếu nội dung không vượt quá giới hạn tokens, giữ nguyên
                split_sections.append(section)
            else:
                # Nếu vượt quá, chia nhỏ theo đoạn văn
                paragraphs = re.split(r'\n\s*\n', content)
                chunks = []
                current_chunk = ""
                current_tokens = 0

                for para in paragraphs:
                    para_tokens = self.count_tokens(para)

                    if current_tokens + para_tokens + 1 <= self.max_tokens:  # +1 for the newline
                        if current_chunk:
                            current_chunk += "\n\n" + para
                        else:
                            current_chunk = para
                        current_tokens += para_tokens + (2 if current_chunk else 0)  # +2 for "\n\n"
                    else:
                        # Nếu đoạn hiện tại không vừa với chunk hiện tại
                        if current_chunk:
                            chunks.append(current_chunk)

                            # Thêm phần overlap
                            last_para = current_chunk.split("\n\n")[-1] if "\n\n" in current_chunk else ""
                            if self.count_tokens(last_para) <= self.overlap_tokens:
                                overlap_text = last_para
                            else:
                                # Lấy phần cuối vừa đủ để không vượt quá overlap_tokens
                                words = last_para.split()
                                overlap_text = ""
                                for word in reversed(words):
                                    if self.count_tokens(word + " " + overlap_text) <= self.overlap_tokens:
                                        overlap_text = word + " " + overlap_text
                                    else:
                                        break

                            # Bắt đầu chunk mới với overlap và đoạn hiện tại
                            current_chunk = overlap_text + "\n\n" + para if overlap_text else para
                            current_tokens = self.count_tokens(current_chunk)
                        else:
                            # Nếu một đoạn đơn lẻ vượt quá giới hạn tokens
                            words = para.split()
                            first_half = []
                            second_half = []
                            current_count = 0

                            # Chia đoạn thành hai phần gần bằng nhau dựa trên tokens
                            for word in words:
                                word_tokens = self.count_tokens(word + " ")
                                if current_count + word_tokens <= self.max_tokens // 2:
                                    first_half.append(word)
                                    current_count += word_tokens
                                else:
                                    second_half.append(word)

                            # Thêm nửa đầu vào chunks
                            chunks.append(" ".join(first_half))

                            # Bắt đầu chunk mới với nửa sau
                            current_chunk = " ".join(second_half)
                            current_tokens = self.count_tokens(current_chunk)

                # Thêm chunk cuối cùng nếu có
                if current_chunk:
                    chunks.append(current_chunk)

                # Tạo các sections mới từ các chunks
                for i, chunk in enumerate(chunks):
                    new_section = section.copy()
                    new_section["content"] = chunk
                    new_section["chunk_id"] = i
                    new_section["total_chunks"] = len(chunks)
                    split_sections.append(new_section)

        return split_sections

    def process_pdf(self, pdf_path: str, start_page: int = None, end_page: int = None) -> List[Document]:
        """
        Xử lý file PDF và trả về danh sách các Document đã được chia theo cấu trúc phân cấp.
        """
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Lọc các trang theo chỉ định
        if start_page is not None:
            start_page = max(0, min(start_page - 1, len(pages) - 1))
            if end_page is None:
                pages = pages[start_page:]
            else:
                end_page = max(start_page, min(end_page - 1, len(pages) - 1))
                pages = pages[start_page:end_page + 1]  # +1 vì Python slicing không bao gồm end

        # Log thông tin
        print(f"Đang xử lý PDF từ trang {start_page + 1 if start_page is not None else 1} "
              f"đến trang {end_page + 1 if end_page is not None else len(pages)} "
              f"của file {pdf_path}")

        # Gộp nội dung các trang
        full_text = "\n".join([page.page_content for page in pages])

        # Trích xuất cấu trúc phân cấp
        sections = self.extract_hierarchical_structure(full_text)

        # Xây dựng cây phân cấp
        hierarchy_tree = self.build_hierarchy_tree(sections)

        # Tìm các nút lá (mức thấp nhất)
        leaf_sections = self.find_leaf_nodes(hierarchy_tree)

        # Chia nhỏ các section lớn
        final_sections = self.split_large_sections(leaf_sections)

        # Tạo các Document
        documents = []
        for section in final_sections:
            # Tạo tiêu đề đầy đủ
            if section["level_type"] == "numeric":
                title = f"{section['level_id']} {section['title']}"
            elif section["level_type"] == "alphabetic":
                title = f"{section['level_id']}. {section['title']}"
            else:  # bullet
                title = f"{section['level_id']} {section['title']}"

            # Thêm thông tin chunk nếu có
            chunk_info = ""
            if "chunk_id" in section:
                chunk_info = f" (Phần {section['chunk_id'] + 1}/{section['total_chunks']})"

            # Tạo nội dung đầy đủ
            page_content = f"{title}{chunk_info}\n\n{section['content']}"

            # Tạo metadata
            metadata = {
                "source": pdf_path,
                "section_level": section["level_id"],
                "section_title": section["title"],
                "section_type": section["level_type"],
                "full_path": section.get("full_path", section["level_id"]),
                "tokens_count": self.count_tokens(page_content)
            }

            # Thêm thông tin chunk nếu có
            if "chunk_id" in section:
                metadata["chunk_id"] = section["chunk_id"]
                metadata["total_chunks"] = section["total_chunks"]

            # Tạo Document
            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))

        return documents