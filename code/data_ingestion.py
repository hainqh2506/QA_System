import os
import glob
import re
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional, Tuple

class PDFProcessor:
    """
    Lớp xử lý tài liệu PDF, bao gồm việc tải và làm sạch nội dung.
    """

    def __init__(self, path: str):
        """
        Khởi tạo đối tượng PDFProcessor.

        :param path: Đường dẫn đến thư mục chứa các file PDF.
        """
        self.path = path

    def load_and_clean(self, file_path: Optional[str] = None, cleaning_rules: Optional[List[Tuple[str, str]]] = None) -> List:
        """
        Tải và làm sạch các tài liệu PDF.

        :param file_path: Đường dẫn đến file PDF cụ thể (tùy chọn).
        :param cleaning_rules: Danh sách các quy tắc làm sạch (tùy chọn).
        :return: Danh sách các tài liệu đã được tải và làm sạch.
        """
        docs = []
        # Nếu có file_path, chỉ xử lý file đó; nếu không, xử lý tất cả file PDF trong thư mục
        pdf_files = [file_path] if file_path else glob.glob(f'{self.path}/*.pdf')
        
        for pdf in tqdm(pdf_files, desc="Đang tải và làm sạch file PDF"):
            if os.path.exists(pdf) and pdf.lower().endswith('.pdf'):
                loader = PyPDFLoader(pdf)
                loaded_docs = loader.load()
                file_name = os.path.basename(pdf)
                
                for doc in loaded_docs:
                    doc.metadata["source"] = file_name
                    self._clean_content(doc, cleaning_rules)
                
                docs.extend(loaded_docs)
        
        return docs

    @staticmethod
    def _clean_content(doc, cleaning_rules: Optional[List[Tuple[str, str]]] = None):
        """
        Làm sạch nội dung của một tài liệu.

        :param doc: Tài liệu cần làm sạch.
        :param cleaning_rules: Danh sách các quy tắc làm sạch (tùy chọn).
        """
        if cleaning_rules is None:
            cleaning_rules = [
                (r'\n+', ' '),  # Thay thế các ký tự xuống dòng bằng khoảng trắng
                (r'\s+', ' '),  # Loại bỏ khoảng trắng thừa
                (r'\[[0-9]+\]', '')  # Loại bỏ các số trong ngoặc vuông
            ]
        
        for pattern, replacement in cleaning_rules:
            doc.page_content = re.sub(pattern, replacement, doc.page_content)

def setup_knowledge_base(pdf_directory: str, specific_pdf: Optional[str] = None) -> List:
    """
    Thiết lập cơ sở tri thức từ các tài liệu PDF.
    How to use: cleaned_documents = setup_knowledge_base('đường/dẫn/đến/thư_mục_pdf', 'đường/dẫn/đến/file_cụ_thể.pdf')

    :param pdf_directory: Đường dẫn đến thư mục chứa các file PDF.
    :param specific_pdf: Đường dẫn đến file PDF cụ thể (tùy chọn).
    :return: documents: Danh sách các tài liệu đã được tải và làm sạch.
    """
    processor = PDFProcessor(pdf_directory)
    return processor.load_and_clean(specific_pdf)

# Cách sử dụng
# cleaned_documents = setup_knowledge_base('đường/dẫn/đến/thư_mục_pdf', 'đường/dẫn/đến/file_cụ_thể.pdf')