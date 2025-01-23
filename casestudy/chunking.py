from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import uuid
from model_config import load_tokenizer
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
tokenizer = load_tokenizer() #tokenizer_model: Tên mô hình tokenizer. Mặc định là "keepitreal/vietnamese-sbert".
# Tùy chỉnh separators và token limit
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=192,  # Giới hạn theo token
    chunk_overlap=64,  
    separators=["\n\n", "\n", ". ", " "],  # Tùy chỉnh separators
    length_function=lambda text: len(tokenizer.encode(text, truncation=False, max_length=512))
)