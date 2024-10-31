from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import uuid

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
