import tiktoken
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer
from langchain_together import ChatTogether , Together
from dotenv import load_dotenv
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pyvi.ViTokenizer import tokenize
from typing import List, Optional
from langchain_core.embeddings import Embeddings
import numpy as np
# Tạo wrapper class cho SentenceTransformer

class VietnameseEmbeddings(Embeddings):
    """Singleton Embeddings for Vietnamese using SentenceTransformer."""
    _instance: Optional['VietnameseEmbeddings'] = None

    def __new__(cls, model_name: str = "dangvantuan/vietnamese-embedding"): #"dangvantuan/vietnamese-embedding" or "keepitreal/vietnamese-sbert"
        # Nếu chưa có instance, tạo mới
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Khởi tạo model chỉ một lần
            cls._instance._initialize_model(model_name)
        return cls._instance

    def _initialize_model(self, model_name: str):
        try:
            print(f"Initializing Vietnamese embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
    def embed_text(self, text: str)-> np.ndarray:  #using in RAPTOR (np.ndarray)
        return self.model.encode(text, convert_to_numpy=True)

# Hàm tải tokenizer
def load_tokenizer(tokenizer_model: str = "keepitreal/vietnamese-sbert"):
    """
    Hàm tải và khởi tạo tokenizer.
    
    Tham số:
    - tokenizer_model: Tên mô hình tokenizer. Mặc định là "keepitreal/vietnamese-sbert".
    
    Trả về: Đối tượng tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_model)
def load_tokenizer2(tokenizer_model: str = "dangvantuan/vietnamese-embedding"):
    return AutoTokenizer.from_pretrained(tokenizer_model)

# Hàm tải mô hình embedding
def load_embedding_model_VN(embedding_model: str = "keepitreal/vietnamese-sbert"):
    """
    Hàm tải và khởi tạo mô hình embedding.
    
    Tham số:
    - embedding_model: Tên mô hình embedding.
    
    Trả về: Đối tượng VietnameseEmbeddings.
    """
    return VietnameseEmbeddings(embedding_model)
def load_embedding_model_VN2(embedding_model: str = "dangvantuan/vietnamese-embedding"):
    """
    Hàm tải và khởi tạo mô hình embedding.
    
    Tham số:
    - embedding_model: Tên mô hình embedding.
    
    Trả về: Đối tượng VietnameseEmbeddings.
    """
    return VietnameseEmbeddings(embedding_model)

# Hàm tải mô hình embedding
def load_embedding_model(embedding_model: str = "dangvantuan/vietnamese-embedding"):
    """
    Hàm tải và khởi tạo mô hình embedding.
    
    Tham số:
    - embedding_model: Tên mô hình embedding. Mặc định là "dangvantuan/vietnamese-embedding".
    
    Trả về: Đối tượng embedding model.
    """
    return HuggingFaceEmbeddings(model_name=embedding_model)

def load_gpt4o_mini_model( model_name: str = "gpt-4o-mini"):
    """
    Hàm tải và khởi tạo mô hình chat từ ChatOpenAI.
    
    Trả về: Đối tượng chat model.
    """
    # Load API keys từ .env file
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name ,temperature=0.7)
# gemini
def load_gemini(model_name: str = "gemini-1.5-flash"): #-8b
        # Load API keys từ .env file
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.5,
    max_tokens=512,
    api_key = GEMINI_API_KEY
    # other params...
)
    return llm
# Hàm tải mô hình chat (nếu cần sử dụng)
def load_gemini2(model_name: str = "gemini-2.0-flash-exp"): #-8b
        # Load API keys từ .env file
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY1")
    
    llm = ChatGoogleGenerativeAI(
    model=model_name,
    api_key = GEMINI_API_KEY,
    temperature=0.7
    # other params...
)
    
    return llm
# Hàm tải mô hình chat (nếu cần sử dụng)
def load_gemini15(model_name: str = "gemini-1.5-flash"): #-8b
        # Load API keys từ .env file
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY1") # doi api o day
    
    llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    api_key = GEMINI_API_KEY
    # other params...
)
    
    return llm
def load_groq_model(chat_model_name: str = "llama-3.3-70b-versatile"):  #lama3-8b-8192 llama-3.1-70b-versatile llama-3.3-70b-versatile llama-3.1-8b-instant  
  # Load API keys từ .env file
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    return ChatGroq(api_key=GROQ_API_KEY, model_name=chat_model_name)


def load_together_model(model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):  
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    return ChatTogether(api_key=TOGETHER_API_KEY, model_name=model_name)
    
# Hàm tải và khởi tạo tokenizer từ tiktoken (nếu cần)
def load_tiktoken(tokenizer_name: str = "o200k_base"):  #cl100k_base
    """
    Hàm tải và khởi tạo tokenizer từ tiktoken.
    
    Tham số:
    - tokenizer_name: Tên tokenizer của tiktoken. Mặc định là "cl100k_base".
    
    Trả về: Đối tượng tokenizer.
    """
    return tiktoken.get_encoding(tokenizer_name)
