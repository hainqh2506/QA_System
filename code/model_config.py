import tiktoken
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pyvi.ViTokenizer import tokenize
# Hàm tải tokenizer
def load_tokenizer(tokenizer_model: str = "keepitreal/vietnamese-sbert"):
    """
    Hàm tải và khởi tạo tokenizer.
    
    Tham số:
    - tokenizer_model: Tên mô hình tokenizer. Mặc định là "sentence-transformers/all-MiniLM-L6-v2".
    
    Trả về: Đối tượng tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_model)
def load_tokenizer_Vi():
    """
    Hàm tải và khởi tạo tokenizer.
    
    
    Trả về: Đối tượng tokenizer.
    """
    return tokenize
# Hàm tải mô hình embedding
def load_embedding_model(embedding_model: str = "dangvantuan/vietnamese-embedding"):
    """
    Hàm tải và khởi tạo mô hình embedding.
    
    Tham số:
    - embedding_model: Tên mô hình embedding. Mặc định là "dangvantuan/vietnamese-embedding".
    
    Trả về: Đối tượng embedding model.
    """
    return HuggingFaceEmbeddings(model_name=embedding_model)
# Hàm tải mô hình embedding
def load_embedding_model_VN(embedding_model: str = "keepitreal/vietnamese-sbert"):
    """
    Hàm tải và khởi tạo mô hình embedding.
    
    Tham số:
    - embedding_model: Tên mô hình embedding. Mặc định là "sentence-transformers/all-MiniLM-L6-v2".
    
    Trả về: Đối tượng embedding model.
    """
    return SentenceTransformer(embedding_model)

# Hàm tải mô hình tóm tắt
def load_summarization_model(summarization_model_name: str = 'llama-3.1-70b-versatile', max_tokens: int = 458):
    """
    Hàm tải và khởi tạo mô hình tóm tắt từ ChatGroq.
    
    Tham số:
    - summarization_model_name: Tên mô hình tóm tắt. Mặc định là 'llama-3.1-70b-versatile'.
    - max_tokens: Số token tối đa cho mô hình tóm tắt. Mặc định là 458.
    
    Trả về: Đối tượng summarization model.
    """
    # Load API keys từ .env file
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=summarization_model_name, max_tokens=max_tokens)
def load_gpt4o_mini_model( model_name: str = "gpt-4o-mini", max_tokens: int = 512):
    """
    Hàm tải và khởi tạo mô hình chat từ ChatOpenAI.
    
    Trả về: Đối tượng chat model.
    """
    # Load API keys từ .env file
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name, max_tokens=max_tokens)
# Hàm tải mô hình chat (nếu cần sử dụng)
def load_chat_model(chat_model_name: str = 'llama-3.1-70b-versatile'):
    """
    Hàm tải và khởi tạo mô hình chat từ ChatGroq.
    
    Tham số:
    - chat_model_name: Tên mô hình chat. Mặc định là 'llama-3.1-70b-versatile'.
    
    Trả về: Đối tượng chat model.
    """
    # Load API keys từ .env file
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=chat_model_name)

# Hàm tải và khởi tạo tokenizer từ tiktoken (nếu cần)
def load_tiktoken(tokenizer_name: str = "cl100k_base"):
    """
    Hàm tải và khởi tạo tokenizer từ tiktoken.
    
    Tham số:
    - tokenizer_name: Tên tokenizer của tiktoken. Mặc định là "cl100k_base".
    
    Trả về: Đối tượng tokenizer.
    """
    return tiktoken.get_encoding(tokenizer_name)
