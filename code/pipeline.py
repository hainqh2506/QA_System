from milvus_db import MilvusDB, MilvusQADB# step4_vector_store
from utils import convert_df_to_documents
from model_config import VietnameseEmbeddings
import os
import pandas as pd
def load_final_df():
    final_df_path = r"D:\DATN\QA_System\data_analyze\finaldf.pkl"
    if os.path.exists(final_df_path):
        final_df = pd.read_pickle(final_df_path)
        print(f"Đã load final_df từ file: {final_df_path}")
        return final_df
    else:
        print("File final_df chưa tồn tại.")
        return None
def step4_vector_store():
    """
    Store document vectors
    """
    final_df = load_final_df()
    documents = convert_df_to_documents(final_df)
    corpus=[doc.page_content for doc in documents]
    # Tạo Milvus database
    milvus_db = MilvusDB(collection_name="hybrid_demo", corpus=corpus)
    milvus_db.create_collection()
    milvus_db.insert_documents(documents)
# Tối ưu 1: Sử dụng singleton pattern để cache DataFrame và corpus
class DataManager:
    _instance = None
    _final_df = None
    _corpus = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_final_df(self):
        if self._final_df is None:
            print("Loading final_df from file...")
            self._final_df = load_final_df()
        return self._final_df
    
    def get_corpus(self):
        if self._corpus is None:
            documents = convert_df_to_documents(self.get_final_df())
            self._corpus = [doc.page_content for doc in documents]
        return self._corpus

# Tối ưu 2: Singleton pattern cho MilvusDB để tránh khởi tạo lại
class MilvusManager:
    _instance = None
    _milvus_db = None
    _milvus_qa_db = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_milvus_db(self, collection_name):
        if self._milvus_db is None:
            data_manager = DataManager.get_instance()
            corpus = data_manager.get_corpus()
            self._milvus_db = MilvusDB(collection_name=collection_name, corpus=corpus)
            self._milvus_db.load_collection()
        return self._milvus_db
    def get_milvus_qa_db(self, qa_collection_name):
            # Khởi tạo QA collection
        if self._milvus_qa_db is None:
            self._milvus_qa_db = MilvusQADB(collection_name=qa_collection_name)
            self._milvus_qa_db.load_qa_collection()
        return self._milvus_qa_db

# Tối ưu 3: Cập nhật hàm step5_retrieval
def step5_retrieval(collection_name: str):
    data_manager = DataManager.get_instance()
    milvus_manager = MilvusManager.get_instance()
    corpus = data_manager.get_corpus()
    milvus_db = milvus_manager.get_milvus_db(collection_name)
    return corpus, milvus_db 
def step6_qa_db(collection_name: str):
    milvus_manager = MilvusManager.get_instance()
    milvus_qa_db = milvus_manager.get_milvus_qa_db(collection_name)
    return milvus_qa_db
# if __name__ == "__main__":
#     copus, milvus_db = step5_retrieval("hybrid_demo")
#     relevant_docs = milvus_db.perform_retrieval(query="giám đốc đại học là ai", top_k=5)
#     print(relevant_docs)
