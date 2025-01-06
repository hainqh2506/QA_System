from typing import List, Dict, Any
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
    WeightedRanker,
)
from pymilvus import MilvusClient, DataType, Function, FunctionType
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from model_config import VietnameseEmbeddings, load_tokenizer 
from pyvi import ViTokenizer
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from milvus_model.sparse import BM25EmbeddingFunction 
from milvus_model.sparse.bm25 import Analyzer , build_default_analyzer, build_analyzer_from_yaml
import json
import csv
import pandas as pd
#import pickle , os
from pathlib import Path
DENSE_DIM = 768
# Khởi tạo tokenizer
tokenizer = load_tokenizer()
#bm25tokenizer = ViTokenizer
class VietnameseAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, text)->List[str]:
        return self.tokenizer.tokenize(text)

class BM25SparseEmbeddingVi(BaseSparseEmbedding):
    
    
    def __init__(self, corpus: List[str], tokenizer):
        self.tokenizer = tokenizer
        vi_analyzer = VietnameseAnalyzer(tokenizer)
        self.analyzer = build_default_analyzer(language="en")
        #self.analyzer = vi_analyzer
        
        self.bm25_ef = BM25EmbeddingFunction(self.analyzer, num_workers=1)
        self.bm25_ef.fit(corpus)
        #self.bm25_ef.save("bm25.json")
# class BM25SparseEmbeddingVi(BaseSparseEmbedding):
#     def __init__(self, corpus: List[str], tokenizer, model_path: str = None):
#         self.tokenizer = tokenizer
#         self.model_path = model_path or "bm25_model.pkl"
#         vi_analyzer = VietnameseAnalyzer(tokenizer)
#         self.analyzer = vi_analyzer
#         self.bm25_ef = BM25EmbeddingFunction(self.analyzer, num_workers=1)
        
#         if model_path and os.path.exists(model_path):
#             self.load(model_path)
#         elif corpus:
#             self.bm25_ef.fit(corpus)
#             if model_path:
#                 self.save(model_path)

#     def save_model(self, path: str):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         try:
#             with open(path, 'wb') as f:
#                 pickle.dump(self.bm25_ef, f)
#             print(f"Saved BM25 model to {path}")
#         except Exception as e:
#             print(f"Error saving model: {e}")

#     def load_model(self, path: str):
#         try:
#             with open(path, 'rb') as f:
#                 self.bm25_ef = pickle.load(f)
#             print(f"Loaded BM25 model from {path}")
#         except Exception as e:
#             print(f"Error loading model: {e}")

#     # ...existing methods remain unchanged...

    def embed_query(self, text: str) -> Dict[int, float]:
        return self._sparse_to_dict(self.bm25_ef.encode_queries([text]))

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        sparse_arrays = self.bm25_ef.encode_documents(texts)
        return [self._sparse_to_dict(sparse_array) for sparse_array in sparse_arrays]

    def _sparse_to_dict(self, sparse_array):
        row_indices, col_indices = sparse_array.nonzero()
        non_zero_values = sparse_array.data
        return {col_index: value for col_index, value in zip(col_indices, non_zero_values)}

class MilvusDB:
    def __init__(self, collection_name: str, uri: str = "http://localhost:19530", corpus: List[str] = None):
        self.collection_name = collection_name
        self.uri = uri
        self.tokenizer = tokenizer
        self.sparse_embedding_func = None
        self.dense_embedding_func = VietnameseEmbeddings()
        self.collection = None
        
        # Establish connection to Milvus
        connections.connect(uri=self.uri)
        
        # Initialize sparse embedding function if a corpus is provided
        if corpus :
            self.sparse_embedding_func = BM25SparseEmbeddingVi(
                corpus=corpus, 
                tokenizer=self.tokenizer
            )
    # def get_dense_embedding_func(self):
    #     return self.dense_embedding_func
    def create_collection(self):
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8191),#enable_analyzer=True
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page", dtype=DataType.INT16),
            FieldSchema(name="level", dtype=DataType.INT8),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        
        if utility.has_collection(self.collection_name):
            Collection(name=self.collection_name).drop()
    #     bm25_function = Function(
    #     name="text_bm25_emb", # Function name
    #     input_field_names=["text"], # Name of the VARCHAR field containing raw text data
    #     output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
    #     function_type=FunctionType.BM25,
    # )

    #     schema.add_function(bm25_function)


        self.collection = Collection(name=self.collection_name, schema=schema, consistency_level="Strong")
        self.collection.create_index("dense_vector", {"index_type": "FLAT", "metric_type": "COSINE"})
        self.collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
        #self.collection.create_index("sparse_vector", {"index_type": "AUTO_INDEX", "metric_type": "BM25"})
        self.collection.load()

    def insert_documents(self, documents):
        if not self.collection:
            raise ValueError("Collection is not initialized. Call `create_collection` first.")
        
        entities = []
        for doc in documents:
            dense_vector = self.dense_embedding_func.embed_documents([doc.page_content])[0]
            sparse_vector = self.sparse_embedding_func.embed_documents([doc.page_content])[0]
            entities.append({
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", -1),
                "level": doc.metadata.get("level", -1),
            })
        
        self.collection.insert(entities)
        self.collection.flush()

    def load_collection(self):
        """Load an existing collection for querying."""
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist.")
        
        self.collection = Collection(name=self.collection_name)
        self.collection.load()

    def perform_retrieval(self, query: str, top_k: int = 5):
        if not self.collection:
            raise ValueError("Collection is not loaded. Call `load_collection` first.")
        
        sparse_search_params = {"metric_type": "IP", "params": {}}
        dense_search_params = {"metric_type": "COSINE", "params": {}}
        field_limits = [50, 50] 
        retriever = MilvusCollectionHybridSearchRetriever(
            collection=self.collection,
            rerank=WeightedRanker(0.7, 0.3),
            anns_fields=["dense_vector", "sparse_vector"],
            field_embeddings=[self.dense_embedding_func, self.sparse_embedding_func],
            field_search_params=[dense_search_params, sparse_search_params],
            field_limits = field_limits,
            top_k=top_k,
            #output_fields = ["text", "source"], #["text", "source", "page", "level"],
            text_field="text",
        )
        return retriever.invoke(query)


from typing import List, Dict, Any
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from model_config import VietnameseEmbeddings

class MilvusQADB(MilvusDB):
    def __init__(self, collection_name: str, uri: str = "http://localhost:19530"):
        super().__init__(collection_name, uri)
        self.dense_embedding_func = VietnameseEmbeddings()

    def create_qa_collection(self):
        """Tạo collection riêng cho Q&A với các trường phù hợp"""
        fields = [
            FieldSchema(name="qa_id", dtype=DataType.VARCHAR, is_primary=True,auto_id=True, max_length=100),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=8191),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=8191),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=255)
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        
        # Xóa collection cũ nếu tồn tại
        if utility.has_collection(self.collection_name):
            Collection(name=self.collection_name).drop()
        
        # Tạo collection mới
        self.collection = Collection(name=self.collection_name, schema=schema, consistency_level="Strong")
        
        # Tạo index cho vector
        self.collection.create_index("dense_vector", {"index_type": "FLAT", "metric_type": "COSINE"})
        self.collection.load()

    def load_qa_collection(self):
        """Nạp lại collection Q&A đã tồn tại"""
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} không tồn tại.")
        
        self.collection = Collection(name=self.collection_name)
        self.collection.load()

    def insert_qa_pairs(self, qa_pairs: List[Dict[str, Any]]):
        """
        Chèn danh sách các cặp câu hỏi-trả lời
        
        Mẫu qa_pairs:
        [
            {
                "question": "Thủ đô của Việt Nam là gì?", 
                "answer": "Hà Nội", 
                "category": "Địa lý"
            },
            ...
        ]
        """
        if not self.collection:
            raise ValueError("Collection chưa được khởi tạo. Hãy gọi `create_qa_collection` trước.")
        
        entities = []
        for qa in qa_pairs:
            # Tạo dense embedding cho câu hỏi
            dense_vector = self.dense_embedding_func.embed_documents([qa['question']])[0]
            
            entities.append({
                "question": qa['question'],
                "answer": qa['answer'],
                "dense_vector": dense_vector,
                "category": qa.get('category', 'Chung')
            })
        
        self.collection.insert(entities)
        self.collection.flush()

    def semantic_qa_search(self, query: str, similarity_threshold: float = 0.85, top_k: int = 5):
        """
        Tìm kiếm ngữ nghĩa với ngưỡng cosine similarity
        
        Trả về:
        - Nếu tìm thấy câu hỏi có similarity > threshold: trả về câu trả lời
        - Nếu không: trả về None hoặc danh sách kết quả gần nhất
        """
        if not self.collection:
            raise ValueError("Collection chưa được nạp. Hãy gọi `load_qa_collection` trước.")
        
        # Tạo embedding cho query
        query_embedding = self.dense_embedding_func.embed_documents([query])[0]
        
        # Chuẩn bị tham số tìm kiếm
        search_params = {
            "metric_type": "COSINE", 
            "params": {}
        }
        
        # Thực hiện tìm kiếm vector
        results = self.collection.search(
            data=[query_embedding], 
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["question", "answer", "category"]
        )
        
        # Kiểm tra similarity
        for result in results[0]:
            if result.distance >= similarity_threshold:
                return {
                    "answer": result.entity.get('answer'),
                    "question": result.entity.get('question'),
                    "category": result.entity.get('category'),
                    "similarity": 1 - result.distance  # Chuyển đổi khoảng cách cosine sang similarity
                }
        
        return None  # Không tìm thấy câu trả lời phù hợp



    @classmethod
    def load_qa_pairs_from_json(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        Đọc các cặp câu hỏi-trả lời từ file JSON
        
        Cấu trúc file JSON mẫu:
        [
            {
                "question": "Thủ đô của Việt Nam là gì?",
                "answer": "Hà Nội",
                "category": "Địa lý"
            },
            ...
        ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                qa_pairs = json.load(file)
            return qa_pairs
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Lỗi đọc file JSON: {e}")
            return []

    @classmethod
    def load_qa_pairs_from_csv(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        Đọc các cặp câu hỏi-trả lời từ file CSV
        
        Cấu trúc file CSV mẫu:
        question,answer,category
        Thủ đô của Việt Nam là gì?,Hà Nội,Địa lý
        """
        try:
            qa_pairs = []
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    # Loại bỏ các khoảng trắng thừa
                    cleaned_row = {k: v.strip() for k, v in row.items()}
                    # Thêm category mặc định nếu không có
                    if 'category' not in cleaned_row:
                        cleaned_row['category'] = 'Chung'
                    qa_pairs.append(cleaned_row)
            return qa_pairs
        except (FileNotFoundError, csv.Error) as e:
            print(f"Lỗi đọc file CSV: {e}")
            return []

    @classmethod
    def load_qa_pairs_from_excel(cls, file_path: str, sheet_name: str = 0) -> List[Dict[str, Any]]:
        """
        Đọc các cặp câu hỏi-trả lời từ file Excel
        
        Cấu trúc sheet Excel mẫu:
        | question | answer | category |
        | Thủ đô của Việt Nam là gì? | Hà Nội | Địa lý |
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Kiểm tra các cột bắt buộc
            required_columns = ['question', 'answer']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Thiếu cột bắt buộc: {col}")
            
            # Thêm cột category nếu chưa có
            if 'category' not in df.columns:
                df['category'] = 'Chung'
            
            # Chuyển đổi DataFrame thành list dictionary
            qa_pairs = df[['question', 'answer', 'category']].to_dict('records')
            return qa_pairs
        except Exception as e:
            print(f"Lỗi đọc file Excel: {e}")
            return []

    def import_qa_pairs(self, qa_pairs: List[Dict[str, Any]]):
        """
        Nhập khẩu và chèn các cặp Q&A vào collection
        Hỗ trợ import từ nhiều nguồn: list, JSON, CSV, Excel
        """
        # Nếu đầu vào là đường dẫn file
        if isinstance(qa_pairs, str):
            # Xác định loại file
            if qa_pairs.endswith('.json'):
                qa_pairs = self.load_qa_pairs_from_json(qa_pairs)
            elif qa_pairs.endswith('.csv'):
                qa_pairs = self.load_qa_pairs_from_csv(qa_pairs)
            elif qa_pairs.endswith(('.xls', '.xlsx')):
                qa_pairs = self.load_qa_pairs_from_excel(qa_pairs)
            else:
                raise ValueError("Định dạng file không được hỗ trợ")
        
        # Kiểm tra dữ liệu
        if not qa_pairs:
            print("Không có dữ liệu Q&A để nhập khẩu")
            return
        
        # Chèn dữ liệu
        self.insert_qa_pairs(qa_pairs)
        print(f"Đã nhập khẩu {len(qa_pairs)} cặp câu hỏi-trả lời")
# # Import các thư viện cần thiết
# import json
# import csv
# import pandas as pd
# from pymilvus import connections

# # Khởi tạo kết nối Milvus (nếu chưa kết nối)
# connections.connect(uri="http://localhost:19530")

# # 1. Tạo mới collection Q&A
# qa_db = MilvusQADB("vietnamese_qa_collection")
# qa_db.create_qa_collection()

# # 2. Chèn Q&A từ list trực tiếp
# manual_qa_pairs = [
#     {
#         "question": "Thủ đô của Việt Nam là gì?", 
#         "answer": "Hà Nội", 
#         "category": "Địa lý",
#     },
#     {
#         "question": "Sông dài nhất Việt Nam là gì?", 
#         "answer": "Sông Mekong", 
#         "category": "Địa lý",
#     }
# ]
# qa_db.insert_qa_pairs(manual_qa_pairs)

# # 3. Import Q&A từ file JSON
# qa_db.import_qa_pairs("path/to/qa_pairs.json")

# # 4. Import Q&A từ file CSV
# qa_db.import_qa_pairs("path/to/qa_pairs.csv")

# # 5. Import Q&A từ file Excel
# qa_db.import_qa_pairs("path/to/qa_pairs.xlsx")

# # 6. Nạp lại collection để sử dụng sau này
# qa_db.load_qa_collection()

# # 7. Thực hiện tìm kiếm ngữ nghĩa
# query = "Thành phố đứng đầu nước Việt Nam là gì?"
# result = qa_db.semantic_qa_search(query, similarity_threshold=0.8)

# if result:
#     print("Câu trả lời được tìm thấy:")
#     print(f"Câu hỏi gốc: {result['question']}")
#     print(f"Câu trả lời: {result['answer']}")
#     print(f"Độ tương đồng: {result['similarity']}")
# else:
#     print("Không tìm thấy câu trả lời phù hợp")

# # 8. Nếu muốn đọc trước dữ liệu từ file
# json_qa_pairs = MilvusQADB.load_qa_pairs_from_json("path/to/qa_pairs.json")
# csv_qa_pairs = MilvusQADB.load_qa_pairs_from_csv("path/to/qa_pairs.csv")
# excel_qa_pairs = MilvusQADB.load_qa_pairs_from_excel("path/to/qa_pairs.xlsx")