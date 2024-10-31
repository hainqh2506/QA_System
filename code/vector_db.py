# vector_db.py
import os
import ast
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from model_config import load_embedding_model_VN , load_gpt4o_mini_model
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
# Load environment variables
load_dotenv()
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

class MilvusStorePipeline:
    def __init__(self, collection_name: str, df: pd.DataFrame):
        """
        Quản lý lưu trữ vector trong Milvus.
        """
        self.collection_name = collection_name
        self.df = df
        self.client = MilvusClient(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

    def create_new_collection(self):
        """
        Tạo collection mới và tạo lại nếu đã tồn tại.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="level", dtype=DataType.INT8),
        ]
        schema = CollectionSchema(fields, description="Text and embedding storage")

        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            # Tạo collection
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            
            # Sử dụng prepare_index_params để thiết lập chỉ mục
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="AUTOINDEX",
                index_name="vector_index"
            )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            print(f"Collection '{self.collection_name}' created with index 'vector_index'.")
    def upload_all_data(self):
        """
        Chèn tất cả dữ liệu từ DataFrame vào Milvus trong một lần.
        """
        entities = [
            {
                "text": row['text'],
                "embedding": row['embedding'] if isinstance(row['embedding'], list) else row['embedding'].tolist(),
                "metadata": ast.literal_eval(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'],
                "level": row['level']
            }
            for _, row in self.df.iterrows()
        ]

        try:
            res = self.client.insert(collection_name=self.collection_name, data=entities)
            print(f"Đã chèn thành công {res.get('insert_count', len(entities))} dữ liệu vào Milvus.")
        except Exception as e:
            print(f"Lỗi khi chèn dữ liệu: {str(e)}")

    def create_collection_cloud(self):
        """
        Tạo collection và chèn vector.
        """
        try:
            self.create_new_collection()
            self.upload_all_data()
        except Exception as e:
            print(f"Error creating collection or uploading data: {e}")


class MilvusRetrieval:
    def __init__(self, collection_name: str, embedding_model=None):
        """
        Manages vector retrieval from Milvus.

        :param collection_name: Name of the Milvus collection.
        :param embedding_function: Function to convert text queries to embedding vectors.
        """
        self.collection_name = collection_name
        self.client = MilvusClient(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)
        self.embedding_model = embedding_model or load_embedding_model_VN()

    def similarity_search(self, query: str, top_k: int = 25):
        """
        Perform a similarity search in Milvus with a query.
        """
        query_vector = self.embedding_model.encode(query)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        # Perform search in Milvus
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="embedding",
            search_params=search_params,
            limit=top_k,
            output_fields=["metadata", "text"]
        )
        docs = list((res["entity"]["text"], res["entity"]["metadata"]["source"],res["distance"]) for res in results[0])


        # Chuyển docs thành danh sách các Document của LangChain
        relevant_docs  = [
            Document(page_content=text, metadata={"source": source, "distance": distance})
            for text, source, distance in docs
        ]
        # Combine the query and the relevant document contents
        combined_input = (
            "Here are some documents that might help answer the question: "
            + query
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])
            + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
            + "### important: answer in Vietnamese"
        )
        return combined_input
