from elasticsearch import Elasticsearch, helpers
from uuid import uuid4
import pandas as pd
from langchain_core.documents import Document
from model_config import VietnameseEmbeddings
from ultil import load_final_df, convert_df_to_documents
import json ,os

class ElasticsearchHelper:
    def __init__(self, host, api_key):
        self.es = Elasticsearch(host, api_key=api_key)

    def create_index(self, index_name, mapping):
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        self.es.indices.create(index=index_name, body=mapping)

    def bulk_insert(self, index_name, actions_generator):
        helpers.bulk(self.es, actions_generator)
        print(f"Data inserted into index '{index_name}' successfully.")

class DataProcessor:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def load_excel_data(self, excel_path):
        final_df = load_final_df(final_df_path=excel_path)
        return convert_df_to_documents(final_df)

    def load_json_data(self, json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def generate_bulk_actions_json(self, data, index_name, vector_field="vector"):
        for record in data:
            question = record.get("question")
            embedding = self.embedding_model.embed_query(question)

            yield {
                "_op_type": "index",
                "_index": index_name,
                "_source": {
                    "question": question,
                    "answer": record.get("answer"),
                    "category": record.get("category"),
                    vector_field: embedding,
                }
            }

    def generate_bulk_actions_documents(self, documents, index_name, vector_field="vector"):
        for doc in documents:
            embedding = self.embedding_model.embed_query(doc.page_content)

            yield {
                "_op_type": "index",
                "_index": index_name,
                "_source": {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    vector_field: embedding,
                }
            }

class DocumentConverter:
    @staticmethod
    def convert_to_json(documents):
        json_documents = []
        for doc in documents:
            metadata = doc.metadata
            if isinstance(metadata.get("source"), str):
                try:
                    metadata["source"] = json.loads(metadata["source"].replace("'", '"'))
                except json.JSONDecodeError:
                    pass

            json_documents.append({
                "metadata": metadata,
                "page_content": doc.page_content
            })
        return json_documents

# Configuration and Initialization
ELASTIC_HOST = os.getenv("ELASTIC_URL")
API_KEY = os.getenv("api_key")
EXCEL_PATH = r"D:\\DATN\\QA_System\\data_analyze\\finaldf0.xlsx"
JSON_PATH = r"D:\DATN\chatbot\code\output.json"

INDEX_FAQ = "faq_data"
MAPPING_FAQ = {
    "mappings": {
        "properties": {
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "category": {"type": "keyword"},
            "vector": {"type": "dense_vector", "dims": 768, "similarity": "cosine", "index": True}
        }
    }
}

INDEX_BASE = "base"
MAPPING_BASE = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "metadata": {"type": "object"},
            "vector": {"type": "dense_vector", "dims": 768, "similarity": "cosine", "index": True}
        }
    }
}

# Main Execution
es_helper = ElasticsearchHelper(ELASTIC_HOST, API_KEY)
data_processor = DataProcessor(VietnameseEmbeddings())

# Load and insert FAQ data
faq_data = data_processor.load_json_data(JSON_PATH)
es_helper.create_index(INDEX_FAQ, MAPPING_FAQ)
es_helper.bulk_insert(INDEX_FAQ, data_processor.generate_bulk_actions_json(faq_data, INDEX_FAQ))

# # Load and insert base data
# documents = data_processor.load_excel_data(EXCEL_PATH)
# json_documents = DocumentConverter.convert_to_json(documents)
# es_helper.create_index(INDEX_BASE, MAPPING_BASE)
# es_helper.bulk_insert(INDEX_BASE, data_processor.generate_bulk_actions_documents(documents, INDEX_BASE))
