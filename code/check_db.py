from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
    MilvusException,
)
from typing import List, Dict

# Kết nối Milvus
def connect_to_milvus(uri="http://localhost:19530"):
    try:
        connections.connect(uri=uri)
        print("[INFO] Connected to Milvus.")
        print("list_collections: ", utility.list_collections())
    except MilvusException as e:
        print(f"[ERROR] Error connecting to Milvus: {e}")
        raise e

# Hàm kiểm tra trạng thái collection
def check_collection_status(col_name: str):
    try:
        if utility.has_collection(col_name):
            collection = Collection(name=col_name)
            print(f"[INFO] Collection '{col_name}' exists. Total entities: {collection.num_entities}")
        else:
            print(f"[INFO] Collection '{col_name}' does not exist.")
    except MilvusException as e:
        print(f"[ERROR] Error checking collection '{col_name}': {e}")
        raise e
# Main logic
if __name__ == "__main__":
    connect_to_milvus()

    # Tên collection
    col_name = "hybrid_demo"

    # Kiểm tra trạng thái collection
    check_collection_status(col_name)
    check_collection_status(col_name="base_qa")