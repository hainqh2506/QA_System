import os
import pandas as pd
from data_ingestion import setup_knowledge_base ,TXTProcessor # step1_loaddata
from chunking import text_splitter  # step2_chunking
from raptor import RaptorPipeline  # step3_RAPTOR
from typing import List
from tqdm import tqdm

# Các bước pipeline
def step1_load_data(txt_file: str) -> List:
    """
    Load dữ liệu từ một file .txt cụ thể.
    """
    processor = TXTProcessor()
    documents = processor.setup_txt(txt_file)
    print(f"Đã load dữ liệu từ file: {txt_file}")
    return documents

def step2_chunking(documents):
    """
    Chunk tài liệu thành các phần nhỏ hơn.
    """
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = str(i)
    chunks_metadata = [chunk.metadata for chunk in chunks]
    chunks_content = [chunk.page_content for chunk in chunks]
    return chunks_metadata, chunks_content

def step3_RAPTOR(chunks_metadata, chunks_content):
    """
    Thực hiện RAPTOR pipeline trên chunks.
    """
    raptor = RaptorPipeline()
    results = raptor.recursive_embed_cluster_summarize(chunks_content, chunks_metadata, level=1, n_levels=3)
    final_df = raptor.build_final_dataframe(results)
    return final_df

def process_directory(directory: str):
    """
    Đọc và xử lý toàn bộ các file trong thư mục với RAPTOR.
    """
    processor = TXTProcessor(directory=directory)
    txt_files = processor.get_txt_files()
    
    for txt_file in tqdm(txt_files, desc="Processing files"):
        # Bước 1: Load dữ liệu
        documents = step1_load_data(txt_file)
        
        # Bước 2: Chunking
        chunks_metadata, chunks_content = step2_chunking(documents)
        
        # Bước 3: RAPTOR
        final_df = step3_RAPTOR(chunks_metadata, chunks_content)
        
        # Lưu kết quả thành file CSV
        output_csv = f"{os.path.splitext(txt_file)[0]}_results.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"Đã lưu kết quả RAPTOR cho file {txt_file} vào {output_csv}")
        # # Lưu kết quả thành file Pickle
        # output_pkl = f"{os.path.splitext(txt_file)[0]}_results.pkl"
        # final_df.to_pickle(output_pkl)  # Lưu DataFrame dưới dạng Pickle
        # print(f"Đã lưu kết quả RAPTOR cho file {txt_file} vào {output_pkl}")
# Thực thi pipeline
if __name__ == "__main__":
    directory_path = r"D:\DATN\QA_System\data sample"  # Thay bằng đường dẫn thực tế
    process_directory(directory_path)
