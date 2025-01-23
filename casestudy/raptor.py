from model_config import load_tokenizer2, load_summarization_model, load_gpt4o_mini_model, load_embedding_model_VN2, load_gemini
from clutering import get_clusters
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import tiktoken
from functools import lru_cache
import time
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from tqdm import tqdm

class RaptorPipeline:
    def __init__(self, embedding_model=None, summarization_model=None):
        print("Initializing RaptorPipeline...")
        self.embedding_model = embedding_model or load_embedding_model_VN2()
        print("Embedding model loaded.")
        self.summarization_model = summarization_model or load_gemini()
        print("Summarization model loaded.")
        self.tokenizer = load_tokenizer2()
        print("Tokenizer loaded.")

    def embed_text(self, texts: List[str]) -> np.ndarray:
        print("Embedding texts...")
        embeddings = self.embedding_model.embed_text(texts)
        print(f"Texts embedded. Number of embeddings: {len(embeddings)}")
        return embeddings

    @lru_cache(maxsize=None)
    def cached_tokenizer_encode(self, text: str, tokenizer: tiktoken.Encoding) -> int:
        """
        Hàm cache số lượng token đã mã hóa để tránh lặp lại tính toán tốn thời gian.
        """
        return len(tokenizer.encode(text))

    def fmt_txt(self, df: pd.DataFrame) -> str:
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def embed_cluster_summarize(
        self,
        texts: List[str],
        metadata: List[Dict],
        level: int,
        tokenizer: tiktoken.Encoding,
        max_tokens_in_cluster: int,
        embedding_function: Callable = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        print(f"Starting embed_cluster_summarize for level {level}...")

        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        if embedding_function is None:
            embedding_function = self.embed_text

        print("Getting clusters...")
        df_clusters = get_clusters(texts, tokenizer=tokenizer, max_length_in_cluster=max_tokens_in_cluster, embedding_function=embedding_function)
        print(f"Clusters obtained. Number of clusters: {df_clusters['cluster'].nunique()}")
        
        if level == 0:
            df_chunks = pd.DataFrame({
                "text": texts,
                "embedding": df_clusters["embd"].tolist(),
                "metadata": metadata,
                "level": 0
            })
        else:
            df_chunks = None

        expanded_list = [
            {"text": row["text"], "embd": row["embd"], "cluster": row["cluster"]}
            for _, row in df_clusters.iterrows()
        ]
        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()

        template = """
        Dưới đây là các đoạn văn bản thuộc tài liệu về các thông tin, quy định và hướng dẫn dành cho sinh viên Đại học Bách Khoa Hà Nội. 
        Hãy tóm tắt ngắn gọn và chi tiết nội dung chính của các văn bản dưới đây. 
        ###Quan trọng: luôn trả lời bằng tiếng Việt (Chỉ trả lời với nội dung đã tóm tắt)

        Các đoạn văn bản:
        {context}
        Tóm tắt (Chỉ trả lời với nội dung đã tóm tắt):
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.summarization_model | StrOutputParser()

        summaries = []
        print("Generating summaries...")
        for i in tqdm(all_clusters, desc=f"Summarizing clusters (level {level})"):
            summary = chain.invoke({"context": self.fmt_txt(expanded_df[expanded_df["cluster"] == i])})
            summaries.append(summary)
            time.sleep(4.5)
        print("Summaries generated.")

        df_summary = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        })

        print(f"Finished embed_cluster_summarize for level {level}.")
        return df_clusters, df_summary, df_chunks

    def recursive_embed_cluster_summarize(
        self,
        texts: List[str],
        metadata: List[Dict],
        level: int = 1,
        n_levels: int = 3,
        max_tokens_in_cluster: int = 5000,
        tokenizer: tiktoken.Encoding = None,
        embedding_function: Callable = None
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]]:
        print(f"Starting recursive_embed_cluster_summarize for level {level}...")

        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        if embedding_function is None:
            embedding_function = self.embed_text

        results = {}
        df_chunks = None

        if level == 1:
            print("Processing base level (level 0)...")
            df_clusters, df_summary, df_chunks = self.embed_cluster_summarize(
                texts, metadata, level=0, tokenizer=tokenizer,
                max_tokens_in_cluster=max_tokens_in_cluster, embedding_function=embedding_function)
            results[0] = (df_clusters, df_summary, df_chunks)
            print("Base level processing completed.")

        print(f"Processing level {level}...")
        df_clusters, df_summary, _ = self.embed_cluster_summarize(
            texts, metadata, level, tokenizer, max_tokens_in_cluster, embedding_function)
        results[level] = (df_clusters, df_summary, df_chunks)
        print(f"Level {level} processing completed.")

        unique_clusters = df_summary["cluster"].nunique()

        if level < n_levels and unique_clusters > 1:
            new_texts = df_summary["summaries"].tolist()
            if len(new_texts) == len(texts):
                print(f"No change in number of texts at level {level}, stopping recursion.")
                return results

            print(f"Recursing to level {level + 1}...")
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, metadata, level + 1, n_levels, max_tokens_in_cluster, tokenizer, embedding_function
            )
            results.update(next_level_results)
            print(f"Recursion to level {level + 1} completed.")
        else:
            results[level] = (df_clusters, df_summary)
        
        print(f"Finished recursive_embed_cluster_summarize for level {level}.")
        return results


    def aggregate_metadata(self, metadata_list: List[Dict]) -> Dict:
        print("Aggregating metadata...")
        aggregated_metadata = {
            "id": [],
            "source": [],
        }
        for md in metadata_list:
            if isinstance(md, str):
                md = json.loads(md)  # Chuyển đổi từ chuỗi JSON trở lại thành dict
            if 'id' in md:
                aggregated_metadata["id"].extend(md["id"] if isinstance(md["id"], list) else [md["id"]])
            if 'source' in md:
                aggregated_metadata["source"].extend(md["source"] if isinstance(md["source"], list) else [md["source"]])

        # Loại bỏ các giá trị trùng lặp và giữ nguyên thứ tự
        aggregated_metadata["id"] = list(dict.fromkeys(aggregated_metadata["id"]))
        aggregated_metadata["source"] = list(dict.fromkeys(aggregated_metadata["source"]))

        #print(f"Metadata aggregated. IDs: {aggregated_metadata['id']}, Sources: {aggregated_metadata['source']}")
        return aggregated_metadata

    def build_final_dataframe(self, results: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
        print("Building final dataframe...")
        final_rows = []

        if 0 in results:
            df_chunks = results[0][2]
        else:
            df_chunks = pd.DataFrame()

        print("Processing chunks (level 0)...")
        for _, row in tqdm(df_chunks.iterrows(), total=len(df_chunks), desc="Processing chunks"):
            final_rows.append({
                "text": row["text"],
                "metadata": row["metadata"],  # Lưu trực tiếp dictionary
                "level": row["level"]
            })

        print("Processing summaries (levels > 0)...")
        for level in tqdm(range(1, max(results.keys()) + 1), desc="Processing summary levels"):
            df_clusters, df_summary = results[level][:2]

            for cluster_id in df_clusters["cluster"].unique():
                cluster_texts = df_clusters[df_clusters["cluster"] == cluster_id]["text"].tolist()

                # Tìm metadata
                cluster_metadata = []
                if level == 1:
                    # Level 1: Lấy metadata từ df_chunks (level 0)
                    for text in cluster_texts:
                        metadata_matches = df_chunks[df_chunks["text"] == text]["metadata"].tolist()
                        cluster_metadata.extend(metadata_matches)
                else:
                    # Level > 1: Lấy metadata từ final_rows của các level trước
                    for text in cluster_texts:
                        for row in final_rows:
                            if row["text"] == text and row["level"] < level:
                                cluster_metadata.append(row["metadata"])  # Lấy trực tiếp dictionary

                aggregated_metadata = self.aggregate_metadata(cluster_metadata)
                summary_text = df_summary[df_summary["cluster"] == cluster_id]["summaries"].values[0]

                final_rows.append({
                    "text": summary_text,
                    "metadata": aggregated_metadata,  # Lưu trực tiếp dictionary
                    "level": level
                })

        final_df = pd.DataFrame(final_rows)

        # Bỏ cột embedding
        final_df = final_df.drop(columns=["embedding"], errors='ignore')

        print("Final dataframe built.")
        return final_df