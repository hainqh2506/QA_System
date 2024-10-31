from model_config import load_tokenizer, load_embedding_model, load_summarization_model, load_gpt4o_mini_model,load_embedding_model_VN
from clutering import get_clusters
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import tiktoken
from functools import lru_cache
import time
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RaptorPipeline:
    def __init__(self, embedding_model=None, summarization_model=None):
        self.embedding_model = embedding_model or load_embedding_model_VN()
        self.summarization_model = summarization_model or load_gpt4o_mini_model()
        self.tokenizer = load_tokenizer()

    def embed_text(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        #embeddings_np = np.array(embeddings)  # Convert to NumPy array #
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
        metadata: List[Dict],  # Thêm biến metadata
        level: int,
        tokenizer: tiktoken.Encoding,
        max_tokens_in_cluster: int,
        embedding_function: Callable = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        if embedding_function is None:
            embedding_function = self.embed_text

        df_clusters = get_clusters(texts, tokenizer=tokenizer, max_length_in_cluster=max_tokens_in_cluster,embedding_function=embedding_function)
        
        if level == 0:
            # Sử dụng metadata từ chunks_metadata
            df_chunks = pd.DataFrame({
                "text": texts,
                "embedding": df_clusters["embd"].tolist(),
                "metadata": metadata,  # Thêm metadata vào DataFrame
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
        
        template = """Đây là một tài liệu.
        Hãy đưa ra bản tóm tắt chi tiết của tài liệu được cung cấp. Tài liệu:
        {context}"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.summarization_model | StrOutputParser()
        
        summaries = [
            chain.invoke({"context": self.fmt_txt(expanded_df[expanded_df["cluster"] == i])})
            for i in all_clusters
        ]
        
        df_summary = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        })
        
        return df_clusters, df_summary, df_chunks

    def recursive_embed_cluster_summarize(
        self,
        texts: List[str],
        metadata: List[Dict],  # Nhận thêm chunks_metadata
        level: int = 1,
        n_levels: int = 3,
        max_tokens_in_cluster: int = 5000,
        tokenizer: tiktoken.Encoding = None,
        embedding_function: Callable = None
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]]:
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        if embedding_function is None:
            embedding_function = self.embed_text

        results = {}
        df_chunks = None

        if level == 1:
            # Truyền embedding_function khi level = 0
            df_clusters, df_summary, df_chunks = self.embed_cluster_summarize(
                texts, metadata, level=0, tokenizer=tokenizer,
                max_tokens_in_cluster=max_tokens_in_cluster, embedding_function=embedding_function)
            results[0] = (df_clusters, df_summary, df_chunks)

        df_clusters, df_summary, _ = self.embed_cluster_summarize(
            texts, metadata, level, tokenizer, max_tokens_in_cluster, embedding_function)
        results[level] = (df_clusters, df_summary, df_chunks)

        unique_clusters = df_summary["cluster"].nunique()

        if level < n_levels and unique_clusters > 1:
            new_texts = df_summary["summaries"].tolist()

            if len(new_texts) == len(texts):
                print(f"No change in number of texts at level {level}, stopping recursion.")
                return results
            
            # Đệ quy truyền metadata và embedding_function
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, metadata, level + 1, n_levels, max_tokens_in_cluster, tokenizer, embedding_function
            )
            results.update(next_level_results)
        else:
            results[level] = (df_clusters, df_summary)
        
        return results


    def aggregate_metadata(self, metadata_list: List[Dict]) -> Dict:
        aggregated_metadata = {
            "id": [md["id"] for md in metadata_list],
            "page": list(set(md["page"] for md in metadata_list)),
            "source": list(set(md["source"] for md in metadata_list)),
        }
        return aggregated_metadata

    def build_final_dataframe(self, results: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
        final_rows = []

        if 0 in results:
            df_chunks = results[0][2]
        else:
            df_chunks = pd.DataFrame()

        for _, row in df_chunks.iterrows():
            final_rows.append({
                "text": row["text"],
                "embedding": row["embedding"],
                "metadata": row["metadata"],
                "level": row["level"]
            })

        for level in range(1, max(results.keys()) + 1):
            df_clusters, df_summary = results[level][:2]
            for cluster_id in df_clusters["cluster"].unique():
                cluster_texts = df_clusters[df_clusters["cluster"] == cluster_id]["text"].tolist()
                cluster_metadata = [
                    df_chunks[df_chunks["text"] == text]["metadata"].values[0] 
                    for text in cluster_texts if text in df_chunks["text"].values
                ]
                aggregated_metadata = self.aggregate_metadata(cluster_metadata)
                summary_text = df_summary[df_summary["cluster"] == cluster_id]["summaries"].values[0]
                summary_embedding = df_clusters[df_clusters["cluster"] == cluster_id]["embd"].values[0]

                final_rows.append({
                    "text": summary_text,
                    "embedding": summary_embedding,
                    "metadata": aggregated_metadata,
                    "level": level
                })
        
        final_df = pd.DataFrame(final_rows)
        return final_df
