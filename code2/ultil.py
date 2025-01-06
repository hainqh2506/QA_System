from uuid import uuid4
import pandas as pd
from langchain_core.documents import Document
import os
def load_final_df(final_df_path = r"D:\DATN\QA_System\data_analyze\finaldf0.xlsx"):
    final_df_path = final_df_path
    if os.path.exists(final_df_path):
        final_df = pd.read_excel(final_df_path)
        print(f"Đã load final_df từ file: {final_df_path}")
        return final_df
    else:
        print("File final_df chưa tồn tại.")
        return None
def convert_df_to_documents(final_df):
    documents = []
    for _, row in final_df.iterrows():
        # Handle metadata - convert to dict if it's a string
        metadata = row["metadata"]
        if isinstance(metadata, str):
            source = metadata
        else:
            source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"

        # Create Document with properly handled metadata
        doc = Document(
            page_content=row["text"],
            metadata={
                "source": source,
                "level": row["level"]
            }
        )
        documents.append(doc)

    # Process documents
    processed_documents = []
    for doc in documents:
        # Handle source field
        if isinstance(doc.metadata["source"], list):
            doc.metadata["source"] = " ".join(doc.metadata["source"]) if doc.metadata["source"] else "..."
        elif doc.metadata["source"] is None:
            doc.metadata["source"] = "..."

        # Skip empty content
        if not doc.page_content.strip():
            print(f"Warning: Page content is empty for document: {doc.metadata['source']}")
            continue

        processed_documents.append(doc)

    print(f"Processed {len(processed_documents)} out of {len(documents)} documents.")
    return processed_documents

