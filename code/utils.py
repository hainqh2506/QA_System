import pandas as pd

# # Đọc file pickle thành DataFrame
# file_path = r"D:\DATN\QA_System\code\final_df768.pkl"  # Thay bằng đường dẫn file của bạn
# df = pd.read_pickle(file_path)
from langchain.schema import Document
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
