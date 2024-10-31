from data_ingestion import setup_knowledge_base # step1_loaddata
from chunking import text_splitter # step2_chunking
from raptor import RaptorPipeline # step3_RAPTOR
from vector_db import MilvusStorePipeline  # step4_vector_store
from vector_db import MilvusRetrieval # step5_retrieval
import pickle
import os
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from model_config import load_gpt4o_mini_model
import streamlit as st 
def step1_load_data():
    """
    Load và làm sạch tài liệu PDF.
    """
    # Đường dẫn đến thư mục chứa file PDF
    pdf_directory = r'D:\DATN\QA_System\data'
    # Đường dẫn đến file PDF cụ thể (có thể có hoặc không)
    specific_pdf = r'D:\DATN\QA_System\data\20230710 1. QĐ Học bổng KKHT 2023.pdf'
    
    # Gọi hàm setup_knowledge_base để load và làm sạch tài liệu
    documents = setup_knowledge_base(pdf_directory, specific_pdf= specific_pdf)

    # In số lượng tài liệu đã load và làm sạch
    print(f"Tổng số trang tài liệu đã load và làm sạch: {len(documents)}")
    return documents
def step2_chunking():
    pages = step1_load_data()
    chunks = text_splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = str(i)
    chunks_metadata = [chunk.metadata for chunk in chunks]
    chunks_content = [chunk.page_content for chunk in chunks]
    return chunks_metadata, chunks_content

def step3_RAPTOR(refresh_final_df: bool = False):
    """
    Gom cụm và tóm tắt các tài liệu.
    
    Tham số:
    - refresh_final_df: Bool, nếu True sẽ làm mới và xây dựng lại final_df từ đầu.
    """
    # Đường dẫn để lưu DataFrame (pickle format)
    final_df_path = "final_df768.pkl"
    
    if not refresh_final_df and os.path.exists(final_df_path):
        # Nếu file tồn tại và không cần làm mới, tải final_df từ file đã lưu
        print("Đang tải final_df từ file...")
        final_df = pd.read_pickle(final_df_path)
    else:
        # Nếu file chưa tồn tại hoặc yêu cầu làm mới, xây dựng final_df từ đầu
        print("Xây dựng final_df từ đầu...")
        chunks_metadata, chunks_content = step2_chunking()
        raptor = RaptorPipeline()
        
        # Thực hiện gom cụm và tóm tắt
        results = raptor.recursive_embed_cluster_summarize(chunks_content, chunks_metadata, level=1, n_levels=3)
        
        # Xây dựng DataFrame cuối cùng
        final_df = raptor.build_final_dataframe(results)
        
        # Lưu final_df vào file pickle
        final_df.to_pickle(final_df_path)
        print(f"Đã lưu final_df vào file: {final_df_path}")
        # Lưu trữ csv
        final_df.to_csv("final_df768.csv", index=False)
        print("Đã lưu trữ vector của các tài liệu vào file final_df.csv.")
    
    print(final_df.head())
    return final_df
def step4_vector_store():
    """
    Store document vectors in Qdrant Cloud.
    """
    final_df = step3_RAPTOR()
    store = MilvusStorePipeline(collection_name="test", df=final_df)
    store.create_collection_cloud()
    print("Stored document vectors in Milvus Cloud.")

def step5_retrieval(query: str, collection_name: str = "test", top_k: int = 5):
    """
    Perform a retrieval operation on Milvus with the specified query.
    
    :param query: The query string for similarity search.
    :param collection_name: The name of the Milvus collection.
    :param top_k: Number of top similar documents to retrieve.
    :return: Retrieved context documents.
    """
    retrieval = MilvusRetrieval(collection_name=collection_name)
    combined_input = retrieval.similarity_search(query=query, top_k=top_k)
    # Create a ChatOpenAI model
    chat_model = load_gpt4o_mini_model()

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = chat_model.invoke(messages)
    return result.content
    # # Display the full result and content only
    # print("\n--- Generated Response ---")
    # print(result.content)

# Streamlit app interface
import time  # Import time for response time calculation

if __name__ == "__main__":
    st.set_page_config(page_title="QA Chatbot System")
    st.title("QA Chatbot System")

    # Initialize session state for input counter if not exists
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    # Use dynamic key for text input
    user_query = st.text_input("Enter your question:", key=f"user_input_{st.session_state.input_key}")

    if st.button("Get Answer"):
        if user_query.strip():
            try:
                # Measure response time
                start_time = time.process_time()
                response = step5_retrieval(query=user_query, top_k=5)  # Original function call
                response_time = time.process_time() - start_time

                # Display response time
                st.write(f"**Response time:** {response_time:.2f} seconds")

                # Check if response is a string or dictionary
                if isinstance(response, str):
                    # If response is a string, display it as the answer
                    st.write("### Answer:")
                    st.write(response)
                else:
                    # If response is a dictionary, display 'answer' and 'context'
                    st.write("### Answer:")
                    st.write(response.get("answer", "No answer available."))

                    # Optionally show document similarity search
                    with st.expander("Document Similarity Search"):
                        for i, doc in enumerate(response.get("context", [])):
                            st.write(f"**Document {i + 1}**")
                            st.write(doc.page_content)
                            st.write("-----------------------------------")

                # Increment the key to force a new input field
                st.session_state.input_key += 1

            except Exception as e:
                st.write("Error:", e)
        else:
            st.write("Please enter a question to get an answer.")
