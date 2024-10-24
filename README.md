# Question_Answering_System
This is a question answering system code implementation.

# Question Answering System with LLM, Qdrant, and Raptor Indexing

## Components

### 1. Data Ingestion
This component loads data from PDF files. It reads the content of the PDFs and splits them into manageable chunks for further processing.

### 2. Text Preprocessing and Creating Vector Embedding
This component preprocesses the ingested text data and creates vector embeddings. It involves tasks such as tokenization, removing stop words, and generating embeddings using models like SentenceTransformers.

### 3. Raptor Indexing and Implementing Collapsed Tree Retrieval
This component performs Raptor indexing on the text data to create hierarchical clusters. It implements collapsed tree retrieval for efficient search and summarization.

### 4. Creating Qdrant vector Database
This component sets up the Drant vector database to store and manage the text embeddings. It involves creating a connection, defining schemas, and inserting data into the database.

### 5. Retrieve Techniques and Reranking
Objective: To create a hybrid search that combines sparse retrievers with dense retrievers.
- **Creating Index**: Index the text embeddings stored in Milvus.
- **Creating Vector Store and Retriever**: Create a vector store and a retriever using the indexed data.
- **Using Reranking Algorithm**: Apply a reranking algorithm like CrossEncoder to refine the retrieved documents.

### 6. LLM for Question Answering
This component uses a Language Model (LLM) to generate answers to user queries based on the context provided by the retrieved documents. It involves constructing prompts and invoking the LLM to generate accurate and relevant responses.

### 7. Video demo (incoming)