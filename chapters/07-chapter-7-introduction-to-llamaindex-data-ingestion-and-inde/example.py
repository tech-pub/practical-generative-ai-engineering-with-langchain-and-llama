import os
from llama_index.readers.base import BaseReader
from llama_index.schema import Document
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms import MockLLM
from llama_index.embeddings import MockEmbedding
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore

# Step 1: Simulate a custom data reader for unstructured text files
# In a real scenario, this would read from PDFs, databases, web pages, etc.
class SimpleFileReader(BaseReader):
    def load_data(self, file_path: str) -> list[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Each document represents a piece of external knowledge
        return [Document(text=text, metadata={"source": file_path})]

# Step 2: Prepare some synthetic data files
# These files represent our "external knowledge base"
os.makedirs("data", exist_ok=True)
with open("data/document1.txt", "w") as f:
    f.write("LlamaIndex is a data framework for LLM applications. It helps connect custom data sources to large language models.")
with open("data/document2.txt", "w") as f:
    f.write("Retrieval Augmented Generation (RAG) improves LLM responses by retrieving relevant information from an external knowledge base.")
with open("data/document3.txt", "w") as f:
    f.write("This chapter explains data ingestion and indexing in LlamaIndex.")

# Step 3: Ingest data using our custom reader
reader = SimpleFileReader()
documents = reader.load_data("data/document1.txt")
documents.extend(reader.load_data("data/document2.txt"))
documents.extend(reader.load_data("data/document3.txt"))

# Step 4: Configure LlamaIndex services using mock components
# For demonstration, we use MockLLM and MockEmbedding to avoid external API calls.
# In a real application, you'd use OpenAI, HuggingFace, etc.
service_context = ServiceContext.from_defaults(
    llm=MockLLM(),
    embed_model=MockEmbedding(),
)

# Step 5: Create a simple in-memory storage context
# This holds our vector store, document store, and index store
storage_context = StorageContext.from_defaults(
    vector_store=SimpleVectorStore(),
    docstore=SimpleDocumentStore(),
    index_store=SimpleIndexStore(),
)

# Step 6: Build the index
# This is the core "indexing" step where documents are processed,
# split into nodes, embedded, and stored for efficient retrieval.
print("Building index from ingested documents...")
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    storage_context=storage_context,
)
print("Index built successfully.")

# Step 7: (Optional) Demonstrate a basic query
# This shows how the indexed data can be queried.
# The query engine will retrieve relevant information based on the question.
query_engine = index.as_query_engine()
query = "What is LlamaIndex used for?"
print(f"\nQuerying: '{query}'")
response = query_engine.query(query)
print(f"LLM Response (from mock): {response}")

# Clean up synthetic data files
os.remove("data/document1.txt")
os.remove("data/document2.txt")
os.remove("data/document3.txt")
os.rmdir("data")

# This example demonstrates:
# 1. Loading custom data (ingestion).
# 2. Using LlamaIndex to process and index this data.
# 3. Setting up a basic in-memory index without external services.
# 4. How this index can then be queried (though the LLM is mocked).
# This process forms the foundation of providing external knowledge to LLMs.
