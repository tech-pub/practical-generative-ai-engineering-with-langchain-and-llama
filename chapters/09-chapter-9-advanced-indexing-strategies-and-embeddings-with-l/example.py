import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, KeywordTableIndex, KnowledgeGraphIndex
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.simple import SimpleGraphStore
from llama_index.llms.mock import MockLLM
from functools import lru_cache

# --- Data Preparation ---
# Create a dummy text file for demonstration
data_dir = "temp_data"
os.makedirs(data_dir, exist_ok=True)
with open(os.path.join(data_dir, "document1.txt"), "w") as f:
    f.write("Alice is a software engineer. Bob is a data scientist. They both work at TechCorp.")
with open(os.path.join(data_dir, "document2.txt"), "w") as f:
    f.write("TechCorp developed a new AI model for healthcare. The model helps diagnose diseases.")

# Load documents
documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
print(f"Loaded {len(documents)} documents.")
[os.remove(os.path.join(data_dir, f)) for f in os.listdir(data_dir)] # Clean up
os.rmdir(data_dir)

# --- Embedding Model (HuggingFace for local execution) ---
# Use an LRU cache to avoid re-initializing the embedding model if called multiple times
@lru_cache(maxsize=1)
def get_embedding_model():
    # Using a small, fast local embedding model for demonstration
    # You might choose 'sentence-transformers/all-MiniLM-L6-v2' for better performance
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# --- Indexing Strategies ---

# 1. Vector Store Index (for semantic similarity)
print("\n--- Creating Vector Store Index ---")
# Instantiate embedding model for the index
vector_index = VectorStoreIndex.from_documents(
    documents,
    embed_model=get_embedding_model(),
)
query_engine_vector = vector_index.as_query_engine()
response_vector = query_engine_vector.query("What does Alice do?")
print(f"Vector Store Response: {response_vector}")

# 2. Keyword Table Index (for exact keyword matching)
print("\n--- Creating Keyword Table Index ---")
keyword_index = KeywordTableIndex.from_documents(documents)
query_engine_keyword = keyword_index.as_query_engine()
response_keyword = query_engine_keyword.query("Who works at TechCorp?")
print(f"Keyword Table Response: {response_keyword}")

# 3. Knowledge Graph Index (for structured relationships, requires an LLM for extraction)
# We use a MockLLM for local execution without an actual LLM service
print("\n--- Creating Knowledge Graph Index ---")
graph_store = SimpleGraphStore()
llm = MockLLM() # Mock LLM for local graph extraction
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    kg_triplet_extract_llm=llm,
    storage_context=None, # Not using persistent storage for this example
    graph_store=graph_store,
    include_embeddings=True, # Embed nodes/relationships in the graph
    embed_model=get_embedding_model(),
    show_progress=False,
)
query_engine_kg = kg_index.as_query_engine()
response_kg = query_engine_kg.query("Tell me about TechCorp's products.")
print(f"Knowledge Graph Response: {response_kg}") # Note: MockLLM makes this output generic but shows pipeline.

print("\nDemonstration complete.")
