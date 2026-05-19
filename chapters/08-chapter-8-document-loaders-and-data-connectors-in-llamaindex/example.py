from llama_index.readers.file import PDFReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.database import CassandraReader
from llama_index.schema import Document
from llama_index.llms import MockLLM
from llama_index.embeddings import MockEmbedding
from llama_index.indices.vector_store import VectorStoreIndex

# 1. Simulate data sources (no actual file/web/db needed for a runnable example)
# In a real scenario, you'd have actual PDF files, live webpages, and a Cassandra cluster.

# Mock PDF content
mock_pdf_content = "This is some text from a mock PDF document. It discusses LlamaIndex features."
with open("mock_document.pdf", "w") as f:
    f.write(mock_pdf_content)

# Mock webpage content
mock_web_url = "http://example.com/mock-page"
mock_web_content = "<html><body><h1>Mock Web Page</h1><p>Information about data connectors.</p></body></html>"
# For SimpleWebPageReader, it would fetch from a live URL.
# Here, we'll simulate the load by directly creating a Document later.

# Mock Cassandra data (simplified for illustration)
mock_cassandra_data = [
    {"id": "1", "text": "Cassandra entry 1: Distributed database info."},
    {"id": "2", "text": "Cassandra entry 2: Scalability concepts."}
]

# 2. Instantiate various LlamaIndex Document Loaders
# For a runnable example *without external services*, we'll manually create Documents
# or use readers that can operate on local mocks if possible.

# A. PDF Loader (using a local mock file)
pdf_loader = PDFReader()
pdf_documents = pdf_loader.load_data(file="mock_document.pdf")
print(f"Loaded {len(pdf_documents)} document(s) from PDF.")

# B. Web Page Loader (simulating the output, as SimpleWebPageReader needs live URL)
# In a real scenario: web_documents = SimpleWebPageReader(urls=[mock_web_url]).load_data()
web_documents = [Document(text=mock_web_content, metadata={"source": mock_web_url})]
print(f"Loaded {len(web_documents)} document(s) from Web Page.")


# C. Database Loader (Cassandra, simulating the output)
# In a real scenario, CassandraReader requires connection details.
# reader = CassandraReader(session=mock_cassandra_session, keyspace="my_keyspace", table="my_table")
# cassandra_documents = reader.load_data()
cassandra_documents = [
    Document(text=row["text"], metadata={"source": "Cassandra", "id": row["id"]})
    for row in mock_cassandra_data
]
print(f"Loaded {len(cassandra_documents)} document(s) from Cassandra.")

# 3. Combine documents from different sources
all_documents = pdf_documents + web_documents + cassandra_documents
print(f"\nTotal documents loaded from all sources: {len(all_documents)}")

# 4. Create an index with the combined data (using MockLLM/MockEmbedding for non-service run)
llm = MockLLM()
embed_model = MockEmbedding(vector_size=1536) # Use a common embedding size

index = VectorStoreIndex.from_documents(
    all_documents,
    llm=llm,
    embed_model=embed_model,
)
print("\nSuccessfully built VectorStoreIndex from diverse data sources.")

# 5. Query the index (demonstrating unified access)
query_engine = index.as_query_engine()
response = query_engine.query("What information is available about data connectors?")
print(f"\nQuery Response: {response}")

# Clean up mock file
import os
os.remove("mock_document.pdf")
