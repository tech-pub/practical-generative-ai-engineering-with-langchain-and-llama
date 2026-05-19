import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# --- LlamaIndex for Data Ingestion and Indexing ---
# Create a dummy data directory and file for demonstration
os.makedirs("data", exist_ok=True)
with open("data/policy.txt", "w") as f:
    f.write("Our company policy states that vacation days accumulate at 1.5 days per month. Maximum carry-over is 10 days.")
    f.write("\nEmployees are eligible for a 5% bonus if they meet all annual performance targets.")

# Load documents and create LlamaIndex vector store
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=2)

# --- LangChain for Conversational RAG ---
# Initialize LangChain's LLM (using a mock or dummy LLM for no external service)
# In a real scenario, you'd use ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="YOUR_KEY")
class MockLLM:
    def invoke(self, prompt, **kwargs):
        if "vacation" in prompt.lower():
            return "From the policy, vacation days accumulate at 1.5 days per month. Max carry-over is 10 days."
        elif "bonus" in prompt.lower():
            return "Based on the policy, a 5% bonus is offered if annual performance targets are met."
        return "I need more information to answer that based on the provided documents."

llm = MockLLM()

# Initialize LangChain's conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a LangChain conversational retrieval chain fused with LlamaIndex retriever
# The `combine_docs_chain` is usually behind the scenes with ConversationalRetrievalChain
# For a full integration, you would typically integrate LlamaIndex's retriever directly.
# For this example, we simulate the retrieval aspect without deep integration
# since LangChain's ConversationalRetrievalChain expects a 'retriever' compatible with LangChain's API.
# Here, we 'wrap' the LlamaIndex retriever for demonstration.
class LlamaIndexRetrieverLangChainAdapter:
    def __init__(self, llama_retriever):
        self.llama_retriever = llama_retriever

    def get_relevant_documents(self, query):
        nodes = self.llama_retriever.retrieve(query)
        # Convert LlamaIndex Nodes to LangChain Document format
        from langchain_core.documents import Document
        return [Document(page_content=node.text_content, metadata=node.metadata) for node in nodes]

langchain_retriever = LlamaIndexRetrieverLangChainAdapter(retriever)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=langchain_retriever,
    memory=memory
)

# --- Example Usage ---
print("--- Starting RAG Conversation ---")

# First query
question1 = "How many vacation days do I get?"
result1 = qa_chain.invoke({"question": question1})
print(f"User: {question1}\nAI: {result1['answer']}\n")

# Second query, should leverage chat history and RAG
question2 = "What about bonuses?"
result2 = qa_chain.invoke({"question": question2})
print(f"User: {question2}\nAI: {result2['answer']}\n")

# Clean up dummy data
os.remove("data/policy.txt")
os.rmdir("data")
