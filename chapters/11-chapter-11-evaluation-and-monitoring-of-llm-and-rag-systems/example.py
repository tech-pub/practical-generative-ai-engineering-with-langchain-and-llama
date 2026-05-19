from langchain.evaluation import EvaluationResult, EvaluatorType
from langchain.evaluation.qa import generate_qa_pairs, QAEvaluator
from langchain.evaluation.retrieval import RetrievalEvaluator
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM
from langchain_community.docstore import InMemoryDocumentStore
from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document

# --- 1. Simulate an LLM and RAG system for demonstration ---
# We'll use a fake LLM and a simple document store/retriever for local testing.

# A fake LLM that cycles through predefined responses
responses = [
    "The capital of France is Paris.",
    "Paris is a beautiful city.",
    "I'm not sure about that.",
    "France is in Europe."
]
llm = FakeListLLM(responses=responses)

# A simple retriever using an in-memory document store
# In a real RAG system, this would be a vector store and a more sophisticated retriever
document_store = InMemoryDocumentStore()
document_store.add_documents([
    Document(page_content="The capital of France is Paris.", metadata={"source": "wiki"}),
    Document(page_content="Eiffel Tower is in Paris.", metadata={"source": "travel"}),
    Document(page_content="France is a democratic republic.", metadata={"source": "gov"}),
])

class SimpleInMemoryRetriever:
    def get_relevant_documents(self, query: str):
        # Very basic keyword matching for demonstration
        relevant_docs = [
            doc for doc in document_store.yield_documents()
            if query.lower() in doc.page_content.lower() or any(k in doc.page_content.lower() for k in query.lower().split())
        ]
        return relevant_docs

retriever = SimpleInMemoryRetriever()

# --- 2. Generate synthetic QA pairs for evaluation ---
# In a real scenario, these could be manually curated or generated from source documents.
# For simplicity, we'll manually define a few.
qa_pairs = [
    {"query": "What is the capital of France?", "answer": "Paris"},
    {"query": "Where is the Eiffel Tower?", "answer": "Paris"},
]

# --- 3. Evaluate the RAG system's retrieval performance ---
print("--- Evaluating Retrieval Performance ---")
retrieval_evaluator = RetrievalEvaluator(retriever=retriever)
retrieval_results: list[EvaluationResult] = []

for qa in qa_pairs:
    retrieval_results.append(retrieval_evaluator.evaluate_retrieval(query=qa["query"], expected_response=qa["answer"]))

for i, res in enumerate(retrieval_results):
    print(f"Query: '{qa_pairs[i]['query']}'")
    print(f"  Retrieval Score: {res.score}")
    print(f"  Retrieved Documents: {[d.page_content for d in res.retrieved_documents[:2]]}") # Show top 2
    print("-" * 20)

# --- 4. Evaluate the LLM generation performance (using a simple QA evaluator) ---
print("\n--- Evaluating Generation Performance ---")
# A simple prompt for our fake LLM
prompt = PromptTemplate.from_template("Answer the question: {question}")
llm_chain = LLMChain(llm=llm, prompt=prompt)

qa_evaluator = QAEvaluator(llm=llm, prompt=prompt) # In a real test, this 'llm' would be a separate, perhaps better, judge LLM

generation_results: list[EvaluationResult] = []

for qa in qa_pairs:
    # Simulate the full RAG process: retrieve then generate
    docs = retriever.get_relevant_documents(qa["query"])
    context = "\n".join([d.page_content for d in docs])
    
    # Pass context to LLM (even if our simple FakeListLLM ignores it)
    predicted_answer = llm_chain.run(question=qa["query"], context=context) 
    
    # Evaluate the generated answer against the ground truth
    generation_results.append(qa_evaluator.evaluate_qa(
        query=qa["query"],
        contexts=[d.page_content for d in docs], # Pass retrieved docs as context for evaluation
        prediction=predicted_answer,
        expected_response=qa["answer"]
    ))

for i, res in enumerate(generation_results):
    print(f"Query: '{qa_pairs[i]['query']}'")
    print(f"  Predicted: '{res.predicted}'")
    print(f"  Expected: '{res.expected}'")
    print(f"  Generation Score (e.g., faithfulness/coherence by judge): {res.score}") # Score would be from judge LLM
    print("-" * 20)

print("\nEvaluation complete. In a real system, these scores would guide improvements.")
