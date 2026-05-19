import random

# --- Self-Correcting RAG (Simulated) ---

class DocumentStore:
    """Simulates a document store with some retrieval and feedback."""
    def __init__(self):
        self.documents = {
            "apple_nutrition": "Apples are rich in fiber and vitamin C. 1 medium apple contains about 95 calories.",
            "banana_origin": "Bananas originated in Southeast Asia. They are a good source of potassium.",
            "orange_vitamins": "Oranges are an excellent source of vitamin C and boost immunity."
        }
        self.feedback_logs = []

    def retrieve(self, query):
        """Simulates document retrieval based on keywords."""
        relevant_docs = []
        query_terms = query.lower().split()
        for doc_id, content in self.documents.items():
            if any(term in content.lower() for term in query_terms):
                relevant_docs.append(content)
        return relevant_docs if relevant_docs else ["No direct document found."]

    def log_feedback(self, query, retrieved_docs, generated_answer, is_correct):
        """Logs feedback for future self-correction."""
        self.feedback_logs.append({
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_answer": generated_answer,
            "is_correct": is_correct
        })
        print(f"Feedback logged: Query='{query}', Correct={is_correct}")

    def get_incorrect_retrievals(self):
        """Identifies past incorrect retrievals for potential re-indexing or re-ranking."""
        return [log for log in self.feedback_logs if not log['is_correct']]

class LLMEmulator:
    """Simulates an LLM for answering questions given context."""
    def generate_answer(self, query, context):
        """Generates a plausible answer based on context, simulating LLM output."""
        if not context or "No direct document found" in context:
            return f"I couldn't find enough information to answer: '{query}'."
        
        # Simple simulation: just combine context
        combined_context = " ".join(context)
        if "apple" in query.lower() and "fiber" in combined_context:
            return f"Based on the documents, {query}: Apples contain fiber and vitamin C."
        if "banana" in query.lower() and "potassium" in combined_context:
            return f"Based on the documents, {query}: Bananas are a good source of potassium."
        if "orange" in query.lower() and "immunity" in combined_context:
            return f"Based on the documents, {query}: Oranges boost immunity with vitamin C."
        
        return f"Based on the provided context: '{combined_context}', I can infer something about '{query}'."


# --- Multi-Agent System (Simulated) ---

class Agent:
    """Base class for a simulated agent."""
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def act(self, task_description, shared_context):
        """Simulates an agent performing an action."""
        raise NotImplementedError

class RetrievalAgent(Agent):
    """Agent responsible for document retrieval."""
    def __init__(self, name, doc_store: DocumentStore):
        super().__init__(name, "Retrieval Specialist")
        self.doc_store = doc_store

    def act(self, task_description, shared_context):
        query = shared_context.get("user_query", task_description)
        print(f"[{self.name}]: Retrieving documents for query: '{query}'")
        retrieved_docs = self.doc_store.retrieve(query)
        shared_context["retrieved_docs"] = retrieved_docs
        return f"Retrieved {len(retrieved_docs)} documents."

class AnsweringAgent(Agent):
    """Agent responsible for generating answers."""
    def __init__(self, name, llm: LLMEmulator):
        super().__init__(name, "Answer Generator")
        self.llm = llm

    def act(self, task_description, shared_context):
        query = shared_context.get("user_query", task_description)
        retrieved_docs = shared_context.get("retrieved_docs", [])
        print(f"[{self.name}]: Generating answer using retrieved docs.")
        answer = self.llm.generate_answer(query, retrieved_docs)
        shared_context["generated_answer"] = answer
        return f"Generated answer: {answer}"

class EvaluationAgent(Agent):
    """Agent responsible for evaluating answers and providing feedback."""
    def __init__(self, name, doc_store: DocumentStore):
        super().__init__(name, "Evaluator")
        self.doc_store = doc_store

    def act(self, task_description, shared_context):
        query = shared_context.get("user_query")
        generated_answer = shared_context.get("generated_answer")
        retrieved_docs = shared_context.get("retrieved_docs")

        # Simulate evaluation - for demonstration, assume random correctness
        is_correct = random.choice([True, False]) 
        print(f"[{self.name}]: Evaluating answer for '{query}'. Is it correct? {is_correct}")
        self.doc_store.log_feedback(query, retrieved_docs, generated_answer, is_correct)
        shared_context["evaluation_result"] = is_correct
        return f"Answer evaluated as {is_correct}."

# --- Orchestration ---

def run_multi_agent_rag_cycle(user_query, doc_store, llm_emulator):
    """Orchestrates a self-correcting RAG cycle with multiple agents."""
    shared_context = {"user_query": user_query}

    retrieval_agent = RetrievalAgent("RetrieveBot", doc_store)
    answering_agent = AnsweringAgent("AnswerBot", llm_emulator)
    evaluation_agent = EvaluationAgent("EvalBot", doc_store)

    agents_in_sequence = [retrieval_agent, answering_agent, evaluation_agent]

    print(f"\n--- Multi-Agent RAG Cycle for: '{user_query}' ---")
    for agent in agents_in_sequence:
        print(f"Executing {agent.name} ({agent.role})...")
        agent_output = agent.act(user_query, shared_context)
        print(f"--> {agent_output}")

    print("\n--- Cycle Complete ---")
    print(f"Final Answer: {shared_context.get('generated_answer')}")
    print(f"Evaluation: {'Correct' if shared_context.get('evaluation_result') else 'Incorrect'}")

# --- Main Execution ---
if __name__ == "__main__":
    doc_store = DocumentStore()
    llm_emulator = LLMEmulator()

    # First cycle
    run_multi_agent_rag_cycle("What are the health benefits of an apple?", doc_store, llm_emulator)

    # Second cycle - simulating a different query
    run_multi_agent_rag_cycle("Where do bananas come from?", doc_store, llm_emulator)
    
    # Third cycle - query that might lead to "incorrect" evaluation more often
    run_multi_agent_rag_cycle("What is the capital of France?", doc_store, llm_emulator)

    print("\n--- After a few cycles, checking for incorrect retrievals ---")
    incorrect = doc_store.get_incorrect_retrievals()
    if incorrect:
        print(f"Found {len(incorrect)} past instances that were marked incorrect.")
        print("This feedback can be used to re-rank documents, add new information, or fine-tune models.")
        # In a real system, you'd trigger re-indexing or human review here
        for item in incorrect[:2]: # Show first 2 for brevity
            print(f"  - Query: '{item['query']}', Generated: '{item['generated_answer']}'")
    else:
        print("No incorrect evaluations logged yet, or all were correct (by chance).")

    print("\nThis simulation demonstrates a basic self-correcting RAG within a multi-agent framework.")
    print("Feedback from evaluation agents can directly inform and improve retrieval and generation components.")
