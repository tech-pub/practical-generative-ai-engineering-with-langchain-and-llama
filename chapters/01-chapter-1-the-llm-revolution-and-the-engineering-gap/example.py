# demonstrate_llm_complexity.py

# This script simulates a simple "chain" of operations that a hypothetical LLM-powered
# application might perform. It highlights how combining even basic text processing
# steps can quickly become complex without a structured framework.

def step_1_data_extraction(text_block: str) -> str:
    """Simulates extracting key information, e.g., finding the 'topic'."""
    # In a real LLM application, this might involve a prompt like:
    # "Given the following text, what is the main topic? [text_block]"
    # For this simulation, we'll just look for a keyword.
    if "Python" in text_block:
        return "Programming Language"
    elif "AI" in text_block:
        return "Artificial Intelligence"
    return "General"

def step_2_sentiment_analysis(extracted_info: str) -> str:
    """Simulates analyzing the sentiment of the extracted information."""
    # In a real LLM application, this might be:
    # "What is the sentiment of this topic: [extracted_info]?"
    # Here, a simple heuristic.
    if "Intelligence" in extracted_info:
        return "Positive"
    elif "General" in extracted_info:
        return "Neutral"
    return "Mixed"

def step_3_response_generation(topic: str, sentiment: str) -> str:
    """Simulates generating a response based on the topic and sentiment."""
    # A real LLM might use a prompt like:
    # "Generate a short, helpful response about [topic] with a [sentiment] tone."
    if sentiment == "Positive":
        return f"That's great you're interested in {topic}! It's a fascinating area."
    elif sentiment == "Neutral":
        return f"The topic is {topic}. Is there anything specific you'd like to know?"
    else:
        return f"Regarding {topic}, there are various perspectives."

def orchestrate_manual_llm_like_process(input_text: str) -> str:
    """
    Manually orchestrates the series of 'LLM' calls.
    This function demonstrates the hard-coded flow and dependencies,
    highlighting the lack of flexibility and reusability without a framework.
    """
    print(f"--- Processing: '{input_text[:30]}...' ---")
    
    # Call Step 1
    topic = step_1_data_extraction(input_text)
    print(f"  Step 1 Result (Topic): {topic}")

    # Call Step 2, dependent on Step 1's output
    sentiment = step_2_sentiment_analysis(topic)
    print(f"  Step 2 Result (Sentiment): {sentiment}")

    # Call Step 3, dependent on Step 1 and Step 2's output
    final_response = step_3_response_generation(topic, sentiment)
    print(f"  Step 3 Result (Response): {final_response}\n")

    return final_response

if __name__ == "__main__":
    example_text_1 = "I really enjoy learning Python programming, it's very versatile for AI development."
    example_text_2 = "Artificial Intelligence is a complex field, with many challenges."
    example_text_3 = "This is just some random text without specific keywords."

    orchestrate_manual_llm_like_process(example_text_1)
    orchestrate_manual_llm_like_process(example_text_2)
    orchestrate_manual_llm_like_process(example_text_3)

    # Imagine adding more steps (e.g., translation, summarization before sentiment).
    # The `orchestrate_manual_llm_like_process` function would grow linearly
    # in complexity and become harder to modify or debug.
    # This complexity is what frameworks like LangChain/LlamaIndex aim to manage.
