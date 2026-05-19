from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# Define a mock LLM for demonstration purposes (no external API calls)
class MockLLM(OpenAI):
    def __init__(self, responses=None):
        super().__init__()
        self.responses = responses if responses is not None else {}
        self.counter = 0

    def _call(self, prompt, stop=None, run_manager=None):
        # Simulate different responses based on call order or prompt content
        if "analyze" in prompt.lower():
            return "This seems like a positive sentiment and highlights innovation."
        elif "summarize" in prompt.lower():
            return "The main point is about positive innovation."
        elif "recommendation" in prompt.lower():
            return "Based on positive innovation, consider investing in new tech."
        else:
            return "Mock LLM response for: " + prompt[:50] + "..."

    @property
    def _llm_type(self):
        return "mock"

# 1. Simple LLMChain: A single prompt and LLM call
print("--- Simple LLMChain Example ---")
llm = MockLLM()
prompt_template = "What is the capital of {country}?"
llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
# print(llm_chain.run("France")) # This would typically call the LLM

# 2. SimpleSequentialChain: Chains multiple LLMChains where the output of one
# becomes the input of the next. Output is a single string.
print("\n--- SimpleSequentialChain Example ---")
template1 = "Analyze the sentiment of the following text: {text}"
first_llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template1))

template2 = "Summarize the key findings from this sentiment analysis: {text}"
second_llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template2))

simple_sequential_chain = SimpleSequentialChain(chains=[first_llm_chain, second_llm_chain], verbose=True)
# print(simple_sequential_chain.run("This new product is incredibly innovative and a game-changer!"))

# 3. SequentialChain: More advanced, allows for multiple inputs and outputs,
# and named intermediate steps for better debugging and control.
print("\n--- SequentialChain Example ---")
# Step 1: Analyze text input
prompt1 = PromptTemplate(
    input_variables=["review"],
    template="Analyze the core theme and sentiment of the following product review: {review}\nAnalysis:",
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="analysis_result")

# Step 2: Formulate a recommendation based on the analysis
prompt2 = PromptTemplate(
    input_variables=["analysis_result"],
    template="Based on the analysis: '{analysis_result}', provide a concise business recommendation:",
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="business_recommendation")

# Combine into a SequentialChain
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["review"],
    output_variables=["analysis_result", "business_recommendation"],
    verbose=True
)

# Run the chain with an example review
review_text = "The new software update is buggy and frustrating, significantly hindering productivity."
output = overall_chain({"review": review_text})
#
# print(f"\nOriginal Review: {review_text}")
# print(f"Analysis Result: {output['analysis_result']}")
# print(f"Business Recommendation: {output['business_recommendation']}")
#
# # Note: The mock LLM provides predefined responses, so actual analysis
# # output will be fixed regardless of input text for this example.
