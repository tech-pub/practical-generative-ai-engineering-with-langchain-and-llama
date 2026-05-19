# Import necessary components from LangChain
from langchain.llms import OpenAI, FakeListLLM  # Simulate LLMs without actual API calls
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import AgentExecutor, create_json_agent
from langchain.tools import Tool
from langchain.agents.agent_toolkits import JsonToolkit
import json

# 1. Define a simulated LLM for demonstration
# In a real scenario, this would be an actual LLM like OpenAI, HuggingFace, etc.
llm = FakeListLLM(responses=["Hello, how can I help you?", "The capital of France is Paris.", "The sum is 15."])

# 2. Demonstrate an LLMChain: A single LLM call with a structured prompt
print("--- LLMChain Example ---")
# Define a prompt template for asking a general question
prompt_template_general = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful AI assistant. Respond to the following question: {question}"
)
# Create an LLMChain
llm_chain_general = LLMChain(prompt=prompt_template_general, llm=llm)
# Run the chain
response_general = llm_chain_general.run("What is the capital of France?")
print(f"LLMChain response: {response_general}\n")


# 3. Demonstrate a SequentialChain: Orchestrating multiple LLM calls in a sequence
print("--- SequentialChain Example ---")
# First chain: Ask a general question (reusing the one above)
# Second chain: Calculate sum based on a number
prompt_template_math = PromptTemplate(
    input_variables=["number1", "number2"],
    template="Calculate the sum of {number1} and {number2}. Provide only the number."
)
llm_chain_math = LLMChain(prompt=prompt_template_math, llm=llm, output_key="sum_result")

# Create a sequential chain
overall_chain = SequentialChain(
    chains=[llm_chain_math],  # For simplicity, we'll only have one chain here to show structure
    input_variables=["number1", "number2"],
    output_variables=["sum_result"],
    verbose=True
)
response_sequential = overall_chain({"number1": 10, "number2": 5})
print(f"SequentialChain response: {response_sequential['sum_result']}\n")


# 4. Demonstrate Agents: LLMs deciding which tool to use
print("--- Agent Example ---")
# Define a simple "tool" that the agent can use
def get_user_info(user_id: str) -> str:
    """Returns information about a specific user."""
    users = {"123": {"name": "Alice", "email": "alice@example.com"}, "456": {"name": "Bob", "email": "bob@example.com"}}
    return json.dumps(users.get(user_id, {"error": "User not found"}))

# Create a LangChain Tool from our function
tool_get_user_info = Tool(
    name="GetUserInformation",
    func=get_user_info,
    description="Useful for getting detailed information about a user by their ID."
)

# For demonstration, we'll use a very basic JSON agent.
# In a real scenario, you'd specify an actual LLM and a more robust toolkit.
toolkit = JsonToolkit(spec=tool_get_user_info.description) # A bit of a hack to simulate, normally takes JSON spec
tools = [tool_get_user_info]

# Create a JSON agent; this agent uses an LLM to decide when and how to use the provided tool.
# We're using FakeListLLM here, meaning the agent's "thought process" is predefined.
agent = create_json_agent(
    llm=llm,
    toolkit=toolkit,
    tools=tools, # Pass the actual callable tool
    verbose=True
)

# Run the agent with a query that requires using the tool
agent_response = agent.run("What is the email of user 123?")
print(f"Agent response: {agent_response}")
