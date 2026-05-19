from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.llms import FakeListLLM # Simulate an LLM for local execution

# 1. Define Tools: Functions the agent can use to interact with the world.
#    These are simple Python functions in this example.

def get_current_weather(location: str) -> str:
    """Returns the current weather for a given location."""
    if "London" in location:
        return "It's cloudy with a temperature of 15°C."
    elif "Paris" in location:
        return "It's sunny and warm at 25°C."
    else:
        return "Weather data not available for this location."

def search_wikipedia(query: str) -> str:
    """Performs a search on Wikipedia and returns a summary."""
    if "Eiffel Tower" in query:
        return "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    elif "Python programming language" in query:
        return "Python is a high-level, general-purpose programming language."
    else:
        return "Could not find information on Wikipedia for that query."

# 2. Package Tools for LangChain Agent
tools = [
    Tool(
        name="get_weather",
        func=get_current_weather,
        description="Useful for finding out the current weather in a specific city."
    ),
    Tool(
        name="search_wiki",
        func=search_wikipedia,
        description="Useful for general knowledge queries by searching Wikipedia."
    ),
]

# 3. Create a Fake LLM for local demonstration without external API keys.
#    This LLM will cycle through predefined responses.
fake_llm_responses = [
    "Thought: I need to find the weather in London. I will use the 'get_weather' tool.",
    "Action: get_weather",
    "Action Input: London",
    "Observation: It's cloudy with a temperature of 15°C.",
    "Thought: I have the weather information. I can now answer the question.",
    "Final Answer: The current weather in London is cloudy with a temperature of 15°C.",
    "Thought: I need to find information about the Eiffel Tower. I will use the 'search_wiki' tool.",
    "Action: search_wiki",
    "Action Input: Eiffel Tower",
    "Observation: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Thought: I have the information about the Eiffel Tower. I can now answer the question.",
    "Final Answer: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
]
fake_llm = FakeListLLM(responses=fake_llm_responses)

# 4. Define the Agent's Prompt Template (ReAct style for decision making)
#    This template guides the LLM on how to reason and use tools.
agent_prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# 5. Create the Agent
#    create_react_agent combines the LLM, tools, and prompt into an agent definition.
agent = create_react_agent(fake_llm, tools, agent_prompt)

# 6. Create the Agent Executor
#    AgentExecutor runs the agent, managing the interaction loop (LLM -> Tool -> LLM).
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. Run the Agent with a query
print("--- Running Agent for Weather Query ---")
agent_executor.invoke({"input": "What is the weather like in London today?"})

print("\n--- Running Agent for Wikipedia Query ---")
agent_executor.invoke({"input": "Tell me about the Eiffel Tower."})
