from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.llms import FakeListLLM  # Simulate LLM responses

# 1. Define a Pydantic model for structured output parsing
class ProductInfo(BaseModel):
    product_name: str = Field(description="The name of the product.")
    category: str = Field(description="The primary category of the product.")
    price_usd: float = Field(description="The price of the product in USD.")
    in_stock: bool = Field(description="Whether the product is currently in stock.")

# Instantiate the parser
parser = PydanticOutputParser(pydantic_object=ProductInfo)

# 2. Few-shot examples for the LLM
examples = [
    {
        "input": "Tell me about the 'AquaGlow Smart Water Bottle'.",
        "output": '{"product_name": "AquaGlow Smart Water Bottle", "category": "Smart Home", "price_usd": 49.99, "in_stock": true}'
    },
    {
        "input": "Specs for the 'Titanium X Drone'.",
        "output": '{"product_name": "Titanium X Drone", "category": "Electronics", "price_usd": 129.00, "in_stock": false}'
    }
]

# 3. Create a template for few-shot examples
example_formatter_template = """Input: {input}
Output: {output}"""
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_formatter_template,
)

# 4. Construct the Few-Shot Prompt Template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Extract product information into a JSON format. "
           "Use the following examples for reference:\n",
    suffix="Input: {query}\nOutput:",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 5. Simulate an LLM with predefined responses
# The responses are designed to match the expected JSON format after parsing
llm = FakeListLLM(responses=[
    '{"product_name": "Quantum Leap Keyboard", "category": "Peripherals", "price_usd": 75.50, "in_stock": true}',
    '{"product_name": "Nebula VR Headset", "category": "Electronics", "price_usd": 299.99, "in_stock": false}'
])

# 6. Combine prompt, LLM and parser into a chain
chain = few_shot_prompt | llm | parser

# 7. Invoke the chain with a new query
query1 = "Details about the 'Quantum Leap Keyboard'."
product_info1 = chain.invoke({"query": query1})
print(f"Query 1: {query1}")
print(f"Parsed Product Info 1: {product_info1}")
print(f"Type 1: {type(product_info1)}\n")

query2 = "Is the Nebula VR Headset available and how much is it?"
product_info2 = chain.invoke({"query": query2})
print(f"Query 2: {query2}")
print(f"Parsed Product Info 2: {product_info2}")
print(f"Type 2: {type(product_info2)}")
