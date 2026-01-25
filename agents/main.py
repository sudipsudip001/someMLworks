import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def multiply(a: int, b: int) -> int:
    """Used for multiplying two numbers."""
    return a * b

def add(a: int, b: int) -> int:
    """Used for adding two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Used for subtracting two numbers."""
    return a - b

def divide(a: int, b: int) -> float:
    """Used for dividing two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

# llm = GoogleGenAI(model_name="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY) #uncomment to use gemini-2.5-flash
# llm = GoogleGenAI(model="models/gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY) #uncomment to use gemini-2.5-flash-lite
llm = GoogleGenAI(model="models/gemini-robotics-er-1.5-preview", api_key=GOOGLE_API_KEY)
agent = FunctionAgent(
    tools=[multiply, add, subtract, divide],
    llm=llm,
    system_prompt="You are a helpful calculator agent. Use the provided functions to perform calculations as needed.",
)

async def main():
    print("Initializing agents...")
    response = await agent.run("What is ((30 * 10) - 5) * 4 + 1?")
    print("Response:", str(response))


if __name__ == "__main__":
    asyncio.run(main())
