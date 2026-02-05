from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for : {query}")
    return "Weather in Tokyo is Sunny"

@tool
def add_numbers(a: int, b: int) -> int:
    """
    Tool that adds two input integers
    Args:
        a: integer input value
        b: integer input value
    Returns:
        Sum of input integers a and b
    """
    return a+b

# llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash-lite",
#         temperature=0.5,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )

llm = ChatOllama(temperature=0, model="gpt-oss:20b")

tools = [search, add_numbers]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello-World from Langchain Course")
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo and what is the sum of 5 & 6")})
    print(result)

if __name__=="__main__":
    main()