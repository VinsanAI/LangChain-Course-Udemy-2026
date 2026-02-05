from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from typing import List
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

# llm = ChatOllama(temperature=0, model="gpt-oss:20b")

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello-World from Langchain Course")
    # result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo and what is the sum of 8 and 9")})
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 jobs postings for an ai engineeer or data scientist with agentic experience in Bengaluru Karnataka on linkedin and list their details"
            )
        }
    )
    # for message in result["messages"]:
    #     message.pretty_print()
    print(result)

if __name__=="__main__":
    main()