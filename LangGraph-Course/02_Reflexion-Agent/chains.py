import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from schemas import AnswerQuestion, ReviseAnswer
load_dotenv()

# llm = ChatOllama(temperature=0.8, model="gpt-oss:20b", reasoning="high")

llm = ChatOllama(temperature=0.2, model="llama3.1:8b")
structured_llm = llm.with_structured_output(AnswerQuestion)
structured_llm_revise = llm.with_structured_output(ReviseAnswer)

# llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=1.0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=1,
#     )
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. You MUST reflect and critique your answer. Be severe to maximize improvement.
3. You MUST recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | structured_llm

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | structured_llm_revise



if __name__=="__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )

    chain = first_responder_prompt_template | structured_llm
    res = chain.invoke(input={"messages": [human_message]})
    print(res)