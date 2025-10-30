from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage
from typing import List

def query_clarifier_agent(llm: BaseLanguageModel, clarification_requests: List[str], chat_history: List[BaseMessage]) -> AIMessage:
    if not clarification_requests:
        return AIMessage(
            name="QueryClarifier",
            content="",
            additional_kwargs={"stage": "final", "process": "response"}
        )

    requests_str = "\\n- ".join(clarification_requests)
    history_str = "\\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    system_prompt = """You are a helpful AI assistant that helps clarify user queries for a data analysis system.

Based on the following clarification requests from the system:
{requests}

And the chat history:
{history}

Craft a polite, helpful and concise response that:
- Informs the user that the query lacks sufficient information.
- Guides the user to provide specific missing information.
"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    response = chain.invoke({"requests": requests_str, "history": history_str})
    content = response.content if hasattr(response, 'content') else str(response)

    return AIMessage(
        name="QueryClarifier",
        content=content,
        additional_kwargs={"stage": "final", "process": "response"}
    )