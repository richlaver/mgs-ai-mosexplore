import logging
from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

def history_summariser(messages: List[BaseMessage], llm: BaseLanguageModel) -> str:
    """
    Summarizes the chat history and updates the retrospective_query in the context.
    """

    current_query_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None
    )
    if not current_query_msg:
        raise ValueError("No query found in chat history")

    current_query = current_query_msg.content

    previous_messages = [
        msg
        for msg in messages
        if msg is not current_query_msg and (
            isinstance(msg, HumanMessage)
            or (
                isinstance(msg, AIMessage)
                and getattr(msg, "additional_kwargs", {}) == {"stage": "final", "process": "response"}
            )
        )
    ]

    if len(previous_messages) == 0:
        return current_query

    prompt_messages = [
        (
            "system",
            """
# Role
You are an expert at extracting relevant context from a chat history and incorporating it into the current query.
# Task
Incorporate relevant context from chat history, if any exists, into the current query so that the answer will address the user's intention.
# Instructions
Carefully analyse the provided chat history and current query. If there is relevant prior context, rewrite the query to include it while preserving critical details and relationships. If not, return the query unchanged.
            """
        ),
        (
            "human",
            f"""
Chat history:
{chr(10).join(f"{'Human: ' if isinstance(msg, HumanMessage) else 'Assistant: '}{msg.content}" for msg in previous_messages)}

Current query:
{current_query}

Instructions:
- If no relevant context exists, return the current query verbatim.
- If relevant context exists, return a rewritten query that incorporates the context and preserves critical details and relationships.
            """
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt | llm

    try:
        response = chain.invoke({})
        retrospective_query = response.content
    except Exception as e:
        logger.error(f"Error generating retrospective query: {e}")
        retrospective_query = current_query

    return retrospective_query