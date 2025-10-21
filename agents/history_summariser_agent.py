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

    current_query = ""
    current_query = next(
        (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
        None
    )
    if not current_query:
        raise ValueError("No query found in chat history")

    previous_messages = [msg for msg in messages[:-1] if isinstance(msg, (HumanMessage, AIMessage))]

    if len(previous_messages) == 0:
        return current_query

    prompt_messages = [
        ("system", """
You are a context-aware assistant. 
Review the chat history and current query, 
then rewrite the current query to include relevant context from previous messages. 
Focus on maintaining critical details and relationships from the conversation that would help answer the current query accurately.
If there is no relevant context from previous messages, just return the current query as-is.
             """),
        ("human", f"""
Chat history (newest last):
{chr(10).join(f"{'Human: ' if isinstance(msg, HumanMessage) else 'Assistant: '}{msg.content}" 
               for msg in previous_messages)}

Current query: {current_query}

Rewrite the query incorporating relevant context from the chat history, or return the query as-is if no relevant context exists.
            """)
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