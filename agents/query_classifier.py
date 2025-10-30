from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from classes import Context

def query_classifier_agent(llm: BaseLanguageModel, messages: list[BaseMessage], context: Context) -> str:
    """
    Uses the LLM to classify the query complexity and decide between 'ReAct' or 'CodeAct'.
    ReAct: Simple queries (single low-volume DB query, optional single plot, no calculation).
    CodeAct: All others.
    """

    prompt = f"""
# Role
You are an expert at deciding the most suitable type of agent to answer a query.
# Task
To decide if the following query should be answered using a ReAct or a CodeAct implementation:
{context.retrospective_query}
# Context
The query-answering agent will have access to a database.
# Steps
1. Analyse the query.
2. Determine if the agent needs to extract more than five rows of data to answer the query.
3. Determine if the agent needs to perform any calculation to answer the query.
4. If any answer to steps 2 or 3 is yes, then CodeAct must be used, otherwise use ReAct.
# Output
Output your answer as either “CodeAct” or “ReAct” with no other words.
"""

    response = llm.invoke(prompt)
    agent_type = response.content.strip()

    # Hardcode to return CodeAct
    # Delete the line below to revert to dynamic classification
    return "CodeAct"

    if agent_type == "ReAct":
        return "ReAct"
    return "CodeAct"