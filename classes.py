"""Defines state classes for the MissionHelp Demo application.

This module provides a typed dictionary for structuring the application's state
in the LangGraph workflow.
"""

from typing import List, TypedDict
from typing_extensions import Annotated
from langgraph.graph import MessagesState


class State(MessagesState):
    """State class for managing conversation and multimedia in LangGraph.

    Attributes:
        messages: List of conversation messages (human, AI, or tool).
        timings: List of dictionaries with timing details for nodes and components.
    """
    timings: List[dict]


class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]