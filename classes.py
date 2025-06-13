"""Defines state classes for the MissionOSExplore Demo application.

This module provides a typed dictionary for structuring the application's state
in the LangGraph workflow.
"""

from typing import List, TypedDict, Optional
from typing_extensions import Annotated
from langgraph.graph import MessagesState
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from parameters import table_info


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


class CustomInfoSQLDatabaseToolInput(BaseModel):
    table_names: List[str] = Field(description='List of names of tables for which to get descriptions, schema and relationships')


class CustomInfoSQLDatabaseTool(BaseTool):
    name: str = 'SchemaGetter'
    description: str = 'Use to decide which tables to use, and at the same time get the schema for the chosen tables.'
    args_schema: Optional[ArgsSchema] = CustomInfoSQLDatabaseToolInput
    return_direct: bool = False

    def _run(
        self, table_names: List[str], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        """Use the tool."""
        return [table for table in table_info if table['name'] in table_names]