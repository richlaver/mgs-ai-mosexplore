from typing import List, Annotated, Optional, Dict, Any
import operator

from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field

class DbField(BaseModel):
    """Info on database fields to get data relevant to query."""
    db_name: str = Field(description="Name of field in database e.g. 'data1', 'calculation2' etc.")
    db_type: str = Field(description="Type of field e.g. 'data', 'calculation' etc.")
    labels: List[str] = Field(description="Descriptive name for field: use for labelling axes, table headers, how to call field in text e.g. 'Settlement', 'Groundwater level' etc.")
    description: str = Field(description="Description of field")
    units: str = Field(description="Measurement unit for field")

class InstrInfo(BaseModel):
    """Background info on instrument type/subtype."""
    labels: List[str] = Field(description="Descriptive names for instrument as referred to in text")
    physical_form: str = Field(description="Physical manifestation of instrument")
    raw_instr_output: List[str] = Field(description="Unprocessed data outputted by instrument")
    derived_measurands: List[str] = Field(description="Data commonly derived from the unprocessed data")
    purposes: List[str] = Field(description="Common purposes for instrument")
    data_interpretation: str = Field(description="Guidelines for interpreting instrument data")
    typical_plots: List[str] = Field(description="Common plots to represent instrument data")

class DbSource(BaseModel):
    """Info on instrument types, subtypes and fields to reference in database relevant to query, along with background info."""
    instr_type: str = Field(description="Type of instrument")
    instr_subtype: str = Field(description="Subtype of instrument")
    instr_info: InstrInfo = Field(description="Background info on instrument type/subtype")
    db_fields: Optional[List[DbField]] = Field(description="Database fields to get data relevant to query")

class QueryWords(BaseModel):
    """Words in query and data sources relevant to them."""
    query_words: str = Field(description="Words extracted from query")
    data_sources: List[DbSource] = Field(description="Data sources and background info relevant to words extracted from query")

class RelevantDateRange(BaseModel):
    """Date range relevant to user query."""
    explanation: str = Field(description="Explanation of what the date range refers to in the query, why the date range was chosen thus and how to apply the date range to answer the query")
    start_datetime: str = Field(description="Start datetime in ISO 8601 format")
    end_datetime: str = Field(description="End datetime in ISO 8601 format")

class RelevantPlaceCoordinates(BaseModel):
    """Geographical coordinates relevant to user query."""
    description: str = Field(description="Description of what in the query the place coordinates refer to and how they should be used")
    easting: float = Field(description="Easting coordinate in decimal degrees")
    northing: float = Field(description="Northing coordinate in decimal degrees")
    radius_metres: float = Field(description="Radius around coordinates in metres")

class Context(BaseModel):
    """Contextual info to interpret query semantics."""
    retrospective_query: str = Field(description="Query rephrased from current query to incorporate chat history")
    ignore_IDs: Optional[List[str]] = Field(description="Query words user has flagged as not instrument IDs", default=None)
    verif_ID_info: Optional[Dict[str, Dict[str, str]]] = Field(description="Mapping of verified instrument IDs to type/subtype info, e.g. {'0003-L-2': {'type': 'LP', 'subtype': 'MOVEMENT'}}", default=None)
    unverif_IDs: Optional[List[str]] = Field(description="Instrument IDs in query not found in database", default=None)
    word_context: Optional[List[QueryWords]] = Field(description="How query words relate to where to get data and background info to answer", default=None)
    edge_case: Optional[bool] = Field(description="Indicates whether the user query is an edge case", default=None)
    relevant_date_ranges: Optional[List[RelevantDateRange]] = Field(description="Date ranges relevant to user query", default=None)
    review_level_context: Optional[str] = Field(description="Context on anything to do with review levels mentioned in query", default=None)
    relevant_place_coordinates: Optional[RelevantPlaceCoordinates] = Field(description="Geographical coordinates relevant to user query", default=None)
    web_info: Optional[str] = Field(description="Additional relevant information from web search", default=None)
    clarification_requests: Optional[List[str]] = Field(description="List of clarification requests from subagents", default=None)

class Suggestion(BaseModel):
    """Follow-on queries suggested to user."""
    id: int = Field(description="ID for follow-on query")
    suggestion: str = Field(description="Query content")

class Execution(BaseModel):
    """An execution run."""
    agent_type: str = Field(description="Type of agent used for the execution run", enum=["ReAct", "CodeAct"])
    parallel_agent_id: int = Field(description="Differentiates between agents running in parallel during an execution run")
    retry_number: int = Field(description="Differentiates between retry attempts")
    codeact_code: str = Field(description="Code generated for CodeAct agents")
    final_response: Optional[AIMessage] = Field(description="Final textual response from the agent", default=None)
    artefacts: List[AIMessage] = Field(description="Plots and CSV files generated by tools initiated by the agent", default_factory=list)
    error_summary: str = Field(description="Summary of errors from the attempt", default="")
    is_best: bool = Field(description="Indicates whether this execution was selected as the best response", default=False)
    is_sufficient: bool = Field(description="Indicates whether this execution's response is sufficient to answer the query", default=False)

def upsert_execution_list(existing: List["Execution"], incoming: List["Execution"]) -> List["Execution"]:
    """Custom reducer for AgentState.executions.

    Replaces matching executions (by agent_type, parallel_agent_id, retry_number)
    and appends new ones if no match exists. Preserves original order where possible.
    """
    if not existing:
        return list(incoming) if incoming else []
    if not incoming:
        return list(existing)

    def _key(ex: "Execution") -> tuple:
        return (ex.agent_type, ex.parallel_agent_id, ex.retry_number)

    index = { _key(ex): i for i, ex in enumerate(existing) }
    result = list(existing)

    for ex in incoming:
        k = _key(ex)
        if k in index:
            result[index[k]] = ex
        else:
            result.append(ex)
    return result

class AgentState(BaseModel):
    """Agent state schema."""
    messages: Annotated[List[BaseMessage], operator.add] = Field(description="Conversation and tool messages", default_factory=list)
    context: Optional[Context] = Field(description="Info giving context to query", default=None)
    executions: Annotated[List[Execution], upsert_execution_list] = Field(description="History of execution runs", default_factory=list)
    suggestions: Annotated[List[Suggestion], operator.add] = Field(description="Follow-on queries suggested to user", default_factory=list)
    
    def get_current_execution(self) -> Optional[Execution]:
        """
        Get the current (most recent) execution.
        
        Returns:
            The current execution, or None if no executions exist
        """
        if not self.executions:
            return None
        return self.executions[-1]
    
    class Config:
        arbitrary_types_allowed = True


class ContextState(BaseModel):
    """State passed through the context graph for orchestrating agents."""
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    context: Annotated[Dict[str, Any], operator.or_] = Field(default_factory=dict)
    instruments_validated: Annotated[bool, operator.or_] = False
    db_context_provided: Annotated[bool, operator.or_] = False
    period_deduced: Annotated[bool, operator.or_] = False

AgentState.model_rebuild()
Context.model_rebuild()