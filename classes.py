from typing import List, Annotated, Optional, Dict, Any
import operator

from langchain_core.messages import BaseMessage
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

class Suggestion(BaseModel):
    """Follow-on queries suggested to user."""
    id: int = Field(description="ID for follow-on query")
    suggestion: str = Field(description="Query content")

class CodingAttempt(BaseModel):
    """Inputs and outputs for a single attempt at generating and executing code."""
    plan: str = Field(description="The detailed numbered plan string")
    code: str = Field(description="Generated Python code for execution")
    analysis: Optional[str] = Field(description="Analysis of code, errors, and suggestions for fixes", default=None)
    execution_output: List[Dict] = Field(description="Output from code execution", default_factory=list)
    errors: List[str] = Field(description="Errors encountered during code generation or execution", default_factory=list)

class AgentState(BaseModel):
    """Agent state schema."""
    messages: Annotated[List[BaseMessage], operator.add] = Field(description="Conversation and tool messages", default_factory=list)
    context: Optional[Context] = Field(description="Info giving context to query", default=None)
    coding_attempts: List[CodingAttempt] = Field(description="History of coding attempts", default_factory=list)
    suggestions: Annotated[List[Suggestion], operator.add] = Field(description="Follow-on queries suggested to user", default_factory=list)
    
    def get_current_coding_attempt(self) -> Optional[CodingAttempt]:
        """
        Get the current (most recent) coding attempt.
        
        Returns:
            The current coding attempt, or None if no attempts exist
        """
        if not self.coding_attempts:
            return None
        return self.coding_attempts[-1]
    
    class Config:
        arbitrary_types_allowed = True


class ContextState(BaseModel):
    """State passed through the context graph for orchestrating agents."""
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    context: Annotated[Dict[str, Any], operator.or_] = Field(default_factory=dict)
    clarification_request: str = ""
    instruments_validated: Annotated[bool, operator.or_] = False
    db_context_provided: Annotated[bool, operator.or_] = False
    period_deduced: Annotated[bool, operator.or_] = False