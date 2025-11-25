"""Review level wrapper agents.

Implements four agents wrapping the existing review level tools:
 - GetReviewStatusByValueTool  -> review_by_value_agent
 - GetReviewStatusByTimeTool   -> review_by_time_agent
 - GetReviewSchemaTool         -> review_schema_agent
 - GetBreachedInstrumentsTool  -> breach_instr_agent

Pattern follows the proven sandbox agents (extraction_sandbox_agent, timeseries_plot_sandbox_agent):
 - Natural language prompt converted to populated input schema JSON via LLM
 - Errors fed back for iterative refinement (max 5 attempts)
 - StateGraph with generate_inputs -> call_tool -> decide_next -> parse_result
 - Returns same output as underlying tool (_run result)
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

from tools.review_level_toolkit import (
    GetReviewStatusByValueTool,
    GetReviewStatusByTimeTool,
    GetReviewSchemaTool,
    GetBreachedInstrumentsTool,
)
from tools.sql_security_toolkit import GeneralSQLQueryTool

logger = logging.getLogger(__name__)

class _GenericReviewAgentState(TypedDict, total=False):
    prompt: str
    attempt_count: int
    previous_errors: List[str]
    previous_failed_inputs: List[Dict[str, Any]]
    messages: List[AIMessage]
    tool_inputs: Dict[str, Any]
    tool_result: Any
    next_path: Optional[str]

_JSON_STRIP_RE = re.compile(r"^```json\n|```$", re.IGNORECASE)


def _clean_json_blocks(text: str) -> str:
    """Remove markdown fences if present."""
    return _JSON_STRIP_RE.sub("", text).strip()


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _should_retry(result: Any) -> bool:
    """Decide if we should retry based on result."""
    if isinstance(result, str):
        for line in result.splitlines():
            if line.strip().startswith("ERROR:"):
                return True
    return False


def _parse_consistency_result(raw: str) -> Dict[str, Any]:
    """Parse the LLM consistency checker output.
    Implements robust parsing rules:
        - Trim whitespace.
        - If any line (case-sensitive) startswith 'ERROR:' treat as error and
            use the FIRST such line as the canonical message.
    - Else if the first or last non-empty line equals exactly 'OK' treat as success.
        - Otherwise treat the whole output as an error (unexpected format) so the agent will regenerate, prefixing with 'ERROR:' for consistency.
    Returns dict: {'is_error': bool, 'message': str}
    """
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return {"is_error": True, "message": "ERROR: empty consistency output"}
    for l in lines:
        if l.startswith("ERROR:"):
            return {"is_error": True, "message": l}
    first = lines[0]
    if first == "OK" or lines[-1] == "OK":
        return {"is_error": False, "message": "OK"}
    return {"is_error": True, "message": f"ERROR: unexpected consistency format -> {first}"}


def _build_generation_prompt(
    required_fields: List[str],
    instructions: str,
    tool_name: str,
    optional_fields: Optional[List[str]] = None,
) -> ChatPromptTemplate:
    """Create an enriched prompt template (self-correcting) guiding LLM to output ONLY JSON.

    Incorporates role, schema description, current date, previous failed inputs & errors, and explicit task list
    mirroring the proven pattern from the time series plot sandbox agent.
    """
    optional_fields = optional_fields or []
    all_fields_list = "\n".join(f"- {f}" for f in required_fields + optional_fields)
    required_fields_list = "\n".join(f"- {f}" for f in required_fields)
    schema_block = (
        "Tool input schema (output EXACTLY these keys; JSON object):\n"
        f"Required keys:\n{required_fields_list}\n"
        + (f"Optional keys (set null if absent):\n" + "\n".join(f"- {f}" for f in optional_fields) if optional_fields else "")
    ).strip()
    template = f"""
You are an expert at formulating structured JSON inputs for the {tool_name}. Produce ONLY valid JSON.

Current date/time (UTC ISO8601): {{now}}

User natural language prompt:
{{prompt}}

Previous failed inputs (if 'None', no previous failure):
{{failed_inputs}}

Previous errors (if 'None', no previous error; correct issues while preserving correct parts):
{{errors}}

{schema_block}

Additional guidance:
{instructions}

Task steps:
1. Parse the prompt to extract each required field. Infer optional fields if present; else set them to null.
2. Validate types: numeric values must be numbers (no units); timestamps must be ISO8601 string (e.g. 2025-02-10T00:00:00Z). If natural language time range given, choose the END timestamp.
3. Ensure required fields are present. If any required element is ambiguous, make a single reasonable assumption and proceed.
4. If previous failed inputs/errors are provided, adjust ONLY the incorrect parts; retain previously correct values.
5. Do NOT invent fields not in the schema. Do NOT output commentary.
6. Output ONLY a single minified JSON object with keys EXACTLY as in schema (no markdown fences, no trailing text). Optional keys must be present with either a value or null.

Output format: strictly one JSON object. No backticks. No explanations.
"""
    return ChatPromptTemplate.from_template(template)


def _build_consistency_check_prompt(
        tool_name: str,
        tool_description: str,
        required_fields: List[str],
        optional_fields: Optional[List[str]] = None,
) -> ChatPromptTemplate:
        """Create a prompt template for factual consistency verification.

        The LLM receives: original natural language prompt and the structured tool_inputs JSON.
        It must attempt to *reverse-engineer* (back-analyse) a concise natural language description
        of the intent implied by the JSON, then compare every factual element with the original
        prompt (instrument IDs, review level names, field names, numeric values, timestamps, types).

        Output rules:
            - If ALL facts are consistent: output only 'OK'.
            - If ANY mismatch or ambiguity that would change meaning: output a single line starting
                with 'ERROR:' followed by a clear explanation of each discrepancy (do not propose fixes).
            - Do not output anything else including your thought process.
            - No markdown, no JSON, no additional prose beyond one line.
        """
        optional_fields = optional_fields or []
        required_list = ", ".join(required_fields) if required_fields else "(none)"
        optional_list = ", ".join(optional_fields) if optional_fields else "(none)"
        template = f"""
You are validating structured inputs for the {tool_name}.

Tool function summary:
{tool_description}

Schema (required -> {required_list}; optional -> {optional_list}).

Original user prompt:
{{original_prompt}}

Generated structured inputs JSON:
{{tool_inputs_json}}

Task:
1. Infer a minimal natural language description implied solely by the JSON values.
2. Compare EVERY factual element to the original prompt.
3. If all facts (IDs, names, numeric values, timestamps, types, review levels, directions) match or are consistent (including reasonable normalization like case or formatting), output 'OK'.
4. Otherwise output one line starting with 'ERROR:' listing each differing field with both values. Do not include suggestions.

Output strictly one line: either 'OK' or 'ERROR: <explanation>'.
"""
        return ChatPromptTemplate.from_template(template)


def _invoke_llm(chain, prompt: str, previous_errors: List[str], previous_failed_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    inputs = {
        "now": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "errors": "\n".join(previous_errors) if previous_errors else "None",
        "failed_inputs": json.dumps(previous_failed_inputs) if previous_failed_inputs else "None",
    }
    raw = chain.invoke(inputs).strip()
    raw = _clean_json_blocks(raw)
    data = _safe_json_loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"LLM did not return valid JSON object: {raw[:200]}")
    return data


# ---------------------------------------------------------------------------
# ReviewByValue Agent
# ---------------------------------------------------------------------------

class ReviewByValueAgentInput(BaseModel):
    prompt: str = Field(
        ..., 
        description=(
            "Natural language prompt to determine the review status (most severe breached review level) of a specified reading value recorded in a specified database field at a specified instrument. Prompt must include:\n"
            "- Instrument ID (instrum.instr_id), e.g. 'INST001'.\n"
            "- Database field name exactly as stored (e.g. 'data1', 'data5', 'calculation2').\n"
            "- The numeric value to test (e.g. 12.7).\n"
            "Example: 'For instrument INST045 check if the value 12.7 on data3 breaches any review levels.'"
        )
    )


from typing import ClassVar

class BaseGenericReviewAgentTool(BaseTool):
    """Generic review level agent base encapsulating shared graph + retry logic."""
    llm: BaseLanguageModel = Field(...)
    sql_tool: GeneralSQLQueryTool = Field(...)
    underlying_tool_cls: ClassVar[Type[Any]] = None
    required_fields: ClassVar[List[str]] = []
    optional_fields: ClassVar[List[str]] = []
    instructions: ClassVar[str] = ""
    param_map: ClassVar[Dict[str, str]] = {}
    max_attempts: ClassVar[int] = 5

    def post_process_inputs(self, tool_inputs: Dict[str, Any]) -> Dict[str, Any]:
        return tool_inputs

    def _run_generic(self, prompt: str, agent_label: str) -> Any:
        underlying = self.underlying_tool_cls(sql_tool=self.sql_tool)
        generate_chain = (
            _build_generation_prompt(
                self.required_fields,
                self.instructions,
                tool_name=agent_label,
                optional_fields=self.optional_fields,
            ) | self.llm | StrOutputParser()
        )
        consistency_chain = (
            _build_consistency_check_prompt(
                tool_name=agent_label,
                tool_description=self.description,
                required_fields=self.required_fields,
                optional_fields=self.optional_fields,
            ) | self.llm | StrOutputParser()
        )

        def generate_inputs(state: _GenericReviewAgentState) -> _GenericReviewAgentState:
            messages = state.get("messages", [])
            messages.append(AIMessage(name=agent_label, content="Generating structured inputs...", additional_kwargs={"stage": "intermediate", "process": "action"}))
            previous_errors = state.get("previous_errors", [])
            try:
                tool_inputs = _invoke_llm(generate_chain, state["prompt"], previous_errors, state.get("previous_failed_inputs", []))
                tool_inputs = self.post_process_inputs(tool_inputs)
                messages.append(AIMessage(name=agent_label, content=f"Generated inputs: {tool_inputs}", additional_kwargs={"stage": "intermediate", "process": "observation"}))
                return {"tool_inputs": tool_inputs, "messages": messages}
            except Exception as e:
                err = f"Generation failed: {e}"; previous_errors.append(err)
                messages.append(AIMessage(name=agent_label, content=err, additional_kwargs={"stage": "intermediate", "process": "observation"}))
                return {"previous_errors": previous_errors, "messages": messages}

        def check_missing_fields(state: _GenericReviewAgentState) -> _GenericReviewAgentState:
            messages = state.get("messages", [])
            tool_inputs = state.get("tool_inputs", {})
            messages.append(AIMessage(name=agent_label, content="Checking for missing required fields...", additional_kwargs={"stage": "intermediate", "process": "action"}))
            missing = [f for f in self.required_fields if f not in tool_inputs]
            if missing:
                err = f"ERROR: missing required fields in generated inputs: {', '.join(missing)}"
                previous_errors = state.get("previous_errors", [])
                previous_failed_inputs = state.get("previous_failed_inputs", [])
                previous_errors.append(err)
                if tool_inputs:
                    previous_failed_inputs.append(tool_inputs)
                messages.append(AIMessage(name=agent_label, content=err, additional_kwargs={"stage": "intermediate", "process": "observation"}))
                return {"previous_errors": previous_errors, "previous_failed_inputs": previous_failed_inputs, "next_path": "generate_inputs", "messages": messages}
            messages.append(AIMessage(name=agent_label, content="All required fields present.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            return {"next_path": "check_consistent_inputs", "messages": messages}

        def check_consistent_inputs(state: _GenericReviewAgentState) -> _GenericReviewAgentState:
            messages = state.get("messages", [])
            tool_inputs = state.get("tool_inputs", {})
            messages.append(AIMessage(name=agent_label, content="Validating factual consistency...", additional_kwargs={"stage": "intermediate", "process": "action"}))
            previous_errors = state.get("previous_errors", [])
            previous_failed_inputs = state.get("previous_failed_inputs", [])
            if not tool_inputs:
                err = "ERROR: no tool_inputs available for consistency check"
                previous_errors.append(err)
                messages.append(AIMessage(name=agent_label, content=err, additional_kwargs={"stage": "intermediate", "process": "observation"}))
                return {"previous_errors": previous_errors, "next_path": "generate_inputs", "messages": messages}
            try:
                consistency_result = consistency_chain.invoke({
                    "original_prompt": state.get("prompt", ""),
                    "tool_inputs_json": json.dumps(tool_inputs, ensure_ascii=False),
                }).strip()
            except Exception as e:
                consistency_result = f"ERROR: exception during consistency check: {e}"
            messages.append(AIMessage(name=agent_label, content=f"Consistency result: {consistency_result}", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            parsed = _parse_consistency_result(consistency_result)
            if parsed["is_error"]:
                previous_errors.append(parsed["message"])
                previous_failed_inputs.append(tool_inputs)
                messages.append(AIMessage(name=agent_label, content=f"Consistency parsed as error -> {parsed['message']}", additional_kwargs={"stage": "intermediate", "process": "observation"}))
                return {"previous_errors": previous_errors, "previous_failed_inputs": previous_failed_inputs, "next_path": "generate_inputs", "messages": messages}
            messages.append(AIMessage(name=agent_label, content="Consistency OK.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            return {"next_path": "call_tool", "messages": messages}

        def call_tool(state: _GenericReviewAgentState) -> _GenericReviewAgentState:
            messages = state.get("messages", [])
            tool_inputs = state.get("tool_inputs", {})
            messages.append(AIMessage(name=agent_label, content="Executing underlying tool...", additional_kwargs={"stage": "intermediate", "process": "action"}))
            attempt = state.get("attempt_count", 0) + 1
            kwargs_run = {under_arg: tool_inputs.get(json_key) for json_key, under_arg in self.param_map.items()}
            try:
                result = underlying._run(**kwargs_run)
            except Exception as e:
                result = f"ERROR: exception during tool run: {e}"
            messages.append(AIMessage(name=agent_label, content=f"Tool result: {result}", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            previous_errors = state.get("previous_errors", [])
            previous_failed_inputs = state.get("previous_failed_inputs", [])
            next_path = "parse_result"
            if _should_retry(result) and attempt < self.max_attempts:
                previous_errors.append(str(result))
                if tool_inputs:
                    previous_failed_inputs.append(tool_inputs)
                messages.append(AIMessage(name=agent_label, content="Retrying due to tool error...", additional_kwargs={"stage": "intermediate", "process": "action"}))
                next_path = "generate_inputs"
            return {"tool_result": result, "attempt_count": attempt, "previous_errors": previous_errors, "previous_failed_inputs": previous_failed_inputs, "next_path": next_path, "messages": messages}

        def parse_result(state: _GenericReviewAgentState) -> _GenericReviewAgentState:
            messages = state.get("messages", [])
            result = state.get("tool_result")
            messages.append(AIMessage(name=agent_label, content="Finished.", additional_kwargs={"stage": "final", "process": "observation"}))
            return {"tool_result": result, "messages": messages}

        graph = StateGraph(_GenericReviewAgentState)
        graph.add_node("generate_inputs", generate_inputs)
        graph.add_node("check_missing_fields", check_missing_fields)
        graph.add_node("check_consistent_inputs", check_consistent_inputs)
        graph.add_node("call_tool", call_tool)
        graph.add_node("parse_result", parse_result)
        graph.set_entry_point("generate_inputs")
        graph.add_edge("generate_inputs", "check_missing_fields")
        graph.add_conditional_edges(
            "check_missing_fields",
            lambda s: s.get("next_path", "check_consistent_inputs"),
            {"generate_inputs": "generate_inputs", "check_consistent_inputs": "check_consistent_inputs"},
        )
        graph.add_conditional_edges(
            "check_consistent_inputs",
            lambda s: s.get("next_path", "call_tool"),
            {"generate_inputs": "generate_inputs", "call_tool": "call_tool"},
        )
        graph.add_conditional_edges(
            "call_tool",
            lambda s: s.get("next_path", "parse_result"),
            {"generate_inputs": "generate_inputs", "parse_result": "parse_result"},
        )
        graph.add_edge("parse_result", END)
        compiled = graph.compile()
        initial_state: _GenericReviewAgentState = {
            "prompt": prompt,
            "attempt_count": 0,
            "previous_errors": [],
            "previous_failed_inputs": [],
            "messages": [AIMessage(name=agent_label, content="Starting agent run.", additional_kwargs={"stage": "intermediate", "process": "action"})],
        }
        final_state = compiled.invoke(initial_state)
        return final_state.get("tool_result")

    def invoke(self, input=None, **kwargs):
        if input is None and "prompt" in kwargs:
            input = kwargs.pop("prompt")
        if isinstance(input, dict) and set(input.keys()) == {"prompt"}:
            input = input["prompt"]
        return super().invoke(input, **kwargs)


class ReviewByValueAgentTool(BaseGenericReviewAgentTool):
    name: str = "review_by_value_agent"
    description: str = (
        "Determines the review status (most severe breached review level) of a specified reading value recorded in a specified database field at a specified instrument described in a natural language prompt.\n"
        "Use when you already know a measurement value and need to know which review level it breaches.\n"
        "Prompt must include:\n"
        "- Instrument ID (instrum.instr_id), e.g. 'INST001'.\n"
        "- Database field name exactly as stored (e.g. 'data1', 'data5', 'calculation2').\n"
        "- The numeric value to test (e.g. 12.7).\n"
        "Example: 'For instrument INST045 check if the value 12.7 on data3 breaches any review levels.'\n"
        "Returns: breached review level name (str) or None if no breach or ERROR: message."
    )
    args_schema: Type[BaseModel] = ReviewByValueAgentInput
    underlying_tool_cls: ClassVar[Type[Any]] = GetReviewStatusByValueTool
    required_fields: ClassVar[List[str]] = ["instrument_id", "db_field_name", "db_field_value"]
    optional_fields: ClassVar[List[str]] = []
    instructions: ClassVar[str] = (
        "Extract the instrument ID, database field name (e.g., data1, calculation1) and numeric field value. If numeric value includes units, strip units."
    )
    param_map: ClassVar[Dict[str, str]] = {
        "instrument_id": "instrument_id",
        "db_field_name": "db_field_name",
        "db_field_value": "db_field_value",
    }

    def _run(self, prompt: str) -> Union[str, None]:
        return self._run_generic(prompt, agent_label="ReviewByValueAgent")


def review_by_value_agent(llm: BaseLanguageModel, db, table_relationship_graph, user_id: int, global_hierarchy_access: bool) -> BaseTool:
    sql_tool = GeneralSQLQueryTool(db=db, table_relationship_graph=table_relationship_graph, user_id=user_id, global_hierarchy_access=global_hierarchy_access)
    return ReviewByValueAgentTool(llm=llm, sql_tool=sql_tool)


# ---------------------------------------------------------------------------
# ReviewByTime Agent
# ---------------------------------------------------------------------------

class ReviewByTimeAgentInput(BaseModel):
    prompt: str = Field(
        ..., 
        description=(
            "Natural language prompt to fetch the breached review status ofthe most recent value BEFORE a timestamp along with the value itself and when it was recorded. Prompt must include:\n"
            "- Instrument ID (instrum.instr_id).\n"
            "- Database field name (dataN or calculationN).\n"
            "- Reference timestamp (ISO8601 or unambiguous natural language that can map to ISO).\n"
            "Example: 'At 2025-01-15T10:30:00Z what is the review status for data4 on instrument INST220?'."
        )
    )

class ReviewByTimeAgentTool(BaseGenericReviewAgentTool):
    name: str = "review_by_time_agent"
    description: str = (
        "Determines the review status of the latest reading before a given timestamp.\n"
        "Use to get the historical or current breach status at a specific time.\n"
        "Prompt MUST contain: instrument ID, database field name, target timestamp (ISO8601 or clearly parseable).\n"
        "Returns: DataFrame with columns (review_status, db_field_value, db_field_value_timestamp) or None or ERROR: message."
    )
    args_schema: Type[BaseModel] = ReviewByTimeAgentInput
    underlying_tool_cls: ClassVar[Type[Any]] = GetReviewStatusByTimeTool
    required_fields: ClassVar[List[str]] = ["instrument_id", "db_field_name", "timestamp"]
    optional_fields: ClassVar[List[str]] = []
    instructions: ClassVar[str] = (
        "Extract instrument_id, db_field_name, and timestamp (ISO8601). If user supplies natural language time range ending at a time, choose the end timestamp."
    )
    param_map: ClassVar[Dict[str, str]] = {
        "instrument_id": "instrument_id",
        "db_field_name": "db_field_name",
        "timestamp": "timestamp",
    }

    def _run(self, prompt: str) -> Union[pd.DataFrame, None, str]:
        return self._run_generic(prompt, agent_label="ReviewByTimeAgent")


def review_by_time_agent(llm: BaseLanguageModel, db, table_relationship_graph, user_id: int, global_hierarchy_access: bool) -> BaseTool:
    sql_tool = GeneralSQLQueryTool(db=db, table_relationship_graph=table_relationship_graph, user_id=user_id, global_hierarchy_access=global_hierarchy_access)
    return ReviewByTimeAgentTool(llm=llm, sql_tool=sql_tool)


# ---------------------------------------------------------------------------
# ReviewSchema Agent
# ---------------------------------------------------------------------------

class ReviewSchemaAgentInput(BaseModel):
    prompt: str = Field(
        ..., 
        description=(
            "Natural language prompt to list all active review levels (thresholds) for a specified instrument and specified database field. Prompt must include:\n"
            "- Instrument ID (instrum.instr_id).\n"
            "- Database field name (dataN or calculationN).\n"
            "Example: 'List all review levels for calculation3 on instrument INST088.'"
        )
    )


class ReviewSchemaAgentTool(BaseGenericReviewAgentTool):
    name: str = "review_schema_agent"
    description: str = (
        "Returns full schema (names, values, direction, color) of active review levels for a specific instrument field.\n"
        "Use to get thresholds to interpret or visualize current measurement values.\n"
        "Prompt MUST contain: instrument ID and database field name.\n"
        "Returns: DataFrame (review_name, review_value, review_direction, review_color) or None or ERROR: message."
    )
    args_schema: Type[BaseModel] = ReviewSchemaAgentInput
    underlying_tool_cls: ClassVar[Type[Any]] = GetReviewSchemaTool
    required_fields: ClassVar[List[str]] = ["instrument_id", "db_field_name"]
    optional_fields: ClassVar[List[str]] = []
    instructions: ClassVar[str] = "Extract the instrument_id and db_field_name whose review schema is requested."
    param_map: ClassVar[Dict[str, str]] = {
        "instrument_id": "instrument_id",
        "db_field_name": "db_field_name",
    }

    def _run(self, prompt: str) -> Union[pd.DataFrame, None, str]:
        return self._run_generic(prompt, agent_label="ReviewSchemaAgent")


def review_schema_agent(llm: BaseLanguageModel, db, table_relationship_graph, user_id: int, global_hierarchy_access: bool) -> BaseTool:
    sql_tool = GeneralSQLQueryTool(db=db, table_relationship_graph=table_relationship_graph, user_id=user_id, global_hierarchy_access=global_hierarchy_access)
    return ReviewSchemaAgentTool(llm=llm, sql_tool=sql_tool)


# ---------------------------------------------------------------------------
# Breached Instruments Agent
# ---------------------------------------------------------------------------

class BreachInstrAgentInput(BaseModel):
    prompt: str = Field(
        ..., 
        description=(
            "Natural language prompt to find instruments whose latest reading before a timestamp is at a specified review status (surpasses specified review level but not surpassing any more severe levels). Prompt must include:\n"
            "- Review level name to test (e.g. 'Alert', 'Warning').\n"
            "- Instrument type (instrum.type1).\n"
            "- Optional instrument subtype (instrum.type2).\n"
            "- Database field name (dataN or calculationN).\n"
            "- Timestamp cutoff (ISO8601 or parseable).\n"
            "Example: 'Which settlement instruments of type SETT with subtype DEEP breach the Warning level on calculation1 as of 2025-02-10T00:00:00Z?'"
        )
    )


class BreachInstrAgentTool(BaseGenericReviewAgentTool):
    name: str = "breach_instr_agent"
    description: str = (
        "Finds instruments whose latest reading before a timestamp is at a specified review status (surpasses specified review level but not surpassing any more severe levels).\n"
        "Use to get a filtered list of currently or historically breached instruments at a particular review status for reporting or alerting.\n"
        "Prompt MUST contain: review level name, instrument type, database field name, timestamp. Optional: instrument subtype.\n"
        "Returns: DataFrame rows (instrument_id, field_value, field_value_timestamp, review_value) or None or ERROR: message."
    )
    args_schema: Type[BaseModel] = BreachInstrAgentInput
    underlying_tool_cls: ClassVar[Type[Any]] = GetBreachedInstrumentsTool
    required_fields: ClassVar[List[str]] = ["review_name", "instrument_type", "db_field_name", "timestamp"]
    optional_fields: ClassVar[List[str]] = ["instrument_subtype"]
    instructions: ClassVar[str] = (
        "Extract review_name (the level to test), instrument_type, optional instrument_subtype (set null if not specified), db_field_name and timestamp (ISO8601)."
    )
    param_map: ClassVar[Dict[str, str]] = {
        "review_name": "review_name",
        "instrument_type": "instrument_type",
        "instrument_subtype": "instrument_subtype",
        "db_field_name": "db_field_name",
        "timestamp": "timestamp",
    }

    def post_process_inputs(self, tool_inputs: Dict[str, Any]) -> Dict[str, Any]:
        if tool_inputs.get("instrument_subtype") in ("", "null", None):
            tool_inputs["instrument_subtype"] = None
        return tool_inputs

    def _run(self, prompt: str) -> Union[pd.DataFrame, None, str]:
        return self._run_generic(prompt, agent_label="BreachInstrAgent")


def breach_instr_agent(llm: BaseLanguageModel, db, table_relationship_graph, user_id: int, global_hierarchy_access: bool) -> BaseTool:
    sql_tool = GeneralSQLQueryTool(db=db, table_relationship_graph=table_relationship_graph, user_id=user_id, global_hierarchy_access=global_hierarchy_access)
    return BreachInstrAgentTool(llm=llm, sql_tool=sql_tool)


def review_level_agent_factory(kind: str, llm: BaseLanguageModel, db, table_relationship_graph, user_id: int, global_hierarchy_access: bool) -> BaseTool:
    """Unified factory returning any review level agent by short kind string.

    Accepted kinds: value, time, schema, breach (case-insensitive).
    """
    kind_norm = kind.lower().strip()
    mapping = {
        "value": review_by_value_agent,
        "time": review_by_time_agent,
        "schema": review_schema_agent,
        "breach": breach_instr_agent,
    }
    if kind_norm not in mapping:
        raise ValueError(f"Unknown review level agent kind: {kind}. Expected one of {list(mapping.keys())}.")
    return mapping[kind_norm](llm=llm, db=db, table_relationship_graph=table_relationship_graph, user_id=user_id, global_hierarchy_access=global_hierarchy_access)


__all__ = [
    "review_by_value_agent",
    "review_by_time_agent",
    "review_schema_agent",
    "breach_instr_agent",
    "ReviewByValueAgentTool",
    "ReviewByTimeAgentTool",
    "ReviewSchemaAgentTool",
    "BreachInstrAgentTool",
    "BaseGenericReviewAgentTool",
    "review_level_agent_factory",
]
