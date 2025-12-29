import json
import logging
from typing import Dict, Any, List, Union
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel
import pandas as pd

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict, Annotated
import operator

from classes import Execution, Context

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    steps: Annotated[int, operator.add]


def tool_calling_agent(
    llm: BaseLanguageModel,
    tools: List[Any],
    context: Context,
    previous_attempts: List[Execution]
) -> Dict[str, Any]:
    """
    Tool-calling agent using LangGraph for parallel tool execution.
    """
    retrospective_query = context.retrospective_query if context else ""
    word_context_json = json.dumps(
        [qw.model_dump() for qw in context.word_context] if context and context.word_context else [],
        indent=2
    )
    verified_instrument_info_json = json.dumps(context.verif_ID_info or {}, indent=2)
    validated_type_info_json = json.dumps(context.verif_type_info or [], indent=2)
    relevant_date_ranges_json = json.dumps(
        [r.model_dump() for r in (context.relevant_date_ranges or [])], indent=2
    )
    platform_context = context.platform_context or ""
    project_specific_context = context.project_specific_context or ""

    current_date = datetime.now().strftime('%B %d, %Y')

    previous_attempts_summary = "No previous failed attempts."
    if previous_attempts:
        previous_attempts_summary = "Summary of previous failed attempts:\n"
        for i, attempt in enumerate(previous_attempts):
            if attempt.error_summary:
                previous_attempts_summary += f"Attempt {i+1} failed.\n"
                previous_attempts_summary += f"Error: {attempt.error_summary}\n\n"

    react_system_prompt = f"""
# Role
You are an expert in answering queries on instrumentation monitoring data via querying a database.

# Context
- Current date: {current_date}
- Validated instrument types and subtypes referenced in the query (complete list from database; do not infer additional types or subtypes):
{validated_type_info_json}
- Validated instrument IDs with their type and subtype mappings (only rely on these ID-type pairs confirmed in the database):
{verified_instrument_info_json}
- Background behind words in query (instrument types, subtypes, DB fields, labels, units):
{word_context_json}
- Date ranges relevant to query and how to apply:
{relevant_date_ranges_json}
- Platform-specific terminology and semantics:
{platform_context}
- Additional context on database:
{project_specific_context}
- Summary of previous failed attempts:
{previous_attempts_summary}

# Available Tools
## `extraction_sandbox_agent`
### How to Use
- Use to extract data from database strictly following `extraction_sandbox_agent.invoke(prompt)` or in async code `await ainvoke(extraction_sandbox_agent, prompt)`.
- Prompt is natural language description of data to extract.
- Specify as much detail as possible in prompt with key:value pairs to minimize misinterpretation.
- Always include `Output columns` in prompt to specify DataFrame column names.
- Returns `pandas.DataFrame` or `None`.
- Combine data extraction steps when possible to reduce latency.
### Example Prompt
"Extract readings:
- Database field name: calculation1
- Database field type: calc
- Label: Settlement
- Unit: mm
- Instrument IDs: 0003-L-1, 0003-L-2
- Instrument type: LP
- Instrument subtype: MOVEMENT
- Time range: 1 January 2025 12:00:00 PM to 31 January 2025 11:59:59 PM
- Order: by settlement ascending
- Filter: settlement < -10 mm
- Output columns: [Timestamp, Settlement (mm)]"

## `timeseries_plot_sandbox_agent`
### How to Use
- Use to display change of data with time for one or more series strictly following `timeseries_plot_sandbox_agent.invoke(prompt)` (or `await ainvoke(timeseries_plot_sandbox_agent, prompt)` if async).
- Prompt is natural language description of plot which MUST include:
  * Instrument IDs for each series
  * Database field names for extracting data
  * Time range
  * Axis label (get from background to query)
  * Axis unit (get from background to query)
- You can also specify:
  * Secondary y-axis with title and unit
  * Whether to highlight y-axis zero line
- Returns artefact ID to access plot in file system or `None`.
- DO NOT use `extraction_sandbox_agent` to extract data for plotting because `timeseries_plot_sandbox_agent` extracts the data it needs.
### Example Prompt
"Plot temperature with time for instrument 0001-L-1 along with settlement with time for instruments 0003-L-1 and 0003-L-2:
- Time range: 1 January 2025 12:00:00 PM to 31 January 2025 11:59:59 PM
- Series 1 on primary axis:
  * Instrument ID: 0001-L-1
  * Database field name: data1
  * Database field type: data
  * Axis label: Temperature
  * Axis unit: °C
- Series 2 on secondary axis:
  * Instrument IDs: 0003-L-1, 0003-L-2
  * Database field name: calculation1
  * Database field type: calc
  * Axis label: Settlement
  * Axis unit: (mm)

## `map_plot_sandbox_agent`
### How to Use
- Use to display spatial distribution of readings or review status strictly following `map_plot_sandbox_agent.invoke(prompt)` (or `await ainvoke(map_plot_sandbox_agent, prompt)` if async).
- Readings or review status can be plotted as at single time or as change over period.
- Can plot multiple series with different instrument types, subtypes and database fields.
- Prompt is natural language description of plot which MUST include:
  * Whether plotting readings or review status
  * Whether plotting at single time or change over period
  * Time if single time or time range if change over period
  * Buffer period to look for missing readings (>= 1 day, get from date ranges relevant to query)
  * For each series:
    + Instrument type
    + Instrument subtype
    + Database field name
    + Label (get from background to query)
    + Unit (get from background to query)
  * Centre of plot either as instrument ID or easting and northing
  * Extent of plot as radius in metres
  * Any specific instrument IDs to exclude from plot e.g. if large values would distort colour scale
- Returns artefact ID to access plot in file system or `None`.
- DO NOT use `extraction_sandbox_agent` to extract data for plotting because `map_plot_sandbox_agent` extracts the data it needs.

### Example Prompt 1
"Plot change of readings over period as a map:
- Time range: 1 January 2025 12:00:00 PM to 31 January 2025 11:59:59 PM
- Buffer for missing readings: 3 days
- Plot centred on instrument ID 0002-P-1
- Plot extent: 200 metres radius
- Series 1:
  * Instrument type: LP
  * Instrument subtype: MOVEMENT
  * Database field name: calculation1
  * Label: Settlement
  * Unit: mm
- Series 2:
  * Instrument type: VWP
  * Instrument subtype: DEFAULT
  * Database field name: data2
  * Label: Groundwater level
  * Unit: mPD
- Instrument IDs to exclude: 0003-L-2"

### Example Prompt 2
"Plot change in review status over period as a map:
- Time range: 14 May 2025 12:00:00 PM to 14 November 2025 11:59:59 PM
- Plot centred on instrument ID 0003-L-2
- Plot extent: 300 metres radius
- Series:
  * Instrument type: LP
  * Instrument subtype: MOVEMENT
  * Database field name: data1
  * Label: Settlement
  * Unit: mm"

## `review_by_value_agent`
### How to Use
- Use to get review status for one or more known measurement values strictly following `review_by_value_agent.invoke(prompt)` or in async code `await ainvoke(review_by_value_agent, prompt)`.
- Prompt is natural language description of review status request that MUST include:
  * Instrument IDs
  * Database field names
  * Database field values
- Returns breached review level name (str) or `None` or `ERROR: <error message>`.
### Example Prompt
"Find review status:
Status Query 1
- Instrument ID: INST045
- Database field name: data3
- Database field value: 15.7
Status Query 2
- Instrument ID: INST047
- Database field name: data2
- Database field value: 12.7"

## `review_by_time_agent`
### How to Use
- Use to get review status of the latest reading before a given timestamp at one or more instruments and data fields strictly following `review_by_time_agent.invoke(prompt)` or in async code `await ainvoke(review_by_time_agent, prompt)`.
- Prompt is natural language description of review status request that MUST include:
  * Instrument IDs
  * Database field names
  * Timestamps
- Returns `pandas.DataFrame` with columns (`review_status`, `db_field_value`, `db_field_value_timestamp`) or `None` or `ERROR: <error message>`.
### Example Prompt
"Find review status for:
Status Query 1
- Instrument ID: INST045
- Database field name: data3
- Timestamp: 14 May 2025 12:00:00 PM
Status Query 2
- Instrument ID: INST046
- Database field name: data1
- Timestamp: 14 Jun 2025 09:00:00 PM"

## `review_schema_agent`
### How to Use
- Use to get full schema (names, values, direction, color) of active review levels for one or more instrument fields strictly following `review_schema_agent.invoke(prompt)` or in async code `await ainvoke(review_schema_agent, prompt)`.
- Prompt is natural language description of review schema request that MUST include:
  * Instrument IDs
  * Database field names
- Returns `pandas.DataFrame` with columns (`review_name`, `review_value`, `review_direction`, `review_color`) or `None` or `ERROR: <error message>`.
### Example Prompt
"List review levels for:
Schema Query 1
- Instrument ID: INST088
- Database field name: calculation2
Schema Query 2
- Instrument ID: INST087
- Database field name: calculation4"

## `breach_instr_agent`
### How to Use
- Use to get instruments whose latest reading before a timestamp surpasses any review level strictly following `breach_instr_agent.invoke(prompt)` or in async code `await ainvoke(breach_instr_agent, prompt)`.
- Prompt is natural language description of review status enquiry that MUST include:
  * Timestamp cut-off
- For more specific requests, you can include the following in the prompt:
  * Review level name to only return breaches at that level
  * Instrument type
  * Instrument subtype
  * Database field name
- Returns `pandas.DataFrame` with columns (instrument_id, db_field_name, review_name, field_value, field_value_timestamp, review_value) or `None` or `ERROR: <error message>`.
### Example Prompt 1
"List breaches for:
- Review level name: ALERT
- Instrument type: LP
- Instrument subtype: MOVEMENT
- Database field name: calculation1
- Timestamp: 14 May 2025 12:00:00 PM"
### Example Prompt 2
"List breaches for:
- Timestamp: 14 Jun 2025 09:00:00 PM"

# Instructions
1. Analyse the user query to understand what is being asked.
2. Deduce the user's underlying need.
3. Plan how to answer the query.
4. Use tools to extract and/or plot data. You must call multiple tools in parallel if their executions are independent (e.g., extracting data for different sets of instruments, or creating different plots). This reduces latency.
5. After receiving tool outputs, parse JSON from `extraction_sandbox_agent` if needed.
6. Collect artefact IDs from plotting tools.
7. Provide a final answer that addresses the query, referencing the data and artefacts as appropriate.

When calling tools, provide detailed natural language prompts as specified in the tool descriptions (pass as the "prompt" argument).

If you have enough information after tool calls, provide the final answer directly in your response.

Begin by analyzing the query.
    """.strip()

    langchain_tools = []
    def _resolve_prompt(arg: Any) -> str:
        """Resolve a prompt string from various possible arg formats produced by LC tool routing."""
        if isinstance(arg, str):
            return arg
        if isinstance(arg, dict):
            for k in ["prompt", "__arg1", "input", "text"]:
                if k in arg and isinstance(arg[k], str):
                    return arg[k]
            for v in arg.values():
                if isinstance(v, str):
                    return v
        return str(arg)

    def make_extraction_wrapper(tool: Any):
        def _fn(arg: Any) -> str:
            prompt = _resolve_prompt(arg)
            logger.info("Invoking %s with resolved prompt: %s", tool.name, prompt)
            result = tool.invoke(prompt)
            logger.info("%s returned: %s", tool.name, result)
            if result is None:
                return "None"
            if not isinstance(result, pd.DataFrame):
                logger.warning("%s did not return a DataFrame", tool.name)
                return "None"
            return result.to_json(orient="records")
        return _fn

    def make_generic_wrapper(tool: Any):
        def _fn(arg: Any) -> str:
            prompt = _resolve_prompt(arg)
            logger.info("Invoking %s with resolved prompt: %s", tool.name, prompt)
            result = tool.invoke(prompt)
            logger.info("%s returned: %s", tool.name, result)
            return str(result) if result is not None else "None"
        return _fn

    for t in tools:
        if t.name in [
            "extraction_sandbox_agent",
            "review_by_time_agent",
            "review_schema_agent",
            "breach_instr_agent"
        ]:
            tool_obj = Tool.from_function(
                func=make_extraction_wrapper(t),
                name=t.name,
                description=t.description
            )
        else:
            tool_obj = Tool.from_function(
                func=make_generic_wrapper(t),
                name=t.name,
                description=t.description
            )
        langchain_tools.append(tool_obj)

    llm_with_tools = llm.bind_tools(langchain_tools)

    def agent_node(state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        result = llm_with_tools.invoke(messages)
        return {"messages": [result], "steps": 1}

    def should_continue(state: AgentState) -> str:
        if state["steps"] >= 5:
            return END
        return tools_condition(state)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    tool_node = ToolNode(langchain_tools)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    input_value = f"Answer the following user query plus an optional extension that adds value:\n{retrospective_query}"

    initial_state = {
        "messages": [SystemMessage(content=react_system_prompt), HumanMessage(content=input_value)],
        "steps": 0,
    }

    raw_messages: List[Union[HumanMessage, AIMessage, ToolMessage]] = []

    try:
        stream = graph.stream(initial_state, stream_mode="updates")

        for chunk in stream:
            logger.debug("Received chunk from LangGraph stream: %s", chunk)
            for node, update in chunk.items():
                if "messages" in update:
                    for msg in update["messages"]:
                        raw_messages.append(msg)

        def _extract_tool_call_fields(call: Any) -> Dict[str, Any]:
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("args", {})
                cid = call.get("id")
                if isinstance(args, dict):
                    arg1 = args.get("prompt", next(iter(args.values())) if args else "")
                else:
                    arg1 = str(args)
            else:
                name = getattr(call, "name", None)
                args = getattr(call, "args", None)
                cid = getattr(call, "id", None)
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        if isinstance(parsed, dict):
                            arg1 = parsed.get("__arg1", parsed.get("prompt", next(iter(parsed.values())) if parsed else ""))
                        else:
                            arg1 = str(parsed)
                    except Exception:
                        arg1 = args
                else:
                    arg1 = str(args) if args is not None else ""
            return {"name": name, "id": cid, "__arg1": arg1}

        tool_input_map: Dict[str, str] = {}
        for m in raw_messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    fields = _extract_tool_call_fields(tc)
                    if fields.get("id"):
                        tool_input_map[str(fields["id"])] = fields.get("__arg1")

        final_ai_msg: AIMessage | None = None
        for m in reversed(raw_messages):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                final_ai_msg = m
                break

        artefact_msgs: List[AIMessage] = []
        streamed_messages: List[AIMessage] = []
        final_msg_obj: AIMessage | None = None

        for m in raw_messages:
            if isinstance(m, AIMessage):
                if m is final_ai_msg:
                    final_msg_obj = AIMessage(
                        name="Tool-Calling",
                        content=m.content or "",
                        additional_kwargs={"stage": "execution_output", "process": "final"},
                    )
                    streamed_messages.append(final_msg_obj)
                else:
                    base_content = m.content or ""
                    tc_list = getattr(m, "tool_calls", None) or []
                    content = base_content
                    if tc_list:
                        invocations: List[str] = []
                        for tc in tc_list:
                            fields = _extract_tool_call_fields(tc)
                            tool_name = fields.get("name") or ""
                            tool_input = fields.get("__arg1") or ""
                            invocations.append(f"Invoking tool `{tool_name}` with input:\n" + "\n".join(["> " + line for line in tool_input.splitlines()]))
                        content = f"{base_content}\n" + "\n".join(invocations)
                    progress_msg = AIMessage(
                        name="Tool-Calling",
                        content=content,
                        additional_kwargs={"stage": "execution_output", "process": "progress"},
                    )
                    streamed_messages.append(progress_msg)
            elif isinstance(m, ToolMessage):
                tool_exec_msg = AIMessage(
                    name="Tool-Calling",
                    content=f"Tool `{m.name}` executed with output:\n" + "\n".join(["> " + line for line in m.content.splitlines()]),
                    additional_kwargs={"stage": "execution_output", "process": "progress"},
                )
                streamed_messages.append(tool_exec_msg)

                if m.name and "plot" in m.name:
                    plot_description = tool_input_map.get(getattr(m, "tool_call_id", ""))
                    if plot_description:
                        artefact_msgs.append(
                            AIMessage(
                                name="Tool-Calling",
                                content=plot_description,
                                additional_kwargs={
                                    "stage": "execution_output",
                                    "process": "plot",
                                    "artefact_id": m.content,
                                },
                            )
                        )

        return {
            "final_response": final_msg_obj,
            "artefacts": artefact_msgs,
            "messages": streamed_messages,
        }

    except Exception as e:
        logger.error("Error in react_agent: %s", str(e))
        error_msg = AIMessage(
            name="Tool-Calling",
            content=str(e),
            additional_kwargs={"stage": "execution_output", "process": "error"},
        )
        return {
            "final_response": None,
            "artefacts": [],
            "messages": [error_msg],
            "error_summary": str(e),
        }