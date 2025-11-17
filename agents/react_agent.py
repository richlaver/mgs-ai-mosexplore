import json
import logging
from typing import Dict, Any, List, Union
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel
import pandas as pd

from classes import Execution, Context

logger = logging.getLogger(__name__)


def react_agent(
    llm: BaseLanguageModel,
    tools: List[Any],
    context: Context,
    previous_attempts: List[Execution]
) -> Dict[str, Any]:
    """
    ReAct agent that answers queries using tools.
    Returns only final_response and artefacts for attaching to Execution.
    """
    retrospective_query = context.retrospective_query if context else ""
    word_context_json = json.dumps(
        [qw.model_dump() for qw in context.word_context] if context and context.word_context else [],
        indent=2
    )
    verified_instrument_info_json = json.dumps(context.verif_ID_info or {}, indent=2)
    relevant_date_ranges_json = json.dumps(
        [r.model_dump() for r in (context.relevant_date_ranges or [])], indent=2
    )

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
- Instrument IDs in query with their type and subtype:
{verified_instrument_info_json}
- Background behind words in query (instrument types, subtypes, DB fields, labels, units):
{word_context_json}
- Date ranges relevant to query and how to apply:
{relevant_date_ranges_json}
- Summary of previous failed attempts:
{previous_attempts_summary}

# Available Tools
## extraction_sandbox_agent
### How to Use
- You MUST use this tool if you need to extract data from the database.
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

## timeseries_plot_sandbox_agent
### How to Use
- You MUST use this tool to create time series plots.
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

## map_plot_sandbox_agent
### How to Use
- You MUST use this tool to create spatial distributions of readings or review statuses.
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

# Instructions
1. Analyse the user query to understand what is being asked.
2. Deduce the user's underlying need.
3. Think up an optional extension to the query that adds value to the user's need.
4. Plan how to answer the query and optional extension.
5. Use tools to extract and/or plot data.
6. Parse JSON from `extraction_sandbox_agent` if needed.
7. Collect artefact IDs from plotting tools.
8. Provide final answer.

Begin!
    """.strip()

    langchain_tools = []
    for t in tools:
        if t.name == "extraction_sandbox_agent":
            def extraction_func(prompt: str, tool=t) -> str:
                logger.info("Invoking %s with prompt: %s", tool.name, prompt)
                result = tool.invoke(prompt)
                logger.info("%s returned: %s", tool.name, result)
                if result is None:
                    return "None"
                if not isinstance(result, pd.DataFrame):
                    logger.warning("%s did not return a DataFrame", tool.name)
                    return "None"
                return result.to_json(orient="records")
            tool = Tool.from_function(
                func=extraction_func,
                name=t.name,
                description=t.description
            )
        else:
            def plot_func(prompt: str, tool=t) -> str:
                logger.info("Invoking %s with prompt: %s", tool.name, prompt)
                result = tool.invoke(prompt)
                logger.info("%s returned: %s", tool.name, result)
                return str(result) if result is not None else "None"
            tool = Tool.from_function(
                func=plot_func,
                name=t.name,
                description=t.description
            )
        langchain_tools.append(tool)

    agent = create_agent(
        llm,
        tools=langchain_tools,
        system_prompt=react_system_prompt,
        debug=True,
    )

    input_value = f"Answer the following user query plus an optional extension that adds value:\n{retrospective_query}"

    try:
        inputs = {"messages": [HumanMessage(content=input_value)]}
        stream = agent.stream(
            inputs,
            {"recursion_limit": 10},
            stream_mode="updates",
        )

        raw_messages: List[Union[HumanMessage, AIMessage, ToolMessage]] = []

        for chunk in stream:
            logger.debug("Received chunk from ReAct agent stream: %s", chunk)
            if "tools" in chunk:
                tools_output = chunk["tools"]
                if isinstance(tools_output, dict) and "messages" in tools_output and tools_output["messages"]:
                    tmsg = tools_output["messages"][-1]
                    if isinstance(tmsg, (ToolMessage, AIMessage)):
                        raw_messages.append(tmsg)
            if "model" in chunk:
                model_output = chunk["model"]
                if isinstance(model_output, dict) and "messages" in model_output and model_output["messages"]:
                    msg = model_output["messages"][-1]
                    if isinstance(msg, AIMessage):
                        raw_messages.append(msg)

        def _extract_tool_call_fields(call: Any) -> Dict[str, Any]:
            try:
                name = call.get("name")
            except AttributeError:
                name = getattr(call, "name", None)
            try:
                args = call.get("args")
            except AttributeError:
                args = getattr(call, "args", None)
            try:
                cid = call.get("id")
            except AttributeError:
                cid = getattr(call, "id", None)
            arg1 = None
            if isinstance(args, dict):
                arg1 = args.get("__arg1")
            elif isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        arg1 = parsed.get("__arg1")
                except Exception:
                    arg1 = None
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
                        name="ReAct",
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
                        name="ReAct",
                        content=content,
                        additional_kwargs={"stage": "execution_output", "process": "progress"},
                    )
                    streamed_messages.append(progress_msg)
            elif isinstance(m, ToolMessage):
                tool_exec_msg = AIMessage(
                    name="ReAct",
                    content=f"Tool `{m.name}` executed with output:\n" + "\n".join(["> " + line for line in m.content.splitlines()]),
                    additional_kwargs={"stage": "execution_output", "process": "progress"},
                )
                streamed_messages.append(tool_exec_msg)

                if m.name and "plot" in m.name:
                    plot_description = tool_input_map.get(getattr(m, "tool_call_id", ""))
                    if plot_description:
                        artefact_msgs.append(
                            AIMessage(
                                name="ReAct",
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
            name="ReAct",
            content=str(e),
            additional_kwargs={"stage": "execution_output", "process": "error"},
        )
        return {
            "final_response": None,
            "artefacts": [],
            "messages": [error_msg],
            "error_summary": str(e),
        }