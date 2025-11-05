import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from classes import Execution, Context

logger = logging.getLogger(__name__)

codeact_coder_prompt = PromptTemplate(
    input_variables=[
        "current_date",
        "retrospective_query",
        "verified_instrument_ids",
        "verified_instrument_info_json",
        "word_context_json",
        "relevant_date_ranges_json",
        "tools_str",
        "previous_attempts_summary",
    ],
    template="""
# Role
You are an expert in creating robust execution plans and implementing them in Python code to answer queries on instrumentation monitoring data in a database.

# Task
Generate code to answer the following user query plus an optional extension that adds value:
{retrospective_query}

# Context
- Current date:
{current_date}
- Instrument IDs in query with their type and subtype:
{verified_instrument_info_json}
- Background behind words in query (instrument types and subtypes, database fields to access, labelling, units and how to use extracted data):
{word_context_json}
- Date ranges relevant to query and how to apply:
{relevant_date_ranges_json}
- Tools available to your code:
{tools_str}
- Summary of previous failed coding attempts:
{previous_attempts_summary}

# Constraints on Your Generated Code
## Structure
- Write a single function named `execute_strategy` that takes no parameters and returns a generator yielding dictionaries defined as:
`def execute_strategy() -> Generator[dict, None, None]:`
- Omit docstrings.
- Dynamically respond to extracted data and errors for robustness.
- Use `try`-`except` blocks to handle exceptions but continuing where possible.
- Already in environment (DO NOT import):
  * `extraction_sandbox_agent`
  * `timeseries_plot_sandbox_agent`
  * `map_plot_sandbox_agent`
  * `datetime` module (NOT class)
- Available for import:
  * `numpy`
  * `pandas`

## Yielded Output
- Yield dictionaries as:
{{
    "content": str,
    "metadata":
        {{
            "timestamp": `datetime.datetime.now(timezone.utc).isoformat()`,
            "step": step number in execution plan (int),
            "type": "progress"|"error"|"final"|"plot"
        }}
}}
- When to yield different types:
  * "progress": at start of each step
  * "error": on exceptions, with error message
  * "final": after all steps with result summary for query and optional extension, self-explanatory and comprehensive for downstream interpretation
  * "plot": when calling a plotting tool
- ALWAYS yield a "final" output.
- For "plot" yields write content as:
{{
    "tool_name": "timeseries_plot_sandbox_agent"|"map_plot_sandbox_agent",
    "description": detailed description of plot,
    "artefact_id": artefact ID of plot
}}

## Commenting
- Divide into code blocks, each corresponding to a step in the execution plan.
- Precede each code block with comments explaining block:
  * Step number and summary
  * Rationale: why step is needed
  * Implementation: how step is implemented
- No other comments apart from block explanations to save tokens.

# Tools
## `extraction_sandbox_agent`
### How to Use
- Use to extract data from database with `extraction_sandbox_agent.invoke(prompt)`
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
- Use to display change of data with time for one or more series with `timeseries_plot_sandbox_agent.invoke(prompt)`.
- Prompt is natural language description of plot which MUST include:
  * Instrument IDs for each series
  * Database field names for extracting data
  * Time range
  * Axis label (get from background to query)
  * Axis unit (get from background to query)
- You can also specify:
  * Secondary y-axis with title and unit
  * Review levels to overlay
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
  * Review levels: -10, -5, -2, 2, 5, 10"

## `map_plot_sandbox_agent`
### How to Use
- Use to display spatial distribution of readings or review status with `map_plot_sandbox_agent.invoke(prompt)`.
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

### Example Prompt
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

# Sequential Instructions
1. Analyse the user query to understand what is being asked.
2. Deduce the user's underlying need.
3. Think up an optional extension to the query that adds value to the user's need.
4. Produce a step-by-step execution plan to answer the query and optional extension. The execution plan defines steps to execute in the code and DOES NOT include these instruction steps.
5. Write the code to implement the execution plan.
6. Check code for:
  - Logic to answer query
  - Adheres to constraints
  - Calls tools correctly with necessary inputs
  - Yielded outputs formed correctly
7. Output only the code.
"""
)

def strip_code_tags(code: str) -> str:
    """Remove markdown and HTML tags from Python code, preserving valid code."""
    logger.debug(f"Original code before tag stripping: {code}")
    
    # Strip markdown code fences (```python, ```, or similar)
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[-1].strip() == "```":
            code = "\n".join(lines[1:-1]).strip()
        else:
            code = "\n".join(lines[1:]).strip()
    
    # Strip HTML <code> tags
    code = re.sub(r'<code>\s*', '', code)
    code = re.sub(r'\s*</code>', '', code)
    
    # Remove any remaining backticks
    code = code.replace('```', '')
    
    logger.debug(f"Code after tag stripping: {code}")
    return code

def codeact_coder_agent(
    llm: BaseLanguageModel,
    tools: List[Any],
    context: Context,
    previous_attempts: List[Execution]
) -> Dict[str, Any]:
    """Generates code in one LLM call.

    Args:
        llm: Language model.
        tools: Available tool instances (including plotting tools).
        context: Context object.
        previous_attempts: Prior Execution objects.

    Returns:
        Dict for state update: {"executions": [new_execution]}
    """
    retrospective_query = context.retrospective_query if context else ""
    word_context_json = json.dumps([qw.model_dump() for qw in context.word_context] if context and context.word_context else [], indent=2)
    verified_instrument_info_json = json.dumps(context.verif_ID_info or {}, indent=2) if context else "{}"
    relevant_date_ranges_json = json.dumps([
        r.model_dump() for r in (context.relevant_date_ranges or [])
    ], indent=2) if context else "[]"

    current_date = datetime.now().strftime('%B %d, %Y')

    tools_str = json.dumps([
        {"name": t.name, "description": t.description}
        for t in tools
    ], indent=2) or "[]"

    previous_attempts_summary = "No previous failed attempts."
    if previous_attempts:
        previous_attempts_summary = "Summary of previous failed coding attempts:\n"
        for i, attempt in enumerate(previous_attempts):
            if attempt.error_summary:
                previous_attempts_summary += f"Attempt {i+1} failed.\n"
                previous_attempts_summary += f"Error summary: {attempt.error_summary}\n\n"

    chain = codeact_coder_prompt | llm

    try:
        response = chain.invoke({
            "current_date": current_date,
            "retrospective_query": retrospective_query,
            "verified_instrument_info_json": verified_instrument_info_json,
            "word_context_json": word_context_json,
            "relevant_date_ranges_json": relevant_date_ranges_json,
            "tools_str": tools_str,
            "previous_attempts_summary": previous_attempts_summary,
        })

        cleaned_code = strip_code_tags(response.content)

        new_execution = Execution(
            agent_type="CodeAct",
            parallel_agent_id=0,
            retry_number=len(previous_attempts),
            codeact_code=cleaned_code,
            final_response=None,
            artefacts=[],
            error_summary=""
        )

        return {"executions": [new_execution]}

    except Exception as e:
        logger.error("Error in codeact_coder_agent: %s", str(e))
        new_execution = Execution(
            agent_type="CodeAct",
            parallel_agent_id=0,
            retry_number=len(previous_attempts),
            codeact_code="",
            final_response=None,
            artefacts=[],
            error_summary=str(e)
        )
        return {"executions": [new_execution]}