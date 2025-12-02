import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from classes import Execution, Context

logger = logging.getLogger(__name__)

code_check_prompt = PromptTemplate(
    input_variables=["code"],
    template="""
# Role
You are an expert code checker.

# Task
Check the following Python code for syntax errors.
If you identify any syntax errors, correct them and output the correct code.
Do not consider missing imports in your check.
If there are no errors, output "CORRECT".

# Common Coding Errors to Look-out For
- AttributeError: 'coroutine' object has no attribute 'get_name'.
- KeyError: <coroutine object as_completed.<locals>._wait_for_one at 0x2b863e6e1b10>
- Missing `await` in async code when invoking tools or calling async functions.
- Attempting to access `now()` method from `datetime` module instead of `datetime` class.
- AttributeError: type object 'datetime.datetime' has no attribute 'datetime'.
- Tool call arguments not exactly matching ainvoke(tool_name, prompt) signature.

# Code
{code}

# Output
Output "CORRECT" if the code is correct.
Output ONLY the corrected code if the code is incorrect. Do not include any other text.
"""
)

codeact_coder_prompt = PromptTemplate(
    input_variables=[
        "current_date",
        "retrospective_query",
        "verified_instrument_ids",
        "verified_instrument_info_json",
        "word_context_json",
        "relevant_date_ranges_json",
        "platform_context",
        "project_specific_context",
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
- Platform-specific terminology and semantics:
{platform_context}
- Additional context on database:
{project_specific_context}
- Tools available to your code:
{tools_str}
- Summary of previous failed coding attempts:
{previous_attempts_summary}

# Constraints on Your Generated Code
## Structure
- Do not include any imports in your code because they are already imported in the sandbox namespace.
- Already in the namespace:
  * `extraction_sandbox_agent`
  * `timeseries_plot_sandbox_agent`
  * `map_plot_sandbox_agent`
  * `review_by_value_agent`
  * `review_by_time_agent`
  * `review_schema_agent`
  * `breach_instr_agent`
  * `datetime` class from `datetime` module
  * `timezone` class from `datetime` module
  * `timedelta` class from `datetime` module
  * `pandas` module as `pd`
  * `numpy` module as `np`
  * `ainvoke` and `asyncio` modules
- Define a single function named `execute_strategy` that takes no parameters.
- Execute tools and code in parallel where possible to reduce latency.
- Declare as `async def` if running parallel, otherwise `def`.
- If async: `async def execute_strategy() -> AsyncGenerator[dict, None]:`
- If sync: `def execute_strategy() -> Generator[dict, None, None]:`
- Omit docstrings.
- Dynamically respond to extracted data and errors for robustness.
- Use `try`-`except` blocks to handle exceptions but continue where possible.
- When running multiple named asyncio Tasks where the result needs to be identified by the task name (e.g., differentiating between a 'plot' task and a 'data' task), ALWAYS use `asyncio.wait(tasks)`. Never use `asyncio.as_completed` for named tasks, as it strips task identity.

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
- For "plot" yields write "content" field of above yield dictionary as:
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
- Returns `pandas.DataFrame` with columns (`instrument_id`, `db_field_name`, `db_field_value`, `review_name`) or `None` or `ERROR: <error message>`.
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
- Returns `pandas.DataFrame` with columns (`instrument_id`, `db_field_name`, `review_name`, `db_field_value`, `db_field_value_timestamp`) or `None` or `ERROR: <error message>`.
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
- Use to get instruments whose latest reading before a timestamp is at a specified review status (surpasses specified review level but not surpassing any more severe levels) strictly following `breach_instr_agent.invoke(prompt)` or in async code `await ainvoke(breach_instr_agent, prompt)`.
- Prompt is natural language description of review status enquiry that MUST include:
  * Review level name
  * Instrument type
  * Instrument subtype (optional)
  * Database field name
  * Timestamp cut-off
- Returns `pandas.DataFrame` with columns (`instrument_id`, `field_value`, `field_value_timestamp`, `review_value`) or `None` or `ERROR: <error message>`.
### Example Prompt
"List breaches for:
- Review level name: ALERT
- Instrument type: LP
- Instrument subtype: MOVEMENT
- Database field name: calculation1
- Timestamp: 14 May 2025 12:00:00 PM"

# Common Coding Errors to Avoid
- AttributeError: 'coroutine' object has no attribute 'get_name'.
- KeyError: <coroutine object as_completed.<locals>._wait_for_one at 0x2b863e6e1b10>
- Missing `await` in async code when invoking tools or calling async functions.
- Attempting to access `now()` method from `datetime` module instead of `datetime` class.
- Tool call arguments not exactly matching ainvoke(tool_name, prompt) signature.

# Sequential Instructions
1. Analyse the user query to understand what is being asked.
2. Deduce the user's underlying need.
3. Think up an optional extension to the query that adds value to the user's need.
4. Produce a step-by-step execution plan to answer the query and optional extension. The execution plan defines steps to execute in the code and DOES NOT include these instruction steps.
5. Write the code to implement the execution plan. Run tools and code in parallel whereever possible. If using async, process results as they complete (e.g., `asyncio.as_completed`).
6. Check code for:
  - Common coding errors above
  - Logic to answer query
  - Adheres to constraints
  - Calls tools correctly with necessary inputs
  - Yielded outputs formed correctly
7. Output only the code. Do not include any other text to save tokens.
"""
)

def extract_code_from_response(response: str) -> str:
  """Extracts the code part from the LLM response by removing preceding thought processes."""
  lines = response.splitlines()

  # Rule 1: Find "async def execute_strategy():" and return from that line onward
  # This strips any imports and comments that precede the strategy function.
  for i, line in enumerate(lines):
    if line.strip().startswith("async def execute_strategy():"):
      return "\n".join(lines[i:])

  # Rule 1: Find "async def execute_strategy():" then search backward
  # for i, line in enumerate(lines):
  #   if line.strip().startswith("async def execute_strategy():"):
  #     start_index = i
  #     for j in range(i - 1, -1, -1):
  #       prev_line = lines[j]
  #       if "import " in prev_line or prev_line.strip() == "":
  #         start_index = j
  #       else:
  #         break
  #     return "\n".join(lines[start_index:])

  # Rule 2: The first line of the first two consecutive lines starting with "import "
  for i in range(len(lines) - 1):
    if lines[i].strip().startswith("import ") and lines[i+1].strip().startswith("import "):
      return "\n".join(lines[i:])

  # Rule 3: The line after the first line with "```python"
  for i, line in enumerate(lines):
    if "```python" in line:
      return "\n".join(lines[i+1:])

  # Rule 4: The line after the first line with "python"
  for i, line in enumerate(lines):
    if "python" in line:
      return "\n".join(lines[i+1:])

  # Rule 5: The line after the first line with "<code>"
  for i, line in enumerate(lines):
    if "<code>" in line:
      return "\n".join(lines[i+1:])

  # Rule 6: The line after the first line with "```"
  for i, line in enumerate(lines):
    if "```" in line:
      return "\n".join(lines[i+1:])

  return response

def strip_code_tags(code: str) -> str:
  """Remove markdown and HTML tags from Python code, preserving valid code."""
  logger.debug(f"Original code before tag stripping: {code}")

  code = code.strip()
  # Strip markdown code fences (```python ... ```)
  if code.startswith("```"):
    lines = code.splitlines()
    if lines and lines[-1].strip() == "```":
      code = "\n".join(lines[1:-1]).strip()
    else:
      code = "\n".join(lines[1:]).strip()

  # Strip HTML <code> tags
  code = re.sub(r"<code>\s*", "", code)
  code = re.sub(r"\s*</code>", "", code)

  # Remove any remaining backticks
  code = code.replace("```", "")

  logger.debug(f"Code after tag stripping: {code}")
  return code

def codeact_coder_agent(
    generating_llm: BaseLanguageModel,
    checking_llm: BaseLanguageModel,
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
    platform_context = context.platform_context or ""
    project_specific_context = context.project_specific_context or ""

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

    generate_chain = codeact_coder_prompt | generating_llm
    check_chain = code_check_prompt | checking_llm

    try:
        response = generate_chain.invoke({
            "current_date": current_date,
            "retrospective_query": retrospective_query,
            "verified_instrument_info_json": verified_instrument_info_json,
            "word_context_json": word_context_json,
            "relevant_date_ranges_json": relevant_date_ranges_json,
            "platform_context": platform_context,
            "project_specific_context": project_specific_context,
            "tools_str": tools_str,
            "previous_attempts_summary": previous_attempts_summary,
        })

        extracted_code = extract_code_from_response(response.content)
        cleaned_code = strip_code_tags(extracted_code)

        check_response = check_chain.invoke({"code": cleaned_code})
        
        if "CORRECT" in check_response.content:
            final_code = cleaned_code
        else:
            checked_extracted_code = extract_code_from_response(check_response.content)
            if checked_extracted_code == check_response.content:
                final_code = cleaned_code
            else:
                final_code = strip_code_tags(checked_extracted_code)

        new_execution = Execution(
            agent_type="CodeAct",
            parallel_agent_id=0,
            retry_number=len(previous_attempts),
            codeact_code=final_code,
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