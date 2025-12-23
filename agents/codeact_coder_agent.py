import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, field_validator

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
- Missing `await` in async code when invoking tools or calling async functions.
- Attempting to access `now()` method from `datetime` module instead of `datetime` class.
- AttributeError: type object 'datetime.datetime' has no attribute 'datetime'.
- Tool call arguments not exactly matching ainvoke(tool_name, prompt) signature.
- SyntaxError: 'return' with value in async generator.
- ainvoke("tool_name", prompt) instead of ainvoke(tool_name, prompt). Do not use a string for tool_name.
- Assuming that `map_plot_sandbox_agent`, `timeseries_plot_sandbox_agent`, and `csv_saver_tool` return "artefact_id=<artefact_id>" instead of <artefact_id>. Only the artefact ID itself is returned as a string.

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
Generate code to answer the following user query:
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
## Imports
- Already in the namespace:
  * `extraction_sandbox_agent`
  * `timeseries_plot_sandbox_agent`
  * `map_plot_sandbox_agent`
  * `review_by_value_agent`
  * `review_by_time_agent`
  * `review_schema_agent`
  * `breach_instr_agent`
  * `csv_saver_tool`
  * `pandas` module as `pd`
  * `numpy` module as `np`
  * `ainvoke` and `asyncio` modules
- If you need the `datetime` class from the `datetime` module, import it as:
    `from datetime import datetime as datetime_class`
  This will avoid confusion with the `datetime` module.

## Structure
- Define a single function named `execute_strategy` that takes no parameters.
- Execute tools and code in parallel where possible to reduce latency.
- Declare as `async def` if running parallel, otherwise `def`.
- If async: `async def execute_strategy() -> AsyncGenerator[dict, None]:` and ensure any nested functions yield instead of return values.
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
            "type": "progress"|"error"|"final"|"plot"|"csv"
        }}
}}
- When to yield different types:
  * "progress": at start of each step
  * "error": on exceptions, with error message
  * "final": after all steps with result summary for query and optional extension, self-explanatory and comprehensive for downstream interpretation
  * "plot": when calling a plotting tool
  * "csv": when calling `csv_saver_tool`
- ALWAYS yield a "final" output.
- For "plot" yields write "content" field of above yield dictionary as:
{{
    "tool_name": "timeseries_plot_sandbox_agent"|"map_plot_sandbox_agent",
    "description": detailed description of plot,
    "artefact_id": artefact ID of plot
}}
- If you need to yield the contents of any DataFrame with more than 20 rows, ALWAYS use `csv_saver_tool` to save the DataFrame as a CSV and yield a "csv" output with "content" field as:
{{
    "tool_name": "csv_saver_tool",
    "description": detailed description of CSV,
    "artefact_id": artefact ID of saved CSV
}}
then yield only the first 20 rows of the DataFrame in the "final" output to save tokens.

## Commenting
- Divide into code blocks, each corresponding to a step in the execution plan.
- Precede each code block with comments explaining block:
  * Step number and summary
  * Rationale: why step is needed
  * Implementation: how step is implemented
- No other comments apart from block explanations to save tokens.

# Tools
## `csv_saver_tool`
### How to Use
- Use to save any pandas DataFrame larger than 20 rows as a CSV file in the file system. The CSV will be downloadable by the user.
- Call as:
```python
await csv_saver_tool.ainvoke({{
    "dataframe": dataframe_to_save,
    "description": "Detailed description of the CSV"
}})
```
or:
```python
await ainvoke(csv_saver_tool, {{
    "dataframe": dataframe_to_save,
    "description": "Detailed description of the CSV"
}})
```
- Returns artefact ID to access CSV in file system or `None`.

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
- If you need review status for multiple measurement values, combine into single prompt to reduce latency.
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
- If you need review status for multiple instruments and data fields, combine into single prompt to reduce latency.
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
- If you need review schemas for multiple instrument fields, combine into single prompt to reduce latency.
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

## `review_changes_across_period_agent`
### How to Use
- Use to find instruments whose review level status changed between two timestamps strictly following `review_changes_across_period_agent.invoke(prompt)` or in async code `await ainvoke(review_changes_across_period_agent, prompt)`.
- Prompt is natural language description of review level status change request that MUST include:
  * Start timestamp
  * End timestamp
- For more specific requests, you can include the following in the prompt:
  * Instrument type
  * Instrument subtype
  * Database field name
  * Start buffer in days to extend start timestamp backwards
  * End buffer in days to extend end timestamp backwards
  * Change direction ('up', 'down', 'both')
- Returns `pandas.DataFrame` with columns (instrument_id, db_field_name, start_review_name, start_review_value, start_field_value, start_field_value_timestamp, end_review_name, end_review_value, end_field_value, end_field_value_timestamp) or `None` or `ERROR: <error message>`.
### Example Prompt 1
"List review changes for:
- Start timestamp: 14 May 2025 12:00:00 PM
- End timestamp: 14 Jun 2025 09:00:00 PM
- Instrument type: LP
- Instrument subtype: MOVEMENT
- Database field name: calculation1
- Start buffer: 7
- End buffer: 7
- Change direction: both"
### Example Prompt 2
"List review changes for:
- Start timestamp: 14 Jul 2025 12:00:00 PM
- End timestamp: 14 Aug 2025 09:00:00 PM
- Change direction: up"

# Common Coding Errors to Avoid
- Missing `await` in async code when invoking tools or calling async functions.
- Attempting to access `now()` method from `datetime` module instead of `datetime` class.
- Tool call arguments not exactly matching ainvoke(tool_name, prompt) signature.
- Tool prompts missing mandatory details. Double check against tool guidance above.
- ainvoke("tool_name", prompt) instead of ainvoke(tool_name, prompt). Do not use a string for tool_name.
- SyntaxError: 'return' with value in async generator.
- Assuming that `map_plot_sandbox_agent`, `timeseries_plot_sandbox_agent`, and `csv_saver_tool` return "artefact_id=<artefact_id>" instead of <artefact_id>. Only the artefact ID itself is returned as a string.
- Assuming the `ainvoke` function does not exist and needs to be defined. Actually the `ainvoke` function is already imported and available for use.

# Sequential Instructions
1. Analyse the user query to understand what is being asked.
2. Deduce the user's underlying need.
3. Produce a step-by-step execution plan to answer the query. The execution plan defines steps to execute in the code and DOES NOT include these instruction steps.
4. Write the code to implement the execution plan. Run tools and code in parallel whereever possible. If using async, process results as they complete (e.g., `asyncio.as_completed`).
5. Check code for:
  - Common coding errors above
  - Logic to answer query
  - Adheres to constraints
  - Calls tools correctly with necessary inputs
  - Yielded outputs formed correctly
6. Output only the code. Do not include any other text to save tokens.

# Output Schema
- Return EXACTLY one JSON object with the following fields and no additional prose, Markdown, or prefixes:
  * `objective`: string describing the underlying analytics objective in user-neutral language (<= 50 words).
  * `plan`: array of concise strings, each string describing one numbered step necessary to fulfil the objective. Keep steps minimal but specific.
  * `code`: string containing the generated code that already follows the commenting constraints above. Do not wrap the code in Markdown fences or HTML tags.
- Ensure the JSON uses double-quoted keys/values per RFC 8259 and is syntactically valid even if embedded verbatim into a Python string literal.
- Do not escape newlines inside the `code` string beyond what is required by JSON; rely on `\n` for line breaks.
- The `code` string must not contain leading explanations, and only include the required block comments preceding each step.
- If you cannot follow the schema, return a JSON object with an `error` key describing the issue instead.
"""
)


class CodeactCoderResponse(BaseModel):
  objective: str
  plan: List[str]
  code: str

  model_config = ConfigDict(extra="forbid")

  @field_validator("objective", "code")
  @classmethod
  def _validate_non_empty_text(cls, value: str) -> str:
    value = value.strip()
    if not value:
      raise ValueError("Value must be a non-empty string.")
    return value

  @field_validator("plan")
  @classmethod
  def _validate_plan(cls, value: List[str]) -> List[str]:
    cleaned = [step.strip() for step in value if isinstance(step, str) and step.strip()]
    if not cleaned:
      raise ValueError("Plan must include at least one non-empty step.")
    return cleaned

def strip_code_tags(code: str) -> str:
  """Remove markdown and HTML tags from Python code, preserving valid code."""
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

  return code


def _make_codeact_message(content: str) -> AIMessage:
    return AIMessage(
        name="CodeActCoder",
        content=content,
        additional_kwargs={"stage": "node", "process": "codeact_coder_branch"},
    )

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
    Dict for state update: {"executions": [new_execution], "messages": [...]}
  """

  retrospective_query = context.retrospective_query if context else ""
  word_context_json = json.dumps([
    qw.model_dump() for qw in context.word_context
  ] if context and context.word_context else [], indent=2)
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

  structured_llm = generating_llm.with_structured_output(CodeactCoderResponse)
  generate_chain = codeact_coder_prompt | structured_llm
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

    objective = response.objective.strip()
    plan_steps = response.plan
    cleaned_code = strip_code_tags(response.code)

    logger.debug("Generated objective: %s", objective)
    logger.debug("Generated plan: %s", plan_steps)

    # Skip checking step because the code checker lacks the full context that was used to generate the code and hence often modifies the code to its detriment.
    # check_response = check_chain.invoke({"code": cleaned_code})

    # if "CORRECT" in check_response.content:
    #     final_code = cleaned_code
    # else:
    #     final_code = strip_code_tags(check_response.content)

    plan_body = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan_steps))
    messages = [
      _make_codeact_message(f"Objective:\n{objective}"),
      _make_codeact_message(f"Plan:\n{plan_body}"),
      _make_codeact_message(f"```python\n{cleaned_code}\n```"),
    ]

    new_execution = Execution(
      agent_type="CodeAct",
      parallel_agent_id=0,
      retry_number=len(previous_attempts),
      codeact_code=cleaned_code,
      final_response=None,
      artefacts=[],
      error_summary=""
    )

    return {"executions": [new_execution], "messages": messages}

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
    error_message = str(e)
    messages = [
      _make_codeact_message(f"Objective generation failed due to error: {error_message}"),
      _make_codeact_message(f"Plan generation failed due to error: {error_message}"),
      _make_codeact_message(f"```python\n# Code generation failed due to error: {error_message}\n```"),
    ]
    return {"executions": [new_execution], "messages": messages}