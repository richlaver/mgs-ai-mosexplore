import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from classes import CodingAttempt, Context

logger = logging.getLogger(__name__)

planner_coder_prompt = PromptTemplate(
    input_variables=[
        "current_date",
        "retrospective_query",
        "verified_instrument_ids",
        "verified_instrument_info_json",
        "word_context_json",
        "tools_str",
        "previous_attempts_summary",
    ],
    template="""You are an expert in creating robust execution plans and implementing them in Python code for helping to answer complex database queries using monitoring instrumentation data.

Current date: {current_date}

User query rephrased to incorporate chat history context:
{retrospective_query}

Verified instrument IDs in query:
{verified_instrument_ids}

Verified instrument ID details (JSON mapping of ID -> {{type, subtype}}):
{verified_instrument_info_json}

Structured instrument & field context (JSON format):
{word_context_json}

Available tools (name and description):
{tools_str}

Summary of previous failed attempts:
{previous_attempts_summary}

Thought processes (DO NOT include in output):
1. Deduce the user's underlying intention or need based on the query and context.
2. Brainstorm 3 alternative high-level strategies to answer the query using provided instrument context, data availability, and calculation feasibility.
   - For each: numbered high-level steps, pros, cons, probability of success (high/medium/low).
   - Explicitly state how each alternative avoids failure modes from previous attempts.
3. Evaluate the three alternatives comparatively (success probability, latency, completeness, robustness, simplicity).
4. Select the best alternative with justification.
5. Based on the deduced intention, brainstorm 3 possible helpful query extensions that add value. Each must be executable with the available database, tools, and coding abilities. Keep to one simple helpful addition.
6. DO NOT output the above in any code comments. This is important to save tokens.
7. Select the one extension most likely to contribute value in relation to the original query, with justification.
8. Think up a detailed execution plan as a numbered list (1., 2., 3., ...). Incorporate the selected extension as optional last steps. Each step must be action-oriented:
   - Specify instrument type/subtype & database field names to extract.
   - Explicit filters (date ranges, instrument IDs) or how to derive them.
   - When to call extraction_sandbox_agent (describe the prompt for SQL generation).
   - When to perform calculations (describe formula/aggregation).
   - Ensure logic: gather data -> validate -> calculate -> optional extension -> finalize.
   - Prefer minimal extractions; combine fields in one if possible.
   - If visualizations enhance the response, invoke plotting tools (timeseries_plot_sandbox_agent or map_plot_sandbox_agent) and yield their results as specified below.
   - Do not extract data for plotting tools — the tools will extract data themselves.

Task for output:
9. Generate the Python code implementing the plan.

Code Generation Constraints:
- Acquire inputs via tools in scope: call tool.invoke(input).
- extraction_sandbox_agent.invoke(prompt_str) expects str prompt, returns pandas.DataFrame or None.
- timeseries_plot_sandbox_agent.invoke(prompt_str) and map_plot_sandbox_agent.invoke(prompt_str) expect a str natural language prompt describing the plot, returns artefact ID to access plot in file system or None.
- Read tool descriptions and write prompts to include ALL required details.
- In scope: llm (BaseLanguageModel), db (SQLDatabase), extraction_sandbox_agent, timeseries_plot_sandbox_agent, map_plot_sandbox_agent, datetime module.
- Import standard libraries inside the function (e.g., from datetime import datetime, timezone; import pandas as pd; import json).
- No network calls or file I/O.
- Function signature: def execute_strategy() -> Generator[dict, None, None]: (NO parameters).
- Only insert comments explaining each step, no other comments
- Comments must be directly relevant to step they precede

Yielded Outputs:
- Yield dicts: {{ "content": str, "metadata": {{ "timestamp": str ISO8601 UTC, "step": int, "node": str, "type": "progress"|"error"|"final"|"plot" }} }}
- timestamp: datetime.now(timezone.utc).isoformat()
- step: the plan step number (int).
- node: "execute_strategy" for all yields.
- type "progress": yield at start ("Starting step N: short summary") and end ("Completed step N: brief status") of each step.
- type "error": on exceptions, with error message; continue if possible (especially for extension).
- type "final": one at end with self-explanatory summarizing result (answer query + optional extension; comprehensive for downstream interpretation).
- type "plot": content is json.dumps({{ "tool_name": "timeseries_plot_sandbox_agent" or "map_plot_sandbox_agent", "description": detailed_description, "artefact_id": artefact_id }}).
- If extension fails, still yield final answering main query.
- Dynamically indicate success of steps.

Code Structure:
- def execute_strategy() -> Generator[dict, None, None]:
- For each step N:
  # Step N: <summary>
  # Rationale: <why>
  # Implementation: <how>
  <code for step N, including yields>
- Wrap risky parts in try/except; yield error and continue/return as appropriate.
- If critical failure, yield error and return.

Think step by step following requirements 1-9 above. Output ONLY the code.
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

def planner_coder_agent(
    llm: BaseLanguageModel,
    tools: List[Any],
    context: Context,
    previous_attempts: List[CodingAttempt]
) -> Dict[str, Any]:
    """Generates code in one LLM call.

    Args:
        llm: Language model.
        tools: Available tool instances (including plotting tools).
        context: Context object.
        previous_attempts: Prior CodingAttempt objects.
        existing_plots: Ignored, as plots are generated in code.

    Returns:
        Dict for state update: {"coding_attempts": [new_attempt]}
    """
    retrospective_query = context.retrospective_query if context else ""
    word_context_json = json.dumps([qw.model_dump() for qw in context.word_context] if context and context.word_context else [], indent=2)
    verified_instrument_ids = ", ".join((context.verif_ID_info or {}).keys()) if context else "None"
    verified_instrument_info_json = json.dumps(context.verif_ID_info or {}, indent=2) if context else "{}"

    current_date = datetime.now().strftime('%B %d, %Y')

    tools_str = json.dumps([
        {"name": t.name, "description": t.description}
        for t in tools
    ], indent=2) or "[]"

    previous_attempts_summary = "No previous failed attempts."
    if previous_attempts:
        previous_attempts_summary = "Summary of previous failed coding attempts:\n"
        for i, attempt in enumerate(previous_attempts):
            if attempt.analysis:
                previous_attempts_summary += f"Attempt {i+1} failed.\n"
                previous_attempts_summary += f"Analysis: {attempt.analysis}\n\n"

    chain = planner_coder_prompt | llm

    try:
        response = chain.invoke({
            "current_date": current_date,
            "retrospective_query": retrospective_query,
            "verified_instrument_ids": verified_instrument_ids,
            "verified_instrument_info_json": verified_instrument_info_json,
            "word_context_json": word_context_json,
            "tools_str": tools_str,
            "previous_attempts_summary": previous_attempts_summary,
        })

        cleaned_code = strip_code_tags(response.content)

        new_attempt = CodingAttempt(
            plan="Plan embedded in code comments",
            code=cleaned_code,
        )

        return {"coding_attempts": [new_attempt]}

    except Exception as e:
        logger.error("Error in planner_coder_agent: %s", str(e))
        new_attempt = CodingAttempt(plan="", code="")
        return {"coding_attempts": [new_attempt]}