import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from classes import Execution, Context
from utils.async_utils import run_async_syncsafe
from utils.timezone_utils import tzinfo_from_offset

logger = logging.getLogger(__name__)

_CODEACT_STATIC_PROMPT_PATH = Path(__file__).resolve().parents[1] / "cached_llm_content" / "codeact_coder_prompt_static.md"


def _load_codeact_static_prompt() -> str:
  try:
    return _CODEACT_STATIC_PROMPT_PATH.read_text(encoding="utf-8")
  except Exception as exc:
    logger.warning("Failed to load codeact static prompt: %s", exc)
    return ""

codeact_coder_prompt = PromptTemplate(
    input_variables=[
        "current_date",
        "retrospective_query",
        "validated_type_info_json",
        "verified_instrument_info_json",
        "word_context_json",
        "relevant_date_ranges_json",
        "timezone_context_json",
        "platform_context",
        "project_specific_context",
        "previous_attempts_summary",
    ],
    template="""
# Runtime Context Values
- User query:
{retrospective_query}
- Current date (project timezone):
{current_date}
- Validated instrument types and subtypes referenced in the query (complete list from the database; do not infer additional types or subtypes):
{validated_type_info_json}
- Validated instrument IDs with their type and subtype mappings (only rely on these confirmed ID-type pairs from the database):
{verified_instrument_info_json}
- Background behind words in query (instrument types and subtypes, database fields to access, labelling, units and how to use extracted data):
{word_context_json}
- Date ranges relevant to query and how to apply:
{relevant_date_ranges_json}
- Timezones (project vs user vs sandbox):
{timezone_context_json}
- Platform-specific terminology and semantics:
{platform_context}
- Additional context on database:
{project_specific_context}
- Summary of previous failed coding attempts:
{previous_attempts_summary}

# Output Schema
- Return EXACTLY one JSON object with the following fields and no additional prose, Markdown, or prefixes:
  * `objective`: string describing the underlying analytics objective in user-neutral language (<= 50 words).
  * `plan`: array of concise strings, each string describing one numbered step necessary to fulfil the objective. Keep steps minimal but specific.
  * `code`: string containing the generated code that already follows the commenting constraints above. Do not wrap the code in Markdown fences or HTML tags.
- Ensure the JSON uses double-quoted keys/values per RFC 8259 and is syntactically valid even if embedded verbatim into a Python string literal.
- Do not escape newlines inside the `code` string beyond what is required by JSON; rely on `\n` for line breaks.
- The `code` string must not contain leading explanations, and only include the required block comment preceding each step.
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

  @classmethod
  def model_json_schema(cls, *args, **kwargs):
    schema = super().model_json_schema(*args, **kwargs)

    def _strip_additional_properties(obj: Any) -> None:
      if isinstance(obj, dict):
        obj.pop("additionalProperties", None)
        for value in obj.values():
          _strip_additional_properties(value)
      elif isinstance(obj, list):
        for value in obj:
          _strip_additional_properties(value)

    _strip_additional_properties(schema)
    return schema

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


def strip_think_blocks(text: str) -> str:
  """Remove <think>...</think> or <think>...<\\think> blocks from text."""
  if not text:
    return text
  cleaned = re.sub(
    r"<think>.*?(</think>|<\\think>)",
    "",
    text,
    flags=re.DOTALL | re.IGNORECASE,
  )
  return cleaned.lstrip("\n").rstrip()


def extract_json_object(text: str) -> str | None:
  if not text:
    return None
  start = text.find("{")
  end = text.rfind("}")
  if start == -1 or end == -1 or end <= start:
    return None
  return text[start:end + 1]


def strip_trailing_asyncio_run_notice(code: str) -> str:
  lines = code.rstrip().splitlines()
  while lines and "asyncio.run(execute_strategy())" in lines[-1]:
    lines.pop()
  return "\n".join(lines).strip()


def has_execute_strategy(code: str) -> bool:
  return "def execute_strategy():" in code


def _make_codeact_message(content: str) -> AIMessage:
  return AIMessage(
    name="CodeActCoder",
    content=content,
    additional_kwargs={
      "level": "debug",
      "is_final": False,
      "timestamp": datetime.now().astimezone().isoformat(),
      "origin": {
        "process": "codeact_coder",
        "thinking_stage": None,
        "branch_id": None,
      },
      "is_child": True,
      "artefacts": [],
    },
  )

def codeact_coder_agent(
    generating_llm: BaseLanguageModel,
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
  validated_type_info_json = json.dumps(context.verif_type_info or [], indent=2) if context else "[]"
  relevant_date_ranges_json = json.dumps([
    r.model_dump() for r in (context.relevant_date_ranges or [])
  ], indent=2) if context else "[]"
  platform_context = context.platform_context or ""
  project_specific_context = context.project_specific_context or ""

  timezone_context_raw: Dict[str, Any] = {}
  if context and context.timezone_context:
    timezone_context_raw = (
      context.timezone_context.model_dump()
      if hasattr(context.timezone_context, "model_dump")
      else context.timezone_context
    ) or {}
  timezone_context_json = json.dumps(timezone_context_raw, indent=2)

  project_offset = timezone_context_raw.get("project_timezone_offset") if isinstance(timezone_context_raw, dict) else None
  project_tzinfo = tzinfo_from_offset(project_offset) or datetime.now().astimezone().tzinfo
  app_now = datetime.now().astimezone()
  project_now = app_now.astimezone(project_tzinfo)
  current_date = project_now.strftime('%B %d, %Y %H:%M (%Z%z) [project timezone]')

  previous_attempts_summary = "No previous failed attempts."
  if previous_attempts:
    previous_attempts_summary = "Summary of previous failed coding attempts:\n"
    for i, attempt in enumerate(previous_attempts):
      if attempt.error_summary:
        previous_attempts_summary += f"Attempt {i+1} failed.\n"
        previous_attempts_summary += f"Error summary: {attempt.error_summary}\n\n"

  structured_llm = generating_llm.with_structured_output(
    CodeactCoderResponse,
    method="json_mode",
  )
  tools_payload = json.dumps(
    [
      {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
      for t in (tools or [])
    ],
    indent=2,
  )
  static_prompt = _load_codeact_static_prompt()
  if static_prompt:
    static_prompt = static_prompt.replace("<<TOOLS_STR>>", tools_payload)

  max_format_retries = 4
  required_entrypoint = "def execute_strategy():"

  try:
    response = None
    last_err: Exception | None = None
    runtime_retry_errors: List[str] = []

    for attempt in range(max_format_retries + 1):
      try:
        attempt_previous_attempts_summary = previous_attempts_summary
        if runtime_retry_errors:
          attempt_previous_attempts_summary = (
            f"{previous_attempts_summary}\n"
            "Runtime errors from immediate regeneration attempts:\n"
            + "\n".join(runtime_retry_errors)
          )

        prompt_payload = {
          "current_date": current_date,
          "retrospective_query": retrospective_query,
          "validated_type_info_json": validated_type_info_json,
          "verified_instrument_info_json": verified_instrument_info_json,
          "word_context_json": word_context_json,
          "relevant_date_ranges_json": relevant_date_ranges_json,
          "timezone_context_json": timezone_context_json,
          "platform_context": platform_context,
          "project_specific_context": project_specific_context,
          "previous_attempts_summary": attempt_previous_attempts_summary,
        }
        prompt_text = codeact_coder_prompt.format(**prompt_payload)
        if static_prompt:
          prompt_text = f"{static_prompt}\n\n{prompt_text}"

        message = HumanMessage(content=prompt_text)
        candidate = run_async_syncsafe(structured_llm.ainvoke([message]))

        if candidate is None:
          raise ValueError("Structured output returned no result.")

        cleaned_code = strip_trailing_asyncio_run_notice(
          strip_code_tags(candidate.code)
        )
        if not cleaned_code:
          raise ValueError("Structured output contained no code.")
        if not has_execute_strategy(cleaned_code):
          raise RuntimeError(
            f"Structured output missing required entrypoint: {required_entrypoint}"
          )

        response = candidate
        break
      except Exception as exc:
        last_err = exc
        if isinstance(exc, RuntimeError):
          runtime_retry_errors.append(str(exc))
        msg = str(exc).lower()
        is_format_issue = isinstance(exc, ValidationError) or any(
          hint in msg for hint in [
            "validation",
            "schema",
            "json",
            "parse",
            "format",
            "structured",
            "code",
            "no generations",
            "no code",
            "entrypoint",
          ]
        )
        if is_format_issue and attempt < max_format_retries:
          try:
            raw_message = run_async_syncsafe(generating_llm.ainvoke([message]))
            raw_content = getattr(raw_message, "content", str(raw_message))
            raw_content = strip_think_blocks(raw_content)
            json_blob = extract_json_object(raw_content)
            if json_blob:
              candidate = CodeactCoderResponse.model_validate_json(json_blob)
              cleaned_code = strip_trailing_asyncio_run_notice(
                strip_code_tags(candidate.code)
              )
              if cleaned_code and has_execute_strategy(cleaned_code):
                response = candidate
                break
          except Exception as raw_exc:
            logger.warning(
              "Raw-response parsing failed after structured output error: %s",
              raw_exc,
            )
          logger.warning(
            "Structured code generation retry (attempt %d/%d); retrying: %s",
            attempt + 1,
            max_format_retries + 1,
            exc,
          )
          continue
        raise

    if response is None:
      raise RuntimeError(f"Code generation failed after retries: {last_err}")

    objective = response.objective.strip()
    plan_steps = response.plan
    cleaned_code = strip_trailing_asyncio_run_notice(
      strip_code_tags(response.code)
    )
    if not has_execute_strategy(cleaned_code):
      raise RuntimeError(
        f"Generated code missing required entrypoint: {required_entrypoint}"
      )

    logger.debug("Generated objective: %s", objective)
    logger.debug("Generated plan: %s", plan_steps)

    plan_body = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan_steps))
    messages = [
      _make_codeact_message(f"Objective:\n{objective}"),
      _make_codeact_message(f"Plan:\n{plan_body}"),
      _make_codeact_message(f"```python\n{cleaned_code}\n```"),
    ]

    new_execution = Execution(
      parallel_agent_id=0,
      retry_number=len(previous_attempts),
      objective=objective,
      plan=list(plan_steps),
      codeact_code=cleaned_code,
      final_response=None,
      artefacts=[],
      error_summary=""
    )

    return {"executions": [new_execution], "messages": messages}

  except Exception as e:
    logger.error("Error in codeact_coder_agent: %s", str(e))
    new_execution = Execution(
      parallel_agent_id=0,
      retry_number=len(previous_attempts),
      objective="",
      plan=[],
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