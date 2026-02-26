import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

import setup
from classes import Execution, Context
from utils.async_utils import run_async_syncsafe
from utils.timezone_utils import tzinfo_from_offset

logger = logging.getLogger(__name__)

codeact_coder_prompt = PromptTemplate(
    input_variables=[
        "current_date",
        "retrospective_query",
        "validated_type_info_json",
        "verified_instrument_info_json",
      "db_sources_json",
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
- Database sources selected to answer the query (instrument types, subtypes, database fields, labels, units, and interpretation context):
{db_sources_json}
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


def _parse_codeact_candidate(raw_message: Any) -> CodeactCoderResponse:
  raw_content = getattr(raw_message, "content", raw_message)
  if isinstance(raw_content, str):
    raw_text = raw_content
  elif isinstance(raw_content, list):
    parts: List[str] = []
    for item in raw_content:
      if isinstance(item, str):
        parts.append(item)
      elif isinstance(item, dict):
        if isinstance(item.get("text"), str):
          parts.append(item["text"])
        elif isinstance(item.get("type"), str) and item.get("type") == "text" and isinstance(item.get("text"), str):
          parts.append(item["text"])
        else:
          parts.append(str(item))
      else:
        parts.append(str(item))
    raw_text = "\n".join(parts)
  elif isinstance(raw_content, dict):
    text_part = raw_content.get("text")
    raw_text = text_part if isinstance(text_part, str) else str(raw_content)
  else:
    raw_text = str(raw_content)

  raw_text = strip_think_blocks(raw_text)
  json_blob = extract_json_object(raw_text)
  if not json_blob:
    raise ValueError("Could not extract JSON object from model response")
  return CodeactCoderResponse.model_validate_json(json_blob)


def is_retryable_generation_error(exc: Exception) -> bool:
  msg = str(exc).lower()
  retry_hints = [
    "clientresponse",
    "not subscriptable",
    "timeout",
    "timed out",
    "deadline",
    "connection",
    "network",
    "temporarily unavailable",
    "service unavailable",
    "rate limit",
    "too many requests",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
  ]
  return any(hint in msg for hint in retry_hints)


def strip_trailing_asyncio_run_notice(code: str) -> str:
  lines = code.rstrip().splitlines()
  while lines and "asyncio.run(execute_strategy())" in lines[-1]:
    lines.pop()
  return "\n".join(lines).strip()


_EXECUTE_STRATEGY_PATTERN = re.compile(
  r"(^|\n)\s*(?:async\s+)?def\s+execute_strategy\s*\(.*?\)\s*(?:->\s*[^:\n]+)?\s*:",
  flags=re.DOTALL,
)


def has_execute_strategy(code: str) -> bool:
  if not code:
    logger.debug("[CodeAct] Entrypoint check: no code supplied")
    return False
  match = _EXECUTE_STRATEGY_PATTERN.search(code)
  if match is None:
    logger.debug("[CodeAct] Entrypoint check: execute_strategy signature not found")
    return False
  signature_preview = match.group(0).strip().replace("\n", " ")
  if len(signature_preview) > 160:
    signature_preview = signature_preview[:157] + "..."
  logger.debug("[CodeAct] Entrypoint check: matched signature='%s'", signature_preview)
  return True


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
  db_sources_json = json.dumps([
    dbs.model_dump() for dbs in context.db_sources
  ] if context and context.db_sources else [], indent=2)
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

  tools_payload = json.dumps(
    [
      {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
      for t in (tools or [])
    ],
    indent=2,
  )
  static_prompt = ""
  try:
    static_prompt = setup.build_codeact_coder_cached_context(tools_payload)
  except Exception as exc:
    logger.warning("Failed to build codeact cached context from static prompt: %s", exc)

  cached_invoke_kwargs: Dict[str, Any] = {}
  if static_prompt and isinstance(generating_llm, ChatGoogleGenerativeAI):
    try:
      cached_content_name = setup.get_or_refresh_cached_content(
        cache_key="codeact_coder",
        content_text=static_prompt,
        llm=generating_llm,
        display_prefix="codeact-coder-cache",
        legacy_hash_keys=["codeact_coder_cached_context_hash"],
      )
      if cached_content_name:
        cached_invoke_kwargs = {"cached_content": cached_content_name}
        logger.info("CodeAct cache attached: %s", cached_content_name)
      else:
        logger.info("CodeAct cache unavailable; embedding static prompt directly")
    except Exception as cache_exc:
      logger.warning("CodeAct cache setup failed; embedding static prompt directly: %s", cache_exc)

  structured_llm = generating_llm.bind(
    response_mime_type="application/json",
    response_json_schema=CodeactCoderResponse.model_json_schema(),
    **cached_invoke_kwargs,
  )
  raw_llm = generating_llm.bind(**cached_invoke_kwargs) if cached_invoke_kwargs else generating_llm

  max_format_retries = 4
  required_entrypoint = "def/async def execute_strategy(...):"

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
          "db_sources_json": db_sources_json,
          "relevant_date_ranges_json": relevant_date_ranges_json,
          "timezone_context_json": timezone_context_json,
          "platform_context": platform_context,
          "project_specific_context": project_specific_context,
          "previous_attempts_summary": attempt_previous_attempts_summary,
        }
        prompt_text = codeact_coder_prompt.format(**prompt_payload)
        if static_prompt and not cached_invoke_kwargs:
          prompt_text = f"{static_prompt}\n\n{prompt_text}"

        message = HumanMessage(content=prompt_text)
        structured_response = run_async_syncsafe(structured_llm.ainvoke([message]))
        usage_metadata = getattr(structured_response, "usage_metadata", None)
        if usage_metadata:
          logger.info("CodeAct structured usage metadata: %s", usage_metadata)
        candidate = _parse_codeact_candidate(structured_response)

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
        is_transient_issue = is_retryable_generation_error(exc)
        if is_transient_issue and attempt < max_format_retries:
          runtime_retry_errors.append(str(exc))
          logger.warning(
            "Structured code generation transient error (attempt %d/%d); retrying: %s",
            attempt + 1,
            max_format_retries + 1,
            exc,
          )
          continue
        if is_format_issue and attempt < max_format_retries:
          try:
            raw_message = run_async_syncsafe(raw_llm.ainvoke([message]))
            raw_usage = getattr(raw_message, "usage_metadata", None)
            if raw_usage:
              logger.info("CodeAct raw fallback usage metadata: %s", raw_usage)
            candidate = _parse_codeact_candidate(raw_message)
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