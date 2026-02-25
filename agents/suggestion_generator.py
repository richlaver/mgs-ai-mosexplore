from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timezone
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from classes import Context, Execution, Suggestion
from suggestions import suggested_queries
from utils.async_utils import run_async_syncsafe
from utils.json_utils import strip_to_json_payload

logger = logging.getLogger(__name__)


class SuggestionPayload(BaseModel):
    suggestion_1: str = Field(..., alias="1")
    suggestion_2: str = Field(..., alias="2")
    suggestion_3: str = Field(..., alias="3")

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
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

    def to_dict(self) -> dict[str, str]:
        return self.model_dump(by_alias=True)


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            return "\n".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
        except Exception:
            return str(content)
    return str(content or "")


def _best_response_text(executions: List[Execution]) -> str:
    best_ex = next((ex for ex in executions if ex.is_best), None)
    if best_ex is None:
        best_ex = next((ex for ex in executions if ex.final_response is not None), None)
    if not best_ex or not best_ex.final_response:
        return ""
    return _flatten_message_content(best_ex.final_response.content)


def _to_suggestion_list(payload: dict[str, Any]) -> list[str]:
    return [
        str(payload.get("1", "")).strip(),
        str(payload.get("2", "")).strip(),
        str(payload.get("3", "")).strip(),
    ]


def _sample_fallback_suggestions() -> dict[str, str]:
    pool = [q for q in suggested_queries if isinstance(q, str) and q.strip()]
    if len(pool) >= 3:
        sampled = random.sample(pool, 3)
    else:
        sampled = pool[:]
        while sampled and len(sampled) < 3:
            sampled.append(sampled[-1])
    while len(sampled) < 3:
        sampled.append("Show latest readings for key instruments this week.")
    return {
        "1": sampled[0],
        "2": sampled[1],
        "3": sampled[2],
    }


def suggestion_generator(
    llm: BaseLanguageModel,
    context: Context,
    messages: List,
    executions: List[Execution],
) -> tuple[List, List[Suggestion]]:
    logger.info("Starting suggestion_generator")

    updated_messages = messages.copy() if messages else []
    retrospective_query = context.retrospective_query if context else ""
    platform_context = context.platform_context if context and context.platform_context else ""
    response_text = _best_response_text(executions)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """# Role
You are a helpful assistant adept at deducing underlying intentions behind user queries on construction monitoring databases and excelling at suggesting helpful, relevant yet feasible follow-on queries.""",
        ),
        (
            "human",
            """# Task
Generate three query suggestions to follow-on from the following query:
<query>
{retrospective_query}
<\\query>
which was responded to with the following response text:
<response>
{response_text}
<\\response>
Assume that any plots or data mentioned in the response text were successfully conveyed to the user.
All of your suggestions must be able to be successfully executed given that:
- Queries can extract construction instrumentation monitoring data
- Queries can plot time series and maps of readings, reading changes, review level statuses and changes in review level status
Your suggestions must be helpful in addressing the user’s underlying need, diverse and relevant to the query and response.
Each suggestion must be stated in less than 13 words.
Each suggestion must be appropriate for button text.
# Steps
1. Analyse the query and deduce the three most likely underlying intentions of the user.
2. Understand the following information which may be relevant to the query:
<information>
{platform_context}
<\\information>
3. Generate one suggestion for each user intention.
# Output Format
Output **only JSON** exactly in the following format with no other text or thoughts:
{{
\"1\": \"first_suggestion\",
\"2\": \"second_suggestion\",
\"3\": \"third_suggestion\",
}}""",
        ),
    ])

    call_inputs = {
        "retrospective_query": retrospective_query,
        "response_text": response_text,
        "platform_context": platform_context,
    }

    parsed_payload: dict[str, str] | None = None
    try:
        structured_llm = llm.with_structured_output(SuggestionPayload, method="json_mode")
        chain = prompt | structured_llm
        structured_response = run_async_syncsafe(chain.ainvoke(call_inputs))
        if structured_response is None:
            raise ValueError("Structured suggestion output returned no result.")
        parsed_payload = structured_response.to_dict()
        logger.info("Structured suggestion generation succeeded")
    except Exception as structured_error:
        logger.warning("Structured suggestion generation failed; falling back to raw parsing: %s", structured_error)

    if parsed_payload is None:
        chain = prompt | llm
        response = run_async_syncsafe(chain.ainvoke(call_inputs))
        try:
            response_text_clean = strip_to_json_payload(
                _flatten_message_content(getattr(response, "content", "")),
                [
                    '"1"',
                    '"2"',
                    '"3"',
                ],
            )
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text_clean, re.DOTALL | re.IGNORECASE)
            if json_match:
                parsed_payload = json.loads(json_match.group(1))
                logger.debug("Successfully extracted suggestion JSON from markdown code block")
            else:
                parsed_payload = json.loads(response_text_clean)
        except Exception as parse_error:
            logger.error("Suggestion generator raw parsing failed: %s", parse_error)
            parsed_payload = _sample_fallback_suggestions()

    suggestion_texts = _to_suggestion_list(parsed_payload or {})
    suggestions: List[Suggestion] = []

    for idx, suggestion_text in enumerate(suggestion_texts, start=1):
        if not suggestion_text:
            continue
        suggestions.append(Suggestion(id=idx, suggestion=suggestion_text))
        updated_messages.append(
            AIMessage(
                name="suggestion_generator",
                content=suggestion_text,
                additional_kwargs={
                    "level": "info",
                    "is_final": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "origin": {
                        "process": "suggestion_generator",
                        "thinking_stage": None,
                        "branch_id": None,
                    },
                    "is_child": False,
                    "artefacts": [],
                },
            )
        )

    logger.info("Completed suggestion_generator with %d suggestions", len(suggestions))
    return updated_messages, suggestions
