"""Agent for enriching context with platform-specific semantics."""
from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.types import Command

from classes import ContextState
from parameters import platform_context

LOGGER = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b")


def _get_retrospective_query(state: ContextState) -> str:
    """Safely extract the retrospective query from the incoming state."""
    ctx = state.context
    if isinstance(ctx, dict):
        value = ctx.get("retrospective_query")
    else:
        value = getattr(ctx, "retrospective_query", None)
    if isinstance(value, str):
        return value.strip()
    return ""


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase terms for keyword matching."""
    return {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)}


def _keyword_matches(keyword: str, normalized_query: str, tokenized_query: set[str]) -> bool:
    """Determine whether a keyword matches the query."""
    lowered = keyword.strip().lower()
    if not lowered:
        return False
    if " " in lowered:
        return lowered in normalized_query
    return lowered in tokenized_query


def platform_expert(state: ContextState) -> Command:
    """Look up platform semantics relevant to the retrospective query."""
    query = _get_retrospective_query(state)
    normalized_query = query.lower()
    tokenized_query = _tokenize(normalized_query)

    matching_blocks: list[str] = []
    matched_keywords: set[str] = set()

    for entry in platform_context:
        keywords = entry.get("query_keywords", [])
        if not isinstance(keywords, list) or not keywords:
            continue

        entry_matches = False
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            if _keyword_matches(keyword, normalized_query, tokenized_query):
                matched_keywords.add(keyword.strip().lower())
                entry_matches = True
                break

        if not entry_matches:
            continue

        context_value = entry.get("context")
        if isinstance(context_value, str):
            context_value = context_value.strip()
            if context_value:
                matching_blocks.append(context_value)

    context_update: dict[str, Any] = {}
    if matching_blocks:
        combined_context = "\n\n".join(matching_blocks)
        context_update["platform_context"] = combined_context
        LOGGER.info(
            "Platform Expert: Retrieved %d platform context block(s) for keywords %s.",
            len(matching_blocks),
            sorted(matched_keywords),
        )
    else:
        if query:
            LOGGER.info(
                "Platform Expert: No platform context matched the retrospective query '%s'.",
                query,
            )
        else:
            LOGGER.warning("Platform Expert: Retrospective query unavailable; skipping lookup.")

    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "platform_context_provided": True,
        },
    )
