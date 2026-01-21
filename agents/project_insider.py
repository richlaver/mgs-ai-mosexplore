"""Agent for enriching context with project-specific insights."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.types import Command

from classes import ContextState
from utils.context_data import get_project_specific_context
from utils.project_selection import get_project_config

LOGGER = logging.getLogger(__name__)


def _lookup_project_context(selected_project_key: str) -> Optional[str]:
    """Return the project-specific context string for the selected project."""
    context_value = get_project_specific_context(selected_project_key)
    if isinstance(context_value, str) and context_value.strip():
        return context_value.strip()
    return None


def project_insider(state: ContextState, selected_project_key: str) -> Command:
    context_update: dict[str, Any] = {}
    if selected_project_key:
        project_context = _lookup_project_context(selected_project_key)
        if project_context:
            LOGGER.info(f"Project Insider: Retrieved context for project key '{selected_project_key}'.")
            context_update["project_specific_context"] = project_context
        else:
            LOGGER.warning(f"Project Insider: No context found for project key '{selected_project_key}'.")
    else:
        LOGGER.warning("Project Insider: No selected project key provided.")

    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "project_specifics_retrieved": True,
        },
    )
