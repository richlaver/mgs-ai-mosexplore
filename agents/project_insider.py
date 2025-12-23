"""Agent for enriching context with project-specific insights."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.types import Command

from classes import ContextState
from utils.context_data import get_project_specific_context

LOGGER = logging.getLogger(__name__)


def _lookup_project_context(selected_project_key: str) -> Optional[str]:
    """Return the project-specific context string for the selected project."""
    # Delete the following block when the project-specific context in the API has been corrected.
    if 'hanoi' in selected_project_key:
        return """
**IMPORTANT**:
You MUST ensure that whenever you call review level tools or extract review levels ALWAYS specify a `data` field in the tool prompt.
If you want to check review levels for a `calculation` field, you MUST convert it to the corresponding `data` field first.
For example, if you want to check review levels for `calculation2`, you MUST specify `data2` in the tool prompt instead.
"""
    if 'lpp' in selected_project_key:
        return """
**IMPORTANT**:
If a reading value is stored in a `calculation` field and the value of the reading is an empty string you MUST regard this as a missing reading.
Therefore when you are extracting such values you MUST use:
```sql
CASE WHEN JSON_VALID(column) AND NULLIF(JSON_VALUE(column, '$.path'), '') IS NOT NULL THEN JSON_VALUE(column, '$.path') ELSE NULL END
```
and:
```sql
WHERE NULLIF(JSON_VALUE(column, '$.path1'), '') IS NOT NULL
```
For `calculation` field values which are not reading values simply use:
```sql
CASE WHEN JSON_VALID(column) THEN JSON_EXTRACT(column, '$.path') ELSE NULL END
```
"""
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
