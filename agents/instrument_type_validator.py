import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Set

from langchain_core.language_models import BaseLanguageModel
from langgraph.types import Command

from classes import ContextState
from utils.context_data import get_instrument_context, _summarize_context

logger = logging.getLogger(__name__)


def _extract_type_subtype_map(instrument_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build mapping of instrument types to their subtypes from the instrument context."""
    type_map: Dict[str, Set[str]] = defaultdict(set)
    if not isinstance(instrument_data, dict):
        return {}

    for entry_key, entry_value in instrument_data.items():
        if not isinstance(entry_value, dict):
            logger.debug("Skipping non-dict instrument entry %s", entry_key)
            continue

        type_name = entry_value.get("type")
        subtype_name = entry_value.get("subtype")
        if not isinstance(type_name, str) or not type_name:
            continue

        type_map[type_name]  # ensure key exists
        if isinstance(subtype_name, str) and subtype_name:
            type_map[type_name].add(subtype_name)

    return type_map


def _tokenize_query(query: str) -> List[str]:
    if not query:
        return []
    return re.findall(r"[A-Za-z0-9_\-/]+", query)


def instrument_type_validator(state: ContextState, selected_project_key: str) -> dict:
    """Detect instrument types/subtypes mentioned in the query and update context."""
    instrument_data = get_instrument_context(selected_project_key)
    logger.debug("Instrument data retrieved: %s", _summarize_context(instrument_data))
    type_subtype_map = _extract_type_subtype_map(instrument_data)
    logger.debug("Instrument type-subtype map: %s", type_subtype_map)

    if isinstance(state.context, dict):
        query = state.context.get("retrospective_query", "")
    else:
        query = getattr(state.context, "retrospective_query", "")

    tokens = _tokenize_query(query)
    token_set = set(tokens)
    logger.debug("Token set from query: %s", token_set)

    verif_type_info: List[Dict[str, List[str]]] = []
    for instr_type, subtypes in type_subtype_map.items():
        if instr_type not in token_set:
            continue

        matched_subtypes = sorted([sub for sub in subtypes if sub in token_set])
        if not matched_subtypes:
            matched_subtypes = sorted(subtypes)

        verif_type_info.append({
            "type": instr_type,
            "subtypes": matched_subtypes,
        })

    context_update = {
        "verif_type_info": verif_type_info if verif_type_info else []
    }

    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "instrument_types_validated": True,
        }
    )
