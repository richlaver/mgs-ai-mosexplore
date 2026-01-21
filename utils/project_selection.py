"""Utilities for selecting and managing the current project configuration.

These helpers read from Streamlit's st.secrets to discover available
project entries under the "project_data.*" namespace.
"""

from __future__ import annotations

import logging
from typing import Dict, List
from collections.abc import Mapping
import re

logger = logging.getLogger(__name__)


def _sanitize_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return safe.lower()


def make_project_key(db_host: str, db_name: str) -> str:
    """Return a safe, deterministic key derived from db_host + db_name."""
    if not db_host or not db_name:
        return ""
    return f"{_sanitize_segment(str(db_host))}__{_sanitize_segment(str(db_name))}"


def list_projects() -> List[Dict[str, str]]:
    """Return a list of available projects from st.secrets.

    Each entry is a dict: {"key": "project_data.<db_host>__<db_name>", "display_name": str}
    Ordered by display_name ascending.
    """
    import streamlit as st
    projects: List[Dict[str, str]] = []
    project_root = st.secrets.get("project_data", {})
    keys_preview = list(project_root.keys()) if isinstance(project_root, Mapping) else []
    if isinstance(project_root, Mapping):
        for sub_key, sub_val in project_root.items():
            display_name = sub_val.get("display_name", sub_key) if isinstance(sub_val, Mapping) else sub_key
            derived_key = ""
            if isinstance(sub_val, Mapping):
                derived_key = make_project_key(str(sub_val.get("db_host", "")), str(sub_val.get("db_name", "")))
            project_key = derived_key or str(sub_key)
            projects.append({"key": f"project_data.{project_key}", "display_name": str(display_name)})
    projects.sort(key=lambda x: x["display_name"].lower())
    return projects


def get_selected_project_key() -> str:
    """Return the currently selected project key from session state, ensuring a default."""
    import streamlit as st
    return st.session_state.get("selected_project_key")


def get_project_config(project_key: str) -> Dict:
    """Return the config dict for the given project key from st.secrets.

    Example: 'project_data.<db_host>__<db_name>' -> st.secrets.get('project_data', {}).get('<matching entry>', {})
    """
    import streamlit as st
    if not project_key:
        return {}
    project_root = st.secrets.get("project_data", {})
    if not isinstance(project_root, Mapping):
        return {}
    if "." in project_key:
        _, sub = project_key.split(".", 1)
    else:
        sub = project_key
    direct = project_root.get(sub)
    if isinstance(direct, Mapping):
        return dict(direct)
    for sub_key, sub_val in project_root.items():
        if not isinstance(sub_val, Mapping):
            continue
        candidate = make_project_key(str(sub_val.get("db_host", "")), str(sub_val.get("db_name", "")))
        if candidate and candidate == sub:
            return dict(sub_val)
    return {}
