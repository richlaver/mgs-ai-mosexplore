"""Utilities for selecting and managing the current project configuration.

These helpers read from Streamlit's st.secrets to discover available
project entries under the "project_data.*" namespace.
"""

from __future__ import annotations

import logging
from typing import Dict, List
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def list_projects() -> List[Dict[str, str]]:
    """Return a list of available projects from st.secrets.

    Each entry is a dict: {"key": "project_data.<name>", "display_name": str}
    Ordered by display_name ascending.
    """
    import streamlit as st
    projects: List[Dict[str, str]] = []
    project_root = st.secrets.get("project_data", {})
    keys_preview = list(project_root.keys()) if isinstance(project_root, Mapping) else []
    if isinstance(project_root, Mapping):
        for sub_key, sub_val in project_root.items():
            display_name = sub_val.get("display_name", sub_key) if isinstance(sub_val, Mapping) else sub_key
            projects.append({"key": f"project_data.{sub_key}", "display_name": str(display_name)})
    projects.sort(key=lambda x: x["display_name"].lower())
    return projects


def get_selected_project_key() -> str:
    """Return the currently selected project key from session state, ensuring a default."""
    import streamlit as st
    return st.session_state.get("selected_project_key")


def get_project_config(project_key: str) -> Dict:
    """Return the config dict for the given dotted project key from st.secrets.

    Example: 'project_data.hanoi_clone' -> st.secrets.get('project_data', {}).get('hanoi_clone', {})
    """
    import streamlit as st
    if not project_key or "." not in project_key:
        return {}
    root, sub = project_key.split(".", 1)
    return dict(st.secrets.get(root, {}).get(sub, {}))
