from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.project_selection import make_project_key
from utils.context_api import (
    ContextAPIClient,
    ContextAPIError,
    ContextApiConfig,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    build_context_api_client,
)

LOGGER = logging.getLogger(__name__)
_DEFAULT_PAYLOAD = {
    "instrument_context": {},
    "project_specific_context": "",
    "project_name": None,
    "latest_version": None,
    "_instrument_loaded": False,
    "_project_context_loaded": False,
}

_CACHE_LOCK = threading.RLock()
_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}
_PROJECT_LIST_CACHE: Optional[List[Dict[str, Any]]] = None
_CONTEXT_CLIENT: Optional[ContextAPIClient] = None
_CONTEXT_CONFIG: Optional[ContextApiConfig] = None
_PROJECT_CONFIGS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_PROJECT_KEY: Optional[str] = None


def configure_context_api(config: Optional[ContextApiConfig] = None, client: Optional[ContextAPIClient] = None) -> None:
    """Configure the default context API client or config for background-safe usage."""
    global _CONTEXT_CONFIG, _CONTEXT_CLIENT
    if config is not None:
        _CONTEXT_CONFIG = config
    if client is not None:
        _CONTEXT_CLIENT = client


def configure_context_api_from_secrets(secrets: Mapping[str, Any]) -> None:
    """Configure the context API from a Streamlit secrets mapping (call from main thread)."""
    cfg_section = secrets.get("context_api", {}) if isinstance(secrets, Mapping) else {}
    base_url = cfg_section.get("base_url") or DEFAULT_BASE_URL
    api_key = cfg_section.get("api_key")
    timeout = cfg_section.get("timeout", DEFAULT_TIMEOUT)
    if not api_key:
        LOGGER.error("context_api.api_key missing from secrets; cannot configure context API")
        return
    config = ContextApiConfig(base_url=base_url, api_key=api_key, timeout=timeout)
    configure_context_api(config=config)


def register_project_configs(project_root: Mapping[str, Any]) -> None:
    """Register project configs so background threads avoid Streamlit secrets access."""
    if not isinstance(project_root, Mapping):
        return
    configs: Dict[str, Dict[str, Any]] = {}
    for sub_key, sub_val in project_root.items():
        if not isinstance(sub_val, Mapping):
            continue
        derived_key = make_project_key(str(sub_val.get("db_host", "")), str(sub_val.get("db_name", "")))
        project_key = f"project_data.{derived_key or sub_key}"
        configs[project_key] = dict(sub_val)
    with _CACHE_LOCK:
        _PROJECT_CONFIGS.clear()
        _PROJECT_CONFIGS.update(configs)


def set_default_project_key(project_key: Optional[str]) -> None:
    """Set the default project key used when none is explicitly provided."""
    global _DEFAULT_PROJECT_KEY
    _DEFAULT_PROJECT_KEY = project_key


def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _toast(message: str, icon: str, quiet: bool) -> None:
    if quiet:
        return
    if not _has_streamlit_context():
        return
    try:
        import streamlit as st

        st.toast(message, icon=icon)
    except Exception:
        return


def _get_cache() -> Dict[str, Dict[str, Any]]:
    with _CACHE_LOCK:
        return _CONTEXT_CACHE


def _store_payload(project_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cache = _get_cache()
    with _CACHE_LOCK:
        cache[project_key] = payload
    return payload


def _empty_payload() -> Dict[str, Any]:
    return dict(_DEFAULT_PAYLOAD)


def _summarize_context(value: Any, limit: int = 200) -> str:
    if value in (None, "", {}, []):
        return "<empty>"
    try:
        text = json.dumps(value)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        return f"{text[:limit]}…"
    return text


def _log_context_snapshot(
    project_label: str,
    payload: Optional[Dict[str, Any]],
    *,
    source: str,
    log_instrument: bool,
    log_project_ctx: bool,
) -> None:
    if not payload:
        return
    if log_instrument:
        LOGGER.info(
            "Instrument context (%s) for %s: %s",
            source,
            project_label,
            _summarize_context(payload.get("instrument_context")),
        )
    if log_project_ctx:
        LOGGER.info(
            "Project insights (%s) for %s: %s",
            source,
            project_label,
            _summarize_context(payload.get("project_specific_context")),
        )


def _parse_version(value: Any) -> datetime:
    if not value:
        return datetime.min
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.min


def _get_cached_project_list() -> Optional[List[Dict[str, Any]]]:
    with _CACHE_LOCK:
        return _PROJECT_LIST_CACHE if _PROJECT_LIST_CACHE else None


def _cache_project_list(projects: List[Dict[str, Any]]) -> None:
    global _PROJECT_LIST_CACHE
    with _CACHE_LOCK:
        _PROJECT_LIST_CACHE = projects


def _obtain_project_list(client: ContextAPIClient, quiet: bool) -> Optional[List[Dict[str, Any]]]:
    cached = _get_cached_project_list()
    if cached is not None:
        return cached
    try:
        projects = client.list_projects()
        if not isinstance(projects, list):
            raise ContextAPIError("Project list response was not a list")
        _cache_project_list(projects)
        return projects
    except ContextAPIError as exc:
        LOGGER.error("Failed to list context projects: %s", exc)
        _toast("Context API project list failed", ":material/error:", quiet)
        return None




def _get_api_client() -> Optional[ContextAPIClient]:
    global _CONTEXT_CLIENT
    if isinstance(_CONTEXT_CLIENT, ContextAPIClient):
        return _CONTEXT_CLIENT
    if _CONTEXT_CONFIG is not None:
        _CONTEXT_CLIENT = build_context_api_client(_CONTEXT_CONFIG)
        return _CONTEXT_CLIENT
    if _has_streamlit_context():
        try:
            import streamlit as st

            cfg_section = st.secrets.get("context_api", {})
            base_url = cfg_section.get("base_url") or DEFAULT_BASE_URL
            api_key = cfg_section.get("api_key")
            timeout = cfg_section.get("timeout", DEFAULT_TIMEOUT)
            if not api_key:
                LOGGER.error("context_api.api_key missing from secrets; cannot fetch project contexts")
                return None
            config = ContextApiConfig(base_url=base_url, api_key=api_key, timeout=timeout)
            _CONTEXT_CLIENT = build_context_api_client(config)
            return _CONTEXT_CLIENT
        except Exception as exc:
            LOGGER.error("Failed to build context API client from Streamlit secrets: %s", exc)
            return None
    LOGGER.error("Context API not configured; call configure_context_api_from_secrets() in main thread.")
    return None


def _get_project_config(project_key: str) -> Dict[str, Any]:
    if not project_key:
        return {}
    with _CACHE_LOCK:
        cached = _PROJECT_CONFIGS.get(project_key)
    if isinstance(cached, dict) and cached:
        return dict(cached)
    if _has_streamlit_context():
        try:
            from utils.project_selection import get_project_config

            return get_project_config(project_key)
        except Exception:
            return {}
    LOGGER.error("Project config not registered for key=%s; register_project_configs() is required.", project_key)
    return {}


def _select_project(projects: List[Dict[str, Any]], host: str, database_name: str) -> Optional[Dict[str, Any]]:
    matches = [
        p
        for p in projects
        if str(p.get("host")) == host
        and str(p.get("database") or p.get("database_name") or p.get("db_name") or p.get("project_name"))
        == database_name
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: _parse_version(item.get("latest_version")), reverse=True)
    return matches[0]


def _project_name(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("project_name", "name", "display_name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def ensure_project_context(
    project_key: Optional[str],
    *,
    force_refresh: bool = False,
    quiet: bool = False,
    strict: bool = False,
    fetch_instrument: bool = True,
    fetch_project_specific: bool = True,
    project_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not project_key:
        return _empty_payload()

    cache = _get_cache()
    cached_payload = cache.get(project_key)
    if cached_payload:
        cached_payload = dict(cached_payload)
        cached_payload.setdefault("_instrument_loaded", False)
        cached_payload.setdefault("_project_context_loaded", False)

    need_instrument = fetch_instrument
    need_project_ctx = fetch_project_specific

    cfg = project_config or _get_project_config(project_key) or {}
    host = cfg.get("db_host")
    db_name = cfg.get("db_name")
    display_name = cfg.get("display_name", project_key)

    if cached_payload and not force_refresh:
        has_instrument = not need_instrument or cached_payload.get("_instrument_loaded", False)
        has_project = not need_project_ctx or cached_payload.get("_project_context_loaded", False)
        if has_instrument and has_project:
            _log_context_snapshot(
                display_name,
                cached_payload,
                source="cache",
                log_instrument=need_instrument,
                log_project_ctx=need_project_ctx,
            )
            return cached_payload

    if not host or not db_name:
        LOGGER.error("Project config for %s is missing host or db_name", project_key)
        if cached_payload:
            _log_context_snapshot(
                display_name,
                cached_payload,
                source="cache-fallback",
                log_instrument=need_instrument,
                log_project_ctx=need_project_ctx,
            )
            return cached_payload
        return _store_payload(project_key, _empty_payload())

    client = _get_api_client()
    if client is None:
        _toast("Context API credentials missing; contexts unavailable", ":material/error:", quiet)
        if cached_payload and not strict:
            _log_context_snapshot(
                display_name,
                cached_payload,
                source="cache-fallback",
                log_instrument=need_instrument,
                log_project_ctx=need_project_ctx,
            )
            return cached_payload
        return _store_payload(project_key, _empty_payload())

    if not client.check_health():
        _toast("Context API unhealthy; using cached contexts", ":material/error:", quiet)
        if cached_payload and not strict:
            _log_context_snapshot(
                display_name,
                cached_payload,
                source="cache-fallback",
                log_instrument=need_instrument,
                log_project_ctx=need_project_ctx,
            )
            return cached_payload
        return _store_payload(project_key, _empty_payload())

    projects = _obtain_project_list(client, quiet)
    if not projects:
        if cached_payload and not strict:
            _log_context_snapshot(
                display_name,
                cached_payload,
                source="cache-fallback",
                log_instrument=need_instrument,
                log_project_ctx=need_project_ctx,
            )
            return cached_payload
        return _store_payload(project_key, _empty_payload())

    entry = _select_project(projects, host, db_name)
    if not entry:
        LOGGER.warning("No matching context entry for host=%s db=%s", host, db_name)
        _toast("No remote context data for this project", ":material/warning:", quiet)
        if cached_payload and not strict:
            _log_context_snapshot(
                display_name,
                cached_payload,
                source="cache-fallback",
                log_instrument=need_instrument,
                log_project_ctx=need_project_ctx,
            )
            return cached_payload
        return _store_payload(project_key, _empty_payload())

    project_name = _project_name(entry)
    payload = dict(cached_payload) if cached_payload else _empty_payload()
    payload.update({
        "project_name": project_name,
        "latest_version": entry.get("latest_version"),
    })

    payload.setdefault("_instrument_loaded", False)
    payload.setdefault("_project_context_loaded", False)

    if fetch_instrument:
        try:
            payload["instrument_context"] = client.fetch_instrument_context(host, db_name) or {}
            payload["_instrument_loaded"] = True
            _log_context_snapshot(
                display_name,
                payload,
                source="api",
                log_instrument=True,
                log_project_ctx=False,
            )
            if quiet:
                LOGGER.info("Instrument context refreshed for %s", display_name)
            else:
                _toast(f"Instrument context loaded for {display_name}", ":material/precision_manufacturing:", quiet)
        except ContextAPIError as exc:
            LOGGER.error("Instrument context fetch failed for db=%s: %s", db_name, exc)
            _toast("Instrument context unavailable; using fallback", ":material/error:", quiet)
    else:
        payload["_instrument_loaded"] = payload.get("_instrument_loaded", False)

    if fetch_project_specific:
        try:
            payload["project_specific_context"] = client.fetch_project_nl_context(host, db_name) or ""
            payload["_project_context_loaded"] = True
            _log_context_snapshot(
                display_name,
                payload,
                source="api",
                log_instrument=False,
                log_project_ctx=True,
            )
            if quiet:
                LOGGER.info("Project insights refreshed for %s", display_name)
            else:
                _toast(f"Project insights loaded for {display_name}", ":material/menu_book:", quiet)
        except ContextAPIError as exc:
            LOGGER.error("Project NL context fetch failed for db=%s: %s", db_name, exc)
            _toast("Project insights unavailable; using fallback", ":material/error:", quiet)
    else:
        payload["_project_context_loaded"] = payload.get("_project_context_loaded", False)

    return _store_payload(project_key, payload)


def get_instrument_context(project_key: Optional[str] = None) -> Dict[str, Any]:
    project_key = project_key or _DEFAULT_PROJECT_KEY
    if not project_key:
        return {}
    cache = _get_cache()
    payload = cache.get(project_key)
    if payload is None:
        payload = ensure_project_context(project_key, quiet=True, fetch_project_specific=False)
    return payload.get("instrument_context", {}) or {}


def get_project_specific_context(project_key: Optional[str] = None) -> Optional[str]:
    project_key = project_key or _DEFAULT_PROJECT_KEY
    if not project_key:
        return None
    cache = _get_cache()
    payload = cache.get(project_key)
    if payload is None:
        payload = ensure_project_context(project_key, quiet=True, fetch_instrument=False)
    context_value = payload.get("project_specific_context")
    if isinstance(context_value, str) and context_value.strip():
        return context_value.strip()
    return None
