from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from utils.project_selection import get_project_config
from utils.context_api import (
    ContextAPIClient,
    ContextAPIError,
    ContextApiConfig,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    build_context_api_client,
)

LOGGER = logging.getLogger(__name__)
_CACHE_KEY = "project_context_cache"
_CLIENT_KEY = "context_api_client"
_PROJECT_LIST_KEY = "context_api_project_list"
_DEFAULT_PAYLOAD = {
    "instrument_context": {},
    "project_specific_context": "",
    "project_name": None,
    "latest_version": None,
    "_instrument_loaded": False,
    "_project_context_loaded": False,
}


def _toast(message: str, icon: str, quiet: bool) -> None:
    if not quiet:
        st.toast(message, icon=icon)


def _get_cache() -> Dict[str, Dict[str, Any]]:
    cache = st.session_state.setdefault(_CACHE_KEY, {})
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_CACHE_KEY] = cache
    return cache


def _store_payload(project_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cache = _get_cache()
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
    projects = st.session_state.get(_PROJECT_LIST_KEY)
    if isinstance(projects, list) and projects:
        return projects
    return None


def _cache_project_list(projects: List[Dict[str, Any]]) -> None:
    st.session_state[_PROJECT_LIST_KEY] = projects


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
    client = st.session_state.get(_CLIENT_KEY)
    if isinstance(client, ContextAPIClient):
        return client
    cfg_section = st.secrets.get("context_api", {})
    base_url = cfg_section.get("base_url") or DEFAULT_BASE_URL
    api_key = cfg_section.get("api_key")
    timeout = cfg_section.get("timeout", DEFAULT_TIMEOUT)
    if not api_key:
        LOGGER.error("context_api.api_key missing from secrets; cannot fetch project contexts")
        return None
    config = ContextApiConfig(base_url=base_url, api_key=api_key, timeout=timeout)
    client = build_context_api_client(config)
    st.session_state[_CLIENT_KEY] = client
    return client


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

    cfg = get_project_config(project_key) or {}
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
    project_key = project_key or st.session_state.get("selected_project_key")
    if not project_key:
        return {}
    cache = _get_cache()
    payload = cache.get(project_key)
    if payload is None:
        payload = ensure_project_context(project_key, quiet=True, fetch_project_specific=False)
    return payload.get("instrument_context", {}) or {}


def get_project_specific_context(project_key: Optional[str] = None) -> Optional[str]:
    project_key = project_key or st.session_state.get("selected_project_key")
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
