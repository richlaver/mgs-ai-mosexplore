from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests


LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_URL = "http://54.151.25.237:8000"
DEFAULT_TIMEOUT = 15.0


class ContextAPIError(Exception):
    """Raised when the context API cannot fulfill a request."""


@dataclass(slots=True)
class ContextApiConfig:
    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT


class ContextAPIClient:
    """Thin HTTP client around the external context service."""

    def __init__(self, config: ContextApiConfig):
        if not config.api_key:
            raise ValueError("Context API client requires a non-empty API key")
        self._base_url = config.base_url.rstrip("/") or DEFAULT_BASE_URL
        self._api_key = config.api_key
        self._timeout = config.timeout or DEFAULT_TIMEOUT
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _headers(self, include_auth: bool = True) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if include_auth:
            headers["x-api-key"] = self._api_key
        return headers

    def check_health(self) -> bool:
        """Return True if the API reports a healthy status."""
        try:
            response = self._session.get(self._url("health"), timeout=self._timeout)
            response.raise_for_status()
            data = response.json() if response.content else {}
            status = (data or {}).get("status", "").lower()
            return status in {"ok", "healthy", "up"}
        except Exception as exc:  # noqa: BLE001 - log and treat as unhealthy
            LOGGER.warning("Context API health check failed: %s", exc)
            return False

    def list_projects(self) -> List[Dict[str, Any]]:
        """Fetch the list of available context projects."""
        try:
            response = self._session.get(
                self._url("api/v1/context/list"),
                headers=self._headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                projects = payload.get("projects") or payload.get("items") or payload.get("data")
                if isinstance(projects, list):
                    return projects
                if "project_name" in payload and "host" in payload:
                    return [payload]
                raise ContextAPIError("Unexpected project list envelope format")
            if isinstance(payload, list):
                return payload
            raise ContextAPIError("Project list response is neither list nor dict")
        except requests.HTTPError as exc:
            raise ContextAPIError(f"Context list request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ContextAPIError(f"Unable to parse project list: {exc}") from exc

    def fetch_instrument_context(self, db_name: str) -> Dict[str, Any]:
        """Retrieve the instrument context JSON for the given database."""
        safe_name = quote(db_name, safe="")
        headers = self._headers()
        LOGGER.debug(
            "Fetching instrument context for db=%s (encoded=%s); auth header=%s",
            db_name,
            safe_name,
            "present" if "x-api-key" in headers else "missing",
        )
        try:
            response = self._session.get(
                self._url(f"api/v1/context/{safe_name}/json"),
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
            if isinstance(data, dict):
                return data
            raise ContextAPIError("Instrument context response was not a JSON object")
        except requests.HTTPError as exc:
            raise ContextAPIError(f"Instrument context fetch failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ContextAPIError(f"Unable to parse instrument context: {exc}") from exc

    def fetch_project_nl_context(self, db_name: str) -> Optional[str]:
        """Retrieve the natural-language project context string for the database."""
        safe_name = quote(db_name, safe="")
        headers = self._headers()
        LOGGER.debug(
            "Fetching project NL context for db=%s (encoded=%s); auth header=%s",
            db_name,
            safe_name,
            "present" if "x-api-key" in headers else "missing",
        )
        try:
            response = self._session.get(
                self._url(f"api/v1/context/{safe_name}/nl-context"),
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
            if isinstance(data, dict):
                nl_context = data.get("nl_context")
                if nl_context is None:
                    LOGGER.warning("Project NL context response missing 'nl_context' key for db=%s", db_name)
                return nl_context
            elif isinstance(data, str):
                return data
            raise ContextAPIError("NL context response was not JSON")
        except requests.HTTPError as exc:
            raise ContextAPIError(f"Project NL context fetch failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ContextAPIError(f"Unable to parse NL context: {exc}") from exc


def build_context_api_client(config: Optional[ContextApiConfig]) -> ContextAPIClient:
    if config is None:
        raise ValueError("Context API configuration is missing")
    return ContextAPIClient(config)
