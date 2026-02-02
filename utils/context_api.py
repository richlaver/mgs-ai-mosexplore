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

    def _encode_query_value(self, value: str, *, encode_dots: bool = False) -> str:
        if value is None:
            return ""
        encoded = quote(str(value), safe="")
        if encode_dots:
            encoded = encoded.replace(".", "%2E")
        return encoded

    def _context_query_url(self, path: str, db_host: str, db_name: str) -> str:
        host_value = self._encode_query_value(db_host, encode_dots=True)
        name_value = self._encode_query_value(db_name)
        key_value = self._encode_query_value(self._api_key)
        query = f"db_host={host_value}&db_name={name_value}&api_key={key_value}"
        return f"{self._base_url.rstrip('/')}/{path.lstrip('/')}?{query}"

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

    def fetch_instrument_context(self, db_host: str, db_name: str) -> Dict[str, Any]:
        """Retrieve the instrument context JSON for the given database."""
        safe_host = self._encode_query_value(db_host, encode_dots=True)
        safe_name = self._encode_query_value(db_name)
        headers = self._headers(include_auth=False)
        LOGGER.debug(
            "Fetching instrument context for host=%s (encoded=%s) db=%s (encoded=%s)",
            db_host,
            safe_host,
            db_name,
            safe_name,
        )
        try:
            response = self._session.get(
                self._context_query_url("json", db_host, db_name),
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

    def fetch_project_nl_context(self, db_host: str, db_name: str) -> Optional[str]:
        """Retrieve the response-gen-info text that describes the project context."""
        safe_host = self._encode_query_value(db_host, encode_dots=True)
        safe_name = self._encode_query_value(db_name)
        headers = self._headers(include_auth=False)
        LOGGER.debug(
            "Fetching project response-gen-info for host=%s (encoded=%s) db=%s (encoded=%s)",
            db_host,
            safe_host,
            db_name,
            safe_name,
        )
        try:
            response = self._session.get(
                self._context_query_url("response-gen-info", db_host, db_name),
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
            if isinstance(data, dict):
                response_gen_info = data.get("response_gen_info")
                if response_gen_info is None:
                    LOGGER.warning(
                        "Response-gen-info payload missing 'response_gen_info' key for db=%s",
                        db_name,
                    )
                return response_gen_info
            elif isinstance(data, str):
                return data
            raise ContextAPIError("Response-gen-info response was not JSON")
        except requests.HTTPError as exc:
            raise ContextAPIError(f"Project response-gen-info fetch failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ContextAPIError(f"Unable to parse response-gen-info payload: {exc}") from exc


def build_context_api_client(config: Optional[ContextApiConfig]) -> ContextAPIClient:
    if config is None:
        raise ValueError("Context API configuration is missing")
    return ContextAPIClient(config)
