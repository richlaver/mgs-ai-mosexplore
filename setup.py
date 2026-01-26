"""Set up dependencies and configuration for the MissionHelp Demo application.

This module initializes the LLM, embeddings, Qdrant vector store, and database,
ensuring all components are ready for the RAG pipeline.
"""

import ast
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import b2sdk.v1 as b2
import psycopg2
import requests
import streamlit as st
from google.auth import default as google_auth_default
from google.auth.transport.requests import AuthorizedSession, Request
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import SQLDatabase as BaseSQLDatabase
from utils.cancelable_llm import InterruptibleChatVertexAI, clone_llm as clone_interruptible_llm
from utils.cancelable_sql import CancelableSQLDatabase
from sqlalchemy import Column, Integer, MetaData, Table, create_engine
from geoalchemy2 import Geometry
from parameters import include_tables, table_info
from collections import defaultdict
from typing import Any, List, Tuple, Dict, Union, Optional
import modal
import logging
from utils.project_selection import (
    get_selected_project_key,
    get_project_config,
)
from utils.context_data import get_instrument_context

logger = logging.getLogger(__name__)
_map_spatial_defaults_cache: Optional[Dict[str, float]] = None

def _normalize_api_endpoint(raw: str) -> str:
    if not raw:
        return ""
    return raw.replace("https://", "").replace("http://", "").rstrip("/")


VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
VERTEX_ENDPOINT = _normalize_api_endpoint(
    os.environ.get(
        "VERTEX_ENDPOINT",
        "aiplatform.googleapis.com"
        if VERTEX_LOCATION == "global"
        else f"{VERTEX_LOCATION}-aiplatform.googleapis.com",
    )
)

_CACHED_CONTENT_IDS: Dict[str, str] = {}
_CACHED_CONTENT_HASHES: Dict[str, str] = {}


@dataclass
class VertexConfig:
    project_id: str
    location: str
    model_id: str
    api_endpoint: str

    @property
    def model_resource(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/publishers/google/models/{self.model_id}"
        )

    @property
    def base_url(self) -> str:
        return f"https://{self.api_endpoint}/v1"


def _get_authed_session() -> AuthorizedSession:
    credentials, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not credentials.valid:
        logger.info("Refreshing Google credentials for Vertex cache")
        credentials.refresh(Request())
    return AuthorizedSession(credentials)


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def _get_llm_model_id(llm: Any, fallback: str) -> str:
    for attr in ("model_name", "model", "model_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return fallback


def _get_vertex_config_for_cache(llm: Any) -> VertexConfig:
    set_google_credentials()
    project_id = get_project_id()
    location = os.environ.get("VERTEX_LOCATION", VERTEX_LOCATION)
    api_endpoint = _normalize_api_endpoint(os.environ.get("VERTEX_ENDPOINT", VERTEX_ENDPOINT))
    model_id = _get_llm_model_id(llm, "gemini-2.0-flash-lite")
    return VertexConfig(
        project_id=project_id,
        location=location,
        model_id=model_id,
        api_endpoint=api_endpoint,
    )


def _load_cached_prompt_template() -> str:
    prompt_path = Path(__file__).resolve().parent / "cached_llm_content" / "codeact_coder_prompt_static.md"
    return prompt_path.read_text(encoding="utf-8")


def _load_instrument_selection_template() -> str:
    prompt_path = Path(__file__).resolve().parent / "cached_llm_content" / "instrument_selection_prompt_static.md"
    return prompt_path.read_text(encoding="utf-8")


def normalize_instrument_context(raw_data: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normalize instrument context to a TYPE_SUBTYPE keyed dict."""
    if not raw_data:
        return {}

    data = raw_data
    if isinstance(raw_data, dict) and isinstance(raw_data.get("instrument_types"), dict):
        data = raw_data["instrument_types"]

    normalized: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return normalized

    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue

        instr_type = entry.get("type")
        instr_subtype = entry.get("subtype")

        if (not instr_type or not instr_subtype) and isinstance(key, str):
            if "/" in key:
                parts = key.split("/", 1)
            elif "_" in key:
                parts = key.split("_", 1)
            else:
                parts = []

            if len(parts) == 2:
                instr_type = instr_type or parts[0]
                instr_subtype = instr_subtype or parts[1]

        if isinstance(instr_type, str) and isinstance(instr_subtype, str):
            entry = {**entry, "type": instr_type, "subtype": instr_subtype}
            normalized[f"{instr_type}_{instr_subtype}"] = entry
        else:
            normalized[str(key)] = entry

    return normalized


def build_instrument_search_context(instrument_data: Dict[str, Any]) -> str:
    """Build searchable instrument context string."""
    if not instrument_data:
        return ""
    context_parts: List[str] = []
    if isinstance(instrument_data, dict):
        context_parts.append("AVAILABLE INSTRUMENTS:")
        type_groups: Dict[str, List[tuple]] = {}
        for instrument_key, instrument_info in instrument_data.items():
            if isinstance(instrument_info, dict):
                instrument_type = instrument_info.get("type", "UNKNOWN")
                type_groups.setdefault(instrument_type, []).append((instrument_key, instrument_info))
        for instrument_type, instruments in sorted(type_groups.items()):
            context_parts.append(f"\n{instrument_type} INSTRUMENTS:")
            for instrument_key, instrument_info in instruments:
                name = instrument_info.get("name", "Unknown")
                subtype = instrument_info.get("subtype", "DEFAULT")
                purpose_raw = instrument_info.get("purpose", "")
                purpose_snippet = (
                    purpose_raw[:250] + "..." if isinstance(purpose_raw, str) and len(purpose_raw) > 250 else purpose_raw
                )
                fields = instrument_info.get("fields", [])
                fields_to_show = fields if len(fields) < 10 else fields[:10]
                fields_snippet = ", ".join(
                    f"{f.get('database_field_name', '')} ({f.get('unit', f.get('units', ''))}): "
                    f"{', '.join(f.get('common_names', []))} - "
                    f"{f.get('description', '') + ('...' if len(f.get('description', '')) > 200 else '')}"
                    for f in fields_to_show
                )
                context_parts.append(
                    f"  - {instrument_key} ({name} - {subtype}): {purpose_snippet}\n    Fields: {fields_snippet}"
                )
    return "\n".join(context_parts)


def build_codeact_coder_cached_context(tools: List[Any]) -> str:
    tools_str = _build_tools_str(tools)
    template = _load_cached_prompt_template()
    return template.replace("<<TOOLS_STR>>", tools_str)


def build_instrument_selection_cached_context(context_text: str) -> str:
    template = _load_instrument_selection_template()
    return template.replace("<<INSTRUMENT_CONTEXT>>", context_text)


def refresh_instrument_selection_cache(selected_project_key: Optional[str], llm: Any) -> Optional[str]:
    """Refresh instrument selection cached content for the selected project."""
    instrument_payload = get_instrument_context(selected_project_key)
    instrument_data = normalize_instrument_context(instrument_payload)
    context_text = build_instrument_search_context(instrument_data)
    cached_context = build_instrument_selection_cached_context(context_text)
    return ensure_cached_content(
        cache_key="instrument_selection",
        content_text=cached_context,
        llm=llm,
        display_prefix="instrument-selection-cache",
        legacy_hash_keys=["instrument_selection_cached_context_hash"],
    )


def _build_tools_str(tools: List[Any]) -> str:
    return json.dumps(
        [{"name": t.name, "description": t.description} for t in tools],
        indent=2,
    ) or "[]"


def _create_cached_content(
    session: AuthorizedSession,
    config: VertexConfig,
    context: str,
    display_prefix: str = "codeact-coder-cache",
) -> str:
    display_name = f"{display_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "displayName": display_name,
        "model": config.model_resource,
        "contents": [{"role": "user", "parts": [{"text": context}]}],
        "ttl": "3600s",
    }
    url = f"{config.base_url}/projects/{config.project_id}/locations/{config.location}/cachedContents"
    logger.info("Creating cached content: %s", display_name)
    logger.info("cachedContents.create POST %s", url)
    resp = session.post(url, json=payload, timeout=20)
    data = _safe_json(resp)
    logger.info("cachedContents.create status=%s", resp.status_code)
    if resp.status_code >= 400:
        logger.error("cachedContents.create error: %s", json.dumps(data, ensure_ascii=False))
        resp.raise_for_status()
    cached_name = data.get("name")
    usage = data.get("usageMetadata", {})
    logger.info("Cached content name=%s usage=%s", cached_name, usage)
    if not cached_name:
        raise RuntimeError("Cached content creation returned no name")
    return cached_name


def ensure_cached_content(
    cache_key: str,
    content_text: str,
    llm: Any,
    display_prefix: str,
    legacy_hash_keys: Optional[List[str]] = None,
) -> Optional[str]:
    """Create or reuse cached content for a given cache key and content."""
    global _CACHED_CONTENT_IDS, _CACHED_CONTENT_HASHES

    content_hash = hashlib.sha256((content_text or "").encode("utf-8")).hexdigest()
    id_key = f"{cache_key}_cached_content_id"
    hash_key = f"{cache_key}_cached_content_hash"
    legacy_keys = list(legacy_hash_keys or [])

    cached_id = None
    cached_hash = None
    try:
        cached_id = st.session_state.get(id_key)
        cached_hash = st.session_state.get(hash_key)
        if cached_hash is None:
            for legacy_key in legacy_keys:
                cached_hash = st.session_state.get(legacy_key)
                if cached_hash is not None:
                    break
    except Exception:
        pass

    if cached_id and cached_hash == content_hash:
        _CACHED_CONTENT_IDS[cache_key] = cached_id
        _CACHED_CONTENT_HASHES[cache_key] = cached_hash
        return cached_id

    if _CACHED_CONTENT_IDS.get(cache_key) and _CACHED_CONTENT_HASHES.get(cache_key) == content_hash:
        return _CACHED_CONTENT_IDS[cache_key]

    config = _get_vertex_config_for_cache(llm)
    session = _get_authed_session()
    cached_name = _create_cached_content(session, config, content_text, display_prefix=display_prefix)
    cached_id = cached_name.split("/")[-1]

    _CACHED_CONTENT_IDS[cache_key] = cached_id
    _CACHED_CONTENT_HASHES[cache_key] = content_hash
    try:
        st.session_state[id_key] = cached_id
        st.session_state[hash_key] = content_hash
    except Exception:
        pass
    return cached_id


def get_cached_content_id(cache_key: str) -> Optional[str]:
    global _CACHED_CONTENT_IDS
    cached_id = _CACHED_CONTENT_IDS.get(cache_key)
    if cached_id:
        return cached_id
    id_key = f"{cache_key}_cached_content_id"
    try:
        cached_id = st.session_state.get(id_key)
        if cached_id:
            _CACHED_CONTENT_IDS[cache_key] = cached_id
            return cached_id
    except Exception:
        pass
    return None

def set_modal_credentials():
    """Set Modal credentials from Streamlit secrets."""
    st.toast("Setting Modal credentials...", icon=":material/key:")
    os.environ["MODAL_TOKEN_ID"] = st.secrets["modal"]["token_id"]
    os.environ["MODAL_TOKEN_SECRET"] = st.secrets["modal"]["token_secret"]
    logger.info("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables")

def build_modal_secrets():
    """Build Modal secrets.

    When running locally (modal.is_local()), construct secrets from Streamlit's st.secrets.
    When remote, reference named secrets pre-configured in Modal. Returns list of Secret objects.
    """
    # Ensure basic logging exists
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger.info("Building Modal secrets (modal.is_local()=%s)", modal.is_local())
    if modal.is_local():
        import streamlit as st
        try:
            st.toast("Building Modal secrets...", icon=":material:key:")
        except Exception:
            pass

        current_project_key = get_selected_project_key() if "selected_project_key" in st.session_state else None
        if st.session_state.get("modal_secrets") and st.session_state.get("modal_secrets_project_key") == current_project_key:
            return st.session_state.modal_secrets

        google_json = st.secrets.get("GOOGLE_CREDENTIALS_JSON")
        selected_project_key = get_selected_project_key()
        db_dict = get_project_config(selected_project_key)
        rds_dict = st.secrets.get("artefact_metadata_rds", {})
        b2_dict = st.secrets.get("artefact_blob_b2", {})
        logger.info("google_json present: %s", bool(google_json))
        logger.info("db_dict keys: %s", sorted(list(db_dict.keys())))
        logger.info("artefact_metadata_rds_dict keys: %s", sorted(list(rds_dict.keys())))
        logger.info("artefact_blob_b2_dict keys: %s", sorted(list(b2_dict.keys())))

        if not google_json:
            logger.error("st.secrets['GOOGLE_CREDENTIALS_JSON'] missing or empty")
            raise ValueError("st.secrets['GOOGLE_CREDENTIALS_JSON'] is missing or empty.")

        required_db_keys = ["db_host", "db_user", "db_pass", "db_name"]
        missing = [k for k in required_db_keys if not db_dict.get(k)]
        if missing:
            logger.error("Selected project '%s' missing required keys: %s", selected_project_key, ", ".join(missing))
            raise ValueError(
                f"st.secrets['{selected_project_key}'] is missing required keys: " + ", ".join(missing)
            )
        required_rds_keys = ["host", "user", "password", "database"]
        missing_rds = [k for k in required_rds_keys if not rds_dict.get(k)]
        if missing_rds:
            logger.error("st.secrets['artefact_metadata_rds'] missing required keys: %s", ", ".join(missing_rds))
            raise ValueError("st.secrets['artefact_metadata_rds'] is missing required keys: " + ", ".join(missing_rds))
        required_b2_keys = ["key_id", "key", "bucket"]
        missing_b2 = [k for k in required_b2_keys if not b2_dict.get(k)]
        if missing_b2:
            logger.error("st.secrets['artefact_blob_b2'] missing required keys: %s", ", ".join(missing_b2))
            raise ValueError("st.secrets['artefact_blob_b2'] is missing required keys: " + ", ".join(missing_b2))

        google_secret = modal.Secret.from_dict({
            "GOOGLE_CREDENTIALS_JSON": str(google_json),
        })

        project_data_raw = st.secrets.get("project_data", {})
        project_data_json = json.dumps(
            project_data_raw,
            default=lambda o: dict(o) if hasattr(o, "items") else o,
        )
        project_data_secret = modal.Secret.from_dict({
            "PROJECT_DATA_JSON": project_data_json,
        })

        mysql_secret = modal.Secret.from_dict({
            "DB_HOST": str(db_dict.get("db_host")),
            "DB_USER": str(db_dict.get("db_user")),
            "DB_PASS": str(db_dict.get("db_pass")),
            "DB_NAME": str(db_dict.get("db_name")),
            "DB_PORT": str(db_dict.get("port", "3306")),
        })
        rds_secret = modal.Secret.from_dict({
            "ARTEFACT_METADATA_RDS_HOST": str(rds_dict.get("host")),
            "ARTEFACT_METADATA_RDS_USER": str(rds_dict.get("user")),
            "ARTEFACT_METADATA_RDS_PASS": str(rds_dict.get("password")),
            "ARTEFACT_METADATA_RDS_DATABASE": str(rds_dict.get("database")),
            "ARTEFACT_METADATA_RDS_PORT": str(rds_dict.get("port", "5432")),
        })
        b2_secret = modal.Secret.from_dict({
            "ARTEFACT_BLOB_B2_KEY_ID": str(b2_dict.get("key_id")),
            "ARTEFACT_BLOB_B2_KEY": str(b2_dict.get("key")),
            "ARTEFACT_BLOB_B2_BUCKET": str(b2_dict.get("bucket")),
        })
        logger.info(
            "Created local Modal secrets from st.secrets for project '%s' (DB_HOST/USER/NAME set: %s/%s/%s, DB_PORT=%s)",
            selected_project_key,
            bool(db_dict.get("db_host")),
            bool(db_dict.get("db_user")),
            bool(db_dict.get("db_name")),
            str(db_dict.get("port", "3306")),
        )
        logger.info("Created local Modal secrets for artefact_metadata_rds and artefact_blob_b2")
        st.session_state.modal_secrets = [google_secret, project_data_secret, mysql_secret, rds_secret, b2_secret]
        st.session_state.modal_secrets_project_key = selected_project_key
        return st.session_state.modal_secrets
    else:
        logger.info(
            "Using named Modal secrets: 'google-credentials-json', 'project-data-json', 'mysql-credentials', 'artefact-metadata-rds-credentials', 'artefact-blob-b2-credentials'"
        )
        return [
            modal.Secret.from_name("google-credentials-json"),
            modal.Secret.from_name("project-data-json"),
            modal.Secret.from_name("mysql-credentials"),
            modal.Secret.from_name("artefact-metadata-rds-credentials"),
            modal.Secret.from_name("artefact-blob-b2-credentials"),
        ]


def enable_tracing():
    """Enables LangSmith tracing."""
    os.environ['LANGSMITH_TRACING'] = st.secrets['LANGSMITH_TRACING']
    os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']


def build_relationship_graph(table_info=table_info) -> defaultdict[str, List[Tuple]]:
    """Build a relationship graph from table_info."""
    st.toast("Building table relationship graph...", icon=":material/account_tree:")
    graph = defaultdict(list)
    for table in table_info:
        table_name = table['name']
        for rel in table.get('relationships', []):
            graph[table_name].append(
                (rel['referenced_table'], rel['column'], rel['referenced_column'])
            )
    return graph


def get_global_hierarchy_access(db: BaseSQLDatabase) -> bool:
    """Check if the user has global hierarchy access."""
    st.toast("Checking global hierarchy access...", icon=":material/lock_open:")
    query = """
            SELECT uagu.group_id
            FROM user_access_groups_users uagu
            RIGHT JOIN geo_12_users gu ON uagu.user_id = gu.id
            WHERE gu.id = %s
        """ % st.session_state.selected_user_id
    result = db.run(query)
    if not result:
        logging.warning("No user access groups found for user ID %s", st.session_state.selected_user_id)
        return False
    
    parsed_result = eval(result)
    row = parsed_result[0]
    logging.debug(f"Global hierarchy access check for user ID {st.session_state.selected_user_id}: {row[0]}")
    return row[0] == 0


def set_google_credentials() -> None:
    """Set Google Cloud credentials for database access.

    Writes credentials from secrets to a temporary file and sets the environment variable.
    """
    try:
        st.toast("Setting Google credentials...", icon=":material/build:")
    except Exception:
        # Allows usage in non-Streamlit contexts (scripts, tests, CI).
        pass
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    temp_file_path = "google_credentials.json"
    with open(temp_file_path, "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path


def get_llms() -> Dict[str, ChatVertexAI]:
    """Initialize the Gemini language models.

    Returns:
        A dictionary of language model instances configured with Google APIs.
    """
    st.toast("Setting up Gemini LLMs...", icon=":material/build:")
    common_kwargs = {
        "location": VERTEX_LOCATION,
        "api_endpoint": VERTEX_ENDPOINT,
    }
    return {
        "FAST": InterruptibleChatVertexAI(
            model="gemini-2.0-flash-lite",
            temperature=0.3,
            **common_kwargs,
        ),
        "BALANCED": InterruptibleChatVertexAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            **common_kwargs,
        ),
        "THINKING": InterruptibleChatVertexAI(
            model="gemini-2.5-pro",
            temperature=0.7,
            **common_kwargs,
        ),
    }


def get_llm(model: str = "FAST") -> ChatVertexAI:
    """Backward compatible alias returning a single LLM by key."""

    llms = get_llms()
    key = (model or "FAST").upper()
    return llms.get(key, llms["FAST"])


def clone_llm_with_overrides(llm: ChatVertexAI, **overrides) -> ChatVertexAI:
    """Clone an LLM while preserving cancellation awareness."""

    return clone_interruptible_llm(llm, **overrides)


def get_db() -> BaseSQLDatabase:
    """Initialize the SQL database connection.

    Returns:
        An SQLDatabase instance connected to the MissionOS Hanoi CP03 database.
    """
    st.toast("Connecting to the selected MissionOS database...", icon=":material/build:")
    try:
        selected_project_key = get_selected_project_key()
        project_cfg = get_project_config(selected_project_key)
        db_host = project_cfg["db_host"]
        db_user = project_cfg["db_user"]
        db_pass = project_cfg["db_pass"]
        db_name = project_cfg["db_name"]
        port = project_cfg["port"]

        logger.info("[DB] Connecting using project=%s host=%s db=%s port=%s user_present=%s", selected_project_key, db_host, db_name, port, bool(db_user))
        db_uri_actual = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{port}/{db_name}"
        db_uri_log = f"mysql+mysqlconnector://{db_user}:***@{db_host}:{port}/{db_name}"
        logger.debug("[DB] SQLAlchemy URI (masked): %s", db_uri_log)
        engine = create_engine(
            url=db_uri_actual,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            echo=False
        )

        # Define metadata with custom Geometry type
        metadata = MetaData()
        Table('3d_condours', metadata,
              Column('id', Integer, primary_key=True),
              Column('contour_bound', Geometry)
        )

        # Initialize SQLDatabase with custom metadata
        db = CancelableSQLDatabase(
            engine=engine,
            metadata=metadata,
            include_tables=include_tables,
            # custom_table_info=custom_table_info,
            sample_rows_in_table_info=3,
            lazy_table_reflection=True
        )
        logging.debug(f"Available tables: {db.get_usable_table_names()}")

        st.toast(
            f"Connected to: {project_cfg.get('display_name', selected_project_key)}",
            icon=":material/plug_connect:",
        )
        return db
    except Exception as e:
        raise Exception(f"Failed to connect to database: {str(e)}")


def _clean_numeric_value(val: Union[str, bytes, bytearray, float, int, None]) -> Optional[float]:
    """Normalize numeric values returned from SQL into floats."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    if isinstance(val, (bytes, bytearray)):
        try:
            val = val.decode("utf-8")  # type: ignore
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if len(s) >= 2 and ((s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'"))):
            s = s[1:-1]
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Return percentile using linear interpolation (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_vals[int(pos)]
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * (pos - lower)


def compute_map_spatial_defaults(db: BaseSQLDatabase) -> Dict[str, float]:
    """Compute median instrument center and a radius covering ~90% of instruments."""
    query = (
        """
        SELECT l.easting, l.northing
        FROM instrum i
        JOIN location l ON i.location_id = l.id
        WHERE l.easting IS NOT NULL AND l.northing IS NOT NULL;
        """
    )
    try:
        raw = db.run(query)
    except Exception as exc:
        logger.warning("Failed to query spatial defaults: %s", exc)
        return {
            "median_easting": 0.0,
            "median_northing": 0.0,
            "radius_90_extent": 500.0,
        }

    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                logger.warning("Could not parse spatial defaults response; using fallbacks")
                parsed = []

    eastings: List[float] = []
    northings: List[float] = []
    for row in parsed or []:
        if isinstance(row, dict):
            e_val = row.get("easting") or row.get("l.easting") or row.get("EASTING")
            n_val = row.get("northing") or row.get("l.northing") or row.get("NORTHING")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            e_val, n_val = row[0], row[1]
        else:
            continue
        e_clean = _clean_numeric_value(e_val)
        n_clean = _clean_numeric_value(n_val)
        if e_clean is None or n_clean is None:
            continue
        eastings.append(e_clean)
        northings.append(n_clean)

    if not eastings or not northings:
        return {
            "median_easting": 0.0,
            "median_northing": 0.0,
            "radius_90_extent": 500.0,
        }

    eastings_sorted = sorted(eastings)
    northings_sorted = sorted(northings)
    med_e = statistics.median(eastings_sorted)
    med_n = statistics.median(northings_sorted)
    p5_e = _percentile(eastings_sorted, 0.05)
    p95_e = _percentile(eastings_sorted, 0.95)
    p5_n = _percentile(northings_sorted, 0.05)
    p95_n = _percentile(northings_sorted, 0.95)
    radius = max(p95_e - p5_e, p95_n - p5_n) / 2.0
    if not math.isfinite(radius) or radius <= 0:
        radius = 500.0

    return {
        "median_easting": float(med_e),
        "median_northing": float(med_n),
        "radius_90_extent": float(radius),
    }


def get_map_spatial_defaults(db: BaseSQLDatabase) -> Dict[str, float]:
    """Return cached map spatial defaults, computing and storing if needed."""
    global _map_spatial_defaults_cache
    if _map_spatial_defaults_cache is not None:
        return _map_spatial_defaults_cache

    defaults = compute_map_spatial_defaults(db)
    _map_spatial_defaults_cache = defaults
    try:
        if hasattr(st, "session_state"):
            st.session_state.map_spatial_defaults = defaults
    except Exception:
        pass
    return defaults



def get_project_id() -> str:
    """Parse and return the Google project_id from GOOGLE_CREDENTIALS_JSON."""
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    project_id = json.loads(credentials_json).get("project_id")
    if not project_id:
        raise ValueError("project_id not found in GOOGLE_CREDENTIALS_JSON")
    logging.info("Parsed project_id: %s", project_id)
    return project_id


def set_metadata_db_env() -> None:
    """Populate environment variables for the artefact metadata PostgreSQL RDS.

    Sets:
        ARTEFACT_METADATA_RDS_HOST
        ARTEFACT_METADATA_RDS_USER
        ARTEFACT_METADATA_RDS_PASS
        ARTEFACT_METADATA_RDS_DATABASE
        ARTEFACT_METADATA_RDS_PORT (default 5432 if absent)

    Values are taken from st.secrets['artefact_metadata_rds'] matching logic in get_metadata_db()
    and the variable names expected inside the Modal sandbox (see modal_sandbox.py).
    """
    st.toast("Setting metadata DB environment variables...", icon=":material/package_2:")
    rds_dict = st.secrets.get("artefact_metadata_rds", {})
    required_keys = ["host", "user", "password", "database"]
    missing = [k for k in required_keys if not rds_dict.get(k)]
    if missing:
        logging.error("artefact_metadata_rds missing required keys: %s", ", ".join(missing))
        raise ValueError("st.secrets['artefact_metadata_rds'] is missing required keys: " + ", ".join(missing))

    os.environ["ARTEFACT_METADATA_RDS_HOST"] = str(rds_dict.get("host"))
    os.environ["ARTEFACT_METADATA_RDS_USER"] = str(rds_dict.get("user"))
    os.environ["ARTEFACT_METADATA_RDS_PASS"] = str(rds_dict.get("password"))
    os.environ["ARTEFACT_METADATA_RDS_DATABASE"] = str(rds_dict.get("database"))
    os.environ["ARTEFACT_METADATA_RDS_PORT"] = str(rds_dict.get("port", 5432))
    logging.info(
        "Set ARTEFACT_METADATA_RDS_* env vars (host=%s db=%s user=%s port=%s)",
        os.environ["ARTEFACT_METADATA_RDS_HOST"],
        os.environ["ARTEFACT_METADATA_RDS_DATABASE"],
        os.environ["ARTEFACT_METADATA_RDS_USER"],
        os.environ["ARTEFACT_METADATA_RDS_PORT"],
    )


def set_blob_db_env() -> None:
    """Populate environment variables for Backblaze B2 artefact blob storage.

    Sets:
        ARTEFACT_BLOB_B2_KEY_ID
        ARTEFACT_BLOB_B2_KEY
        ARTEFACT_BLOB_B2_BUCKET

    Values are sourced from st.secrets['artefact_blob_b2'] mirroring get_blob_db()
    and names expected in the Modal sandbox (see modal_sandbox.py).
    """
    st.toast("Setting blob storage environment variables...", icon=":material/package_2:")
    b2_dict = st.secrets.get("artefact_blob_b2", {})
    required_keys = ["key_id", "key", "bucket"]
    missing = [k for k in required_keys if not b2_dict.get(k)]
    if missing:
        logging.error("artefact_blob_b2 missing required keys: %s", ", ".join(missing))
        raise ValueError("st.secrets['artefact_blob_b2'] is missing required keys: " + ", ".join(missing))

    os.environ["ARTEFACT_BLOB_B2_KEY_ID"] = str(b2_dict.get("key_id"))
    os.environ["ARTEFACT_BLOB_B2_KEY"] = str(b2_dict.get("key"))
    os.environ["ARTEFACT_BLOB_B2_BUCKET"] = str(b2_dict.get("bucket"))
    logging.info(
        "Set ARTEFACT_BLOB_B2_* env vars (bucket=%s key_id_present=%s)",
        os.environ["ARTEFACT_BLOB_B2_BUCKET"],
        bool(os.environ.get("ARTEFACT_BLOB_B2_KEY_ID")),
    )


def set_db_env() -> None:
    """Populate environment variables for the primary MySQL database.

    Sets (matching names consumed in modal_sandbox.py):
        DB_HOST
        DB_USER
        DB_PASS
        DB_NAME
        DB_PORT (defaults to 3306 if not provided)

    Values are sourced from st.secrets['project_data'] entry matching the selected project key.
    """
    st.toast("Setting primary DB environment variables...", icon=":material/package_2:")
    selected_project_key = get_selected_project_key()
    db_dict = get_project_config(selected_project_key)
    required_keys = ["db_host", "db_user", "db_pass", "db_name"]
    missing = [k for k in required_keys if not db_dict.get(k)]
    if missing:
        logging.error("database secrets for '%s' missing required keys: %s", selected_project_key, ", ".join(missing))
        raise ValueError(f"st.secrets['{selected_project_key}'] is missing required keys: " + ", ".join(missing))

    os.environ["DB_HOST"] = str(db_dict.get("db_host"))
    os.environ["DB_USER"] = str(db_dict.get("db_user"))
    os.environ["DB_PASS"] = str(db_dict.get("db_pass"))
    os.environ["DB_NAME"] = str(db_dict.get("db_name"))
    os.environ["DB_PORT"] = str(db_dict.get("port", 3306))
    logging.info(
        "Set DB_* env vars for '%s' (host=%s db=%s user=%s port=%s)",
        selected_project_key,
        os.environ["DB_HOST"],
        os.environ["DB_NAME"],
        os.environ["DB_USER"],
        os.environ["DB_PORT"],
    )
    

def get_metadata_db() -> psycopg2.extensions.connection:
    """Initialize the PostgreSQL RDS connection for artefact metadata."""
    st.toast("Connecting to PostgreSQL RDS for metadata...", icon=":material/page_info:")
    try:
        host = st.secrets["artefact_metadata_rds"]["host"]
        user = st.secrets["artefact_metadata_rds"]["user"]
        password = st.secrets["artefact_metadata_rds"]["password"]
        database = st.secrets["artefact_metadata_rds"]["database"]
        port = st.secrets["artefact_metadata_rds"].get("port", 5432)

        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        logging.info("Connected to PostgreSQL RDS metadata_db")
        return conn
    except Exception as e:
        raise Exception(f"Failed to connect to metadata database: {str(e)}")


def get_blob_db() -> b2.Bucket:
    """Initialize the Backblaze B2 bucket instance."""
    st.toast("Setting up Backblaze B2 bucket...", icon=":material/glass_cup:")
    try:
        key_id = st.secrets["artefact_blob_b2"]["key_id"]
        key = st.secrets["artefact_blob_b2"]["key"]
        bucket_name = st.secrets["artefact_blob_b2"]["bucket"]

        info = b2.InMemoryAccountInfo()
        b2_api = b2.B2Api(info)
        b2_api.authorize_account("production", key_id, key)
        bucket = b2_api.get_bucket_by_name(bucket_name)
        logging.info("Authorized and got B2 bucket: %s", bucket_name)
        return bucket
    except Exception as e:
        raise Exception(f"Failed to set up Backblaze B2: {str(e)}")