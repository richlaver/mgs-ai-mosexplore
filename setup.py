"""Set up dependencies and configuration for the MissionHelp Demo application.

This module initializes the LLM, embeddings, Qdrant vector store, and database,
ensuring all components are ready for the RAG pipeline.
"""

import ast
import json
import math
import os
import statistics
import b2sdk.v1 as b2
import psycopg2
import streamlit as st
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import SQLDatabase as BaseSQLDatabase
from utils.cancelable_llm import InterruptibleChatVertexAI, clone_llm as clone_interruptible_llm
from utils.cancelable_sql import CancelableSQLDatabase
from sqlalchemy import Column, Integer, MetaData, Table, create_engine
from geoalchemy2 import Geometry
from parameters import include_tables, table_info
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional
import modal
import logging
from utils.project_selection import (
    get_selected_project_key,
    get_project_config,
)

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
    st.toast("Setting Google credentials...", icon=":material/build:")
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