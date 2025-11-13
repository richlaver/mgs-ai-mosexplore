"""Set up dependencies and configuration for the MissionHelp Demo application.

This module initializes the LLM, embeddings, Qdrant vector store, and database,
ensuring all components are ready for the RAG pipeline.
"""

import json
import os
import b2sdk.v1 as b2
import psycopg2
import streamlit as st
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table, Column, Integer
from geoalchemy2 import Geometry
from parameters import include_tables, table_info
from collections import defaultdict
from typing import List, Tuple
import modal
import logging
from utils.project_selection import (
    get_selected_project_key,
    get_project_config,
)

logger = logging.getLogger(__name__)

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


def get_global_hierarchy_access(db: SQLDatabase) -> bool:
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


def get_llm() -> ChatVertexAI:
    """Initialize the Grok 3 Beta language model.

    Returns:
        A ChatOpenAI instance configured with xAI API.
    """
    st.toast("Setting up the Gemini 2.0 Flash LLM...", icon=":material/build:")
    return ChatVertexAI(
        model="gemini-2.0-flash-001",
        temperature = 0.1
        # model="gemini-2.5-pro-preview-05-06"        
    )


def get_db() -> SQLDatabase:
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
        db = SQLDatabase(
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

    Values are sourced from st.secrets['project_data.hanoi_clone'] mirroring get_db().
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