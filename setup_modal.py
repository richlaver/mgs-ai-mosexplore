"""Modal-specific setup helpers.

Keep this module lightweight so Modal sandbox images don't pull in
Streamlit-only or heavy app dependencies unnecessarily.
"""

from __future__ import annotations

import json
import logging
import os

import modal

logger = logging.getLogger(__name__)


def set_modal_credentials() -> None:
    """Set Modal credentials from Streamlit secrets."""
    import streamlit as st

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
        from utils.project_selection import (
            get_selected_project_key,
            get_project_config,
        )
        try:
            st.toast("Building Modal secrets...", icon=":material/key:")
        except Exception:
            pass

        current_project_key = get_selected_project_key() if "selected_project_key" in st.session_state else None
        if (
            st.session_state.get("modal_secrets")
            and st.session_state.get("modal_secrets_project_key") == current_project_key
        ):
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
        st.session_state.modal_secrets = [
            google_secret,
            project_data_secret,
            mysql_secret,
            rds_secret,
            b2_secret,
        ]
        st.session_state.modal_secrets_project_key = selected_project_key
        return st.session_state.modal_secrets
    else:
        logger.info(
            "Using named Modal secrets: 'google-credentials-json', 'project-data-json', 'mysql-credentials', "
            "'artefact-metadata-rds-credentials', 'artefact-blob-b2-credentials'"
        )
        return [
            modal.Secret.from_name("google-credentials-json"),
            modal.Secret.from_name("project-data-json"),
            modal.Secret.from_name("mysql-credentials"),
            modal.Secret.from_name("artefact-metadata-rds-credentials"),
            modal.Secret.from_name("artefact-blob-b2-credentials"),
        ]
