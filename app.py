import os
import sys

import streamlit as st
st.set_page_config(
    page_title="MissionOS Explore",
    page_icon="mgs-small-logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

import session
from parameters import table_info
from classes import AgentState, Context
from utils.context_data import (
    ensure_project_context,
    configure_context_api_from_secrets,
    register_project_configs,
    set_default_project_key,
)
from utils.timezone_utils import init_timezones
import logging

def setup_logging():
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("hpack").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

def perform_setup():
    setup_logging()
    import setup
    setup.set_e2b_api_key()
    setup.set_google_genai_env()
    setup.set_project_data_env()
    setup.set_e2b_template_env()
    setup.set_parallel_executions_env()
    setup.enable_tracing()
    setup.set_blob_db_env()
    setup.set_metadata_db_env()
    setup.set_db_env()
    
    st.session_state.llms = setup.get_llms()
    st.session_state.metadata_db = setup.get_metadata_db()
    st.session_state.blob_db = setup.get_blob_db()
    st.session_state.table_relationship_graph = setup.build_relationship_graph()

    st.session_state.db = setup.get_db()
    st.session_state.map_spatial_defaults = setup.get_map_spatial_defaults(st.session_state.db)
    st.session_state.global_hierarchy_access = setup.get_global_hierarchy_access(db=st.session_state.db)
    init_timezones(st.session_state.db)
    configure_context_api_from_secrets(st.secrets)
    register_project_configs(st.secrets.get("project_data", {}))
    set_default_project_key(st.session_state.get("selected_project_key"))
    ensure_project_context(st.session_state.get("selected_project_key"), force_refresh=True, strict=True)
    try:
        setup.refresh_instrument_selection_cache(
            st.session_state.get("selected_project_key"),
            st.session_state.llms["LONG"],
        )
    except Exception as exc:
        logger.warning("Failed to refresh instrument selection cache at startup: %s", exc)
    try:
        setup.refresh_codeact_coder_cache(
            st.session_state.llms.get("CODING", st.session_state.llms["LONG"]),
            tools_payload="[]",
        )
    except Exception as exc:
        logger.warning("Failed to refresh CodeAct coder cache at startup: %s", exc)

    raw_parallel = os.environ.get("MGS_NUM_PARALLEL_EXECUTIONS")
    try:
        parallel_count = int(raw_parallel) if raw_parallel is not None else 1
    except Exception:
        parallel_count = 1
    parallel_count = max(1, parallel_count)
    st.session_state.min_successful_responses = min(
        st.session_state.get("min_successful_responses", 3),
        parallel_count,
    )
    st.session_state.min_explained_variance = float(st.session_state.get("min_explained_variance", 0.7))

    import graph
    st.session_state.graph = graph.build_graph(
        llms=st.session_state.llms,
        db=st.session_state.db,
        blob_db=st.session_state.blob_db,
        metadata_db=st.session_state.metadata_db,
        table_info=table_info,
        table_relationship_graph=st.session_state.table_relationship_graph,
        thread_id=st.session_state.thread_id,
        user_id=st.session_state.selected_user_id,
        global_hierarchy_access=st.session_state.global_hierarchy_access,
        min_successful_responses=st.session_state.min_successful_responses,
        min_explained_variance=st.session_state.min_explained_variance,
        selected_project_key=st.session_state.get("selected_project_key"),
    )
    st.session_state.graph_agent_state_cls_id = id(AgentState)
    st.session_state.graph_context_cls_id = id(Context)

def main() -> None:
    needs_setup = not st.session_state.get("setup_complete", False)
    cached_agent_cls_id = st.session_state.get("graph_agent_state_cls_id")
    cached_context_cls_id = st.session_state.get("graph_context_cls_id")
    if cached_agent_cls_id and cached_agent_cls_id != id(AgentState):
        logger.info("Detected updated AgentState class definition; rebuilding graph.")
        needs_setup = True
    if cached_context_cls_id and cached_context_cls_id != id(Context):
        logger.info("Detected updated Context class definition; rebuilding graph.")
        needs_setup = True

    if needs_setup:
        perform_setup()
        st.session_state.setup_complete = True
        st.toast("Set-up complete!", icon=":material/celebration:")
    
    import ui
    if not st.session_state.get("admin_logged_in", False):
        ui.login_modal()
    ui.render_initial_ui()

    if st.session_state.graph and st.session_state.admin_logged_in:
        ui.render_chat_content()

if __name__ == "__main__":
    session.setup_session()
    main()