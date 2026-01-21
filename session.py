"""Manages session state for the MissionHelp Demo application.

This module initializes and maintains Streamlit session state variables used
across the application for configuration, conversation history, and multimedia.
"""

import streamlit as st
import uuid


def setup_session() -> None:
    """Initializes Streamlit session state variables with default values."""
    # Initialize core components
    if "llms" not in st.session_state:
        st.session_state.llms = {}
    if "db" not in st.session_state:
        st.session_state.db = False
    if "metadata_db" not in st.session_state:
        st.session_state.metadata_db = False
    if "blob_db" not in st.session_state:
        st.session_state.blob_db = False
    if "modal_secrets" not in st.session_state:
        st.session_state.modal_secrets = False
    if "graph" not in st.session_state:
        st.session_state.graph = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "new_message_added" not in st.session_state:
        st.session_state.new_message_added = False
    if "admin_logged_in" not in st.session_state:
        # st.session_state.admin_logged_in = False
        st.session_state.admin_logged_in = True
    if "intermediate_steps_history" not in st.session_state:
        st.session_state.intermediate_steps_history = []
    # if "user_permissions" not in st.session_state:
    #     st.session_state.user_permissions = None
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = 1 # Default to Super Admin user
    if "global_hierarchy_access" not in st.session_state:
        st.session_state.global_hierarchy_access = False
    if "table_relationship_graph" not in st.session_state:
        st.session_state.table_relationship_graph = None
    if "developer_view" not in st.session_state:
        st.session_state.developer_view = True
    if "show_sandbox_stream_logs" not in st.session_state:
        st.session_state.show_sandbox_stream_logs = False
    if "parallel_plan" not in st.session_state:
        st.session_state.parallel_plan = "Performance"
    if "num_parallel_executions" not in st.session_state:
        st.session_state.num_parallel_executions = 7
    if "completion_strategy" not in st.session_state:
        st.session_state.completion_strategy = "Intelligent"
    if "min_successful_responses" not in st.session_state:
        # st.session_state.min_successful_responses = 3
        st.session_state.min_successful_responses = 2
    if "min_explained_variance" not in st.session_state:
        # st.session_state.min_explained_variance = 0.7
        st.session_state.min_explained_variance = 0.6
    if "num_completions_before_response" not in st.session_state:
        st.session_state.num_completions_before_response = 2
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
    if "need_rebuild_graph" not in st.session_state:
        st.session_state.need_rebuild_graph = False
    if "selected_project_key" not in st.session_state:
        # st.session_state.selected_project_key = "project_data.18_167_246_137__db_lpp"
        st.session_state.selected_project_key = "project_data.cp03monitoringlive_cmvxgc0aonjr_ap_southeast_1_rds_amazonaws_com__db_cp03monitoring"
    if "project_context_cache" not in st.session_state:
        st.session_state.project_context_cache = {}
    if "container_warm" not in st.session_state:
        st.session_state.container_warm = False
    if "container_warm_count" not in st.session_state:
        st.session_state.container_warm_count = 0