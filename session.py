"""Manages session state for the MissionHelp Demo application.

This module initializes and maintains Streamlit session state variables used
across the application for configuration, conversation history, and multimedia.
"""

import streamlit as st
import uuid


def setup_session() -> None:
    """Initializes Streamlit session state variables with default values."""
    # Initialize core components
    if "llm" not in st.session_state:
        st.session_state.llm = False
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
    if "sandbox_mode" not in st.session_state:
        st.session_state.sandbox_mode = "Remote"
    if "num_parallel_executions" not in st.session_state:
        st.session_state.num_parallel_executions = 2
    if "num_completions_before_response" not in st.session_state:
        st.session_state.num_completions_before_response = 2
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
    if "need_rebuild_graph" not in st.session_state:
        st.session_state.need_rebuild_graph = False
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "CodeAct"