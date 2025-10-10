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
    if "user_id" not in st.session_state:
        st.session_state.user_id = 0
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
    if "show_llm_responses" not in st.session_state:
        st.session_state.show_llm_responses = True
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False