"""Main entry point for the MissionHelp Demo application.

This Streamlit app initializes the environment, sets up dependencies (LLM, vector
store, LangGraph), and renders the chatbot interface.
"""

import streamlit as st
import graph
import session
import setup
import ui
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def main() -> None:
    """Initializes and runs the MissionHelp Demo application."""
    st.set_page_config(
        page_title="MissionOS Explore",
        page_icon="mgs-small-logo.svg",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    session.setup_session()
    if not st.session_state.admin_logged_in:
        ui.login_modal()
    ui.render_initial_ui()

    if not st.session_state.setup_complete:
        setup.enable_tracing()
        setup.set_google_credentials()
        st.session_state.llm = setup.get_llm()
        st.session_state.db = setup.get_db()
        st.session_state.user_permissions = setup.get_user_permissions()
        st.session_state.table_relationship_graph = setup.build_relationship_graph()

        st.session_state.graph = graph.build_graph(
            llm=st.session_state.llm,
            db=st.session_state.db,
            user_permissions=st.session_state.user_permissions,
            table_relationship_graph=st.session_state.table_relationship_graph
        )
        st.session_state.setup_complete = True
        st.toast("Set-up complete!", icon=":material/check_circle:")

    if st.session_state.graph and st.session_state.admin_logged_in:
        ui.render_chat_content()


if __name__ == "__main__":
    main()