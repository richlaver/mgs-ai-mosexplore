import streamlit as st
st.set_page_config(
    page_title="MissionOS Explore",
    page_icon="mgs-small-logo.svg",
    layout="centered",
    initial_sidebar_state="expanded",
)

import session
from parameters import table_info

def perform_setup():
    import setup
    setup.set_google_credentials()
    setup.enable_tracing()
    setup.set_blob_db_env()
    setup.set_metadata_db_env()
    setup.set_db_env()
    
    st.session_state.llm = setup.get_llm()
    st.session_state.metadata_db = setup.get_metadata_db()
    st.session_state.blob_db = setup.get_blob_db()
    st.session_state.table_relationship_graph = setup.build_relationship_graph()

    st.session_state.db = setup.get_db()
    st.session_state.global_hierarchy_access = setup.get_global_hierarchy_access(db=st.session_state.db)

    import graph
    st.session_state.graph = graph.build_codeact_graph(
        llm=st.session_state.llm,
        db=st.session_state.db,
        table_info=table_info,
        table_relationship_graph=st.session_state.table_relationship_graph,
        thread_id=st.session_state.thread_id,
        user_id=st.session_state.selected_user_id,
        global_hierarchy_access=st.session_state.global_hierarchy_access
    )

def main() -> None:
    if not st.session_state.get("setup_complete", False):
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