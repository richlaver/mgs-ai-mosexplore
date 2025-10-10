import streamlit as st
import plotly.io as pio
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk

import json
import uuid
import logging

from parameters import users
import setup
from tools.artefact_toolkit import ReadArtefactsTool

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

blob_db = setup.get_blob_db()
metadata_db = setup.get_metadata_db()
llm = setup.get_llm()

@st.dialog("Login")
def login_modal():
    """Renders the login modal using Streamlit dialog."""
    with st.form("login_form", enter_to_submit=True):
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter admin password"
        )
        if st.form_submit_button("Login"):
            if password == st.secrets.admin_password:
                st.session_state.admin_logged_in = True
                st.toast("Logged in as admin", icon=":material/check_circle:")
                st.success("Logged in as admin")
                st.rerun()
            else:
                st.error("Incorrect password")

@st.dialog("Change User")
def user_modal():
    """Renders the user selection modal using Streamlit dialog."""
    with st.form("user_form", enter_to_submit=True):
        display_names = [user['display_name'] for user in users]
        
        selected_user = next((user for user in users 
                            if user['id'] == st.session_state.get('selected_user_id', None)), None)
        default_index = display_names.index(selected_user['display_name']) if selected_user else 0
        
        selected_display_name = st.selectbox(
            "Change User",
            display_names,
            key="user_select",
            label_visibility="visible",
            index=default_index
        )
        
        if st.form_submit_button("Select"):
            if selected_display_name:
                selected_user = next(user for user in users 
                                   if user['display_name'] == selected_display_name)
                st.session_state.selected_user_id = selected_user['id']
                st.session_state.global_hierarchy_access = setup.get_global_hierarchy_access()
                st.rerun()
            else:
                st.error("Please select a user")

def new_chat() -> None:
    """Clear chat history and start a new chat by resetting session state variables."""
    import uuid
    st.session_state.clear_chat = True

def handle_clear_chat() -> None:
    """Handle clearing the chat if the clear_chat flag is set."""
    if st.session_state.get('clear_chat', False):
        st.session_state.clear_chat = False
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.intermediate_steps_history = []

def render_initial_ui() -> None:
    """Renders the initial UI components (sidebar, app title, popover, disabled chat input) before setup."""
    st.markdown(
        """
        <style>
        .stAppHeader {
            background-color: rgba(255, 255, 255, 0.0);
            visibility: visible;
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        .chat-messages {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 60px;
            margin-top: 60px;
        }
        .stChatInput {
            position: fixed;
            bottom: 10px;
            width: 100%;
            max-width: 720px;
            margin: 0 auto;
            background-color: white;
            z-index: 1000;
        }
        .inline-image {
            margin: 10px 0;
            max-width: 100%;
        }
        .sidebar-title {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-bottom: 20px;
            text-transform: uppercase;
        }
        .app-title {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-top: 20px;
            text-transform: uppercase;
        }
        [data-testid="stPopover"] {
            width: auto !important;
            max-width: 48px !important;
            padding: 4px !important;
        }
        [data-testid="stPopover"] > div {
            width: 48px !important;
        }
        .popover-buttons {
            display: flex;
            flex-direction: column;
            gap: 4px;
            width: 40px;
        }
        .popover-container {
            display: flex;
            justify-content: flex-end;
        }
        .intermediate-step {
            background-color: #f5f5f5;
            border-left: 4px solid #ccc;
            padding: 10px;
            margin: 5px 0;
            font-size: 0.9em;
            color: #666;
        }
        .intermediate-step code {
            background-color: #e8e8e8;
            padding: 2px 4px;
            border-radius: 3px;
        }
        [data-testid="stExpander"] > details,
        [data-testid="stExpander"] > details > summary,
        [data-testid="stExpander"] > details > div {
            background-color: #FFE4CC;
            border: none !important;
            box-shadow: none !important;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        <script>
        function scrollToBottom() {
            if (window.newMessageAdded) {
                setTimeout(() => {
                    const chatMessages = document.querySelector('.chat-messages');
                    if (chatMessages) {
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                        window.newMessageAdded = false;
                    }
                }, 100);
            }
        }
        document.addEventListener('streamlit:render', scrollToBottom);
        </script>
        """,
        unsafe_allow_html=True,
    )

    st.logo(
        image="mgs-full-logo.svg",
        size="small",
        link="https://www.maxwellgeosystems.com/",
        icon_image="mgs-small-logo.svg"
    )

    with st.sidebar:
        st.divider()
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown('<div class="app-title">MISSION EXPLORE</div>', unsafe_allow_html=True)
        with cols[1]:
            st.button(
                label="", 
                icon=":material/chat_add_on:", 
                key="new_chat_button", 
                help="New chat",
                on_click=lambda: new_chat(),
                use_container_width=True
            )
            
            with st.popover(
                label="",
                icon=":material/menu:",
                use_container_width=False
            ):
                st.button(label="Switch User", icon=":material/account_circle:", key="user_button", help="Change user", on_click=user_modal, use_container_width=True)
                st.toggle(label="Show LLM Responses", key="show_llm_responses", help="Toggle to show or hide raw LLM responses")

    with st.empty():
        st.chat_input(
            placeholder="Setting up, please wait...",
            disabled=True,
            key="initial_chat_input"
        )

def prepend_message_icon(message):
    process_icon_map = {
        'query_enricher': 'search',
        'router': 'shuffle',
        'planner_coder': 'lightbulb',
        'code_checker': 'policy',
        'code_executor': 'terminal',
        'error_summariser': 'bug_report',
        'reporter': 'edit_square'
    }

    process = message.additional_kwargs.get('process')
    stage = message.additional_kwargs.get('stage')

    if process and (stage != 'final'):
        if stage == 'execution_output':
            icon_name = 'sprint'
        else:
            icon_name = process_icon_map.get(process)
        if icon_name:
            stripped = message.content.lstrip()
            if not stripped.startswith(':material/'):
                message.content = f":material/{icon_name}: " + stripped

    return message

def is_message_visible(message: AIMessage | AIMessageChunk, is_final: bool) -> bool:
    """Determine if a message should be visible based on its type and additional_kwargs."""
    additional_kwargs = message.additional_kwargs or {}

    stage = additional_kwargs.get('stage', 'unknown')
    process = additional_kwargs.get('process', 'unknown')
    if is_final:
        if stage == 'final':
            return True
    else:
        if isinstance(message, AIMessageChunk) and st.session_state.get('show_llm_responses', False):
            return True
        if stage == 'node' or (stage == 'execution_output' and process in ['progress', 'error']):
            return True
    return False

def render_message_content(message: AIMessage):
    """Render message content based on its type, handling artifacts for final and observation messages."""
    content = message.content
    additional_kwargs = message.additional_kwargs or {}
    process = additional_kwargs.get('process')
    if is_message_visible(message=message, is_final=True):
        if process == 'response':
            st.markdown(content)
        elif process == 'plot':
            artefact_id = additional_kwargs.get('artefact_id')
            if artefact_id:
                read_tool = ReadArtefactsTool(blob_db=blob_db, metadata_db=metadata_db)
                result = read_tool._run(metadata_only=False, artefact_ids=[artefact_id])
                if result['success'] and result['artefacts']:
                    artefact = result['artefacts'][0]
                    blob = artefact['blob']
                    try:
                        fig_json = json.loads(blob.decode('utf-8'))
                        fig = pio.from_json(json.dumps(fig_json))
                        with st.container():
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Failed to render Plotly figure: {str(e)}")
                        st.error("Error rendering plot")
        elif process == 'csv':
            artefact_id = additional_kwargs.get('artefact_id')
            if artefact_id:
                read_tool = ReadArtefactsTool(blob_db=blob_db, metadata_db=metadata_db)
                result = read_tool._run(metadata_only=False, artefact_ids=[artefact_id])
                if result['success'] and result['artefacts']:
                    artefact = result['artefacts'][0]
                    blob = artefact['blob'].decode('utf-8')
                    desc = artefact['metadata']['description_text']
                    prompt = f"Generate a short, descriptive filename for this CSV based on the description: {desc}. Do not include the .csv extension."
                    filename_response = llm.invoke(prompt)
                    filename = filename_response.content.strip() + '.csv'
                    with st.container():
                        st.download_button(
                            label=f"Download {filename}",
                            data=blob,
                            file_name=filename,
                            mime='text/csv',
                            key=f"csv_download_{artefact_id}"
                        )
    elif is_message_visible(message=message, is_final=False):
        return content

def render_chat_content() -> None:
    """Renders chat messages, history, and enabled chat input after setup is complete."""
    if not st.session_state.setup_complete:
        return
    
    logger.info(f"Show LLM Responses: {st.session_state.show_llm_responses}")
        
    handle_clear_chat()

    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    current_query_steps = []
    query_index = 0
    previous_message_type = None

    for message in st.session_state.get('messages', []):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
            if query_index < len(st.session_state.intermediate_steps_history):
                with st.expander(f"Intermediate Steps for Query {query_index + 1}", expanded=False):
                    rendered_content = ""
                    for step in st.session_state.intermediate_steps_history[query_index]:
                        rendered_content += step
                    st.markdown(rendered_content)
            query_index += 1
            previous_message_type = HumanMessage
        elif isinstance(message, (AIMessage, AIMessageChunk)):
            message = prepend_message_icon(message)
            if is_message_visible(message=message, is_final=True):
                with st.chat_message("assistant"):
                    render_message_content(message)
            elif is_message_visible(message=message, is_final=False):
                separator = "" if previous_message_type is AIMessageChunk and isinstance(message, AIMessageChunk) else "\n\n"
                current_query_steps.append(separator + (message.content or ""))
                with st.chat_message("assistant"):
                    rendered_content = ""
                    prev_type = None
                    for step in current_query_steps:
                        if step.startswith("\n\n"):
                            if prev_type is AIMessageChunk:
                                rendered_content += step[2:]
                                step = step[2:]
                            else:
                                rendered_content += step
                        else:
                            rendered_content += step
                        prev_type = AIMessageChunk if step and is_message_visible(message, is_final=False) and isinstance(message, AIMessageChunk) else AIMessage
                    st.markdown(rendered_content)
            previous_message_type = type(message)
    
    st.markdown("</div>", unsafe_allow_html=True)

    if question := st.chat_input(
        placeholder="Ask a query about project data:",
        key="active_chat_input"
    ):
        user_message = HumanMessage(content=question, additional_kwargs={"type": "query"})
        st.session_state.messages.append(user_message)
        st.session_state.new_message_added = True
        with st.chat_message("user"):
            st.markdown(question)

        MAX_HISTORY = 1000
        if len(st.session_state.messages) > MAX_HISTORY:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id
            }
        }
        initial_state = {
            "messages": st.session_state.messages,
            "timings": []
        }

        with st.spinner("Generating..."):
            with st.chat_message("assistant"):
                with st.expander(f"Intermediate Steps for Query {query_index + 1}", expanded=True):
                    thinking_container = st.empty()

                current_query_steps = []
                previous_message_type = None

                for state_update in st.session_state.graph.stream(initial_state, stream_mode="messages", config=config):
                    logger.debug(f'Raw state update in ui.py: {state_update[:100]}')

                    if isinstance(state_update, tuple):
                        state = state_update[0]
                        if isinstance(state, dict):
                            messages = state.get('messages', [])
                        else:
                            messages = [state]
                    else:
                        messages = state_update.get('messages', [])
                    
                    for message in messages:
                        logger.info(f'Streamed message: {message}')
                        if isinstance(message, (AIMessage, AIMessageChunk)):
                            message = prepend_message_icon(message)
                            if is_message_visible(message=message, is_final=True):
                                st.session_state.messages.append(message)
                                render_message_content(message)
                            elif is_message_visible(message=message, is_final=False):
                                separator = "" if previous_message_type is AIMessageChunk and isinstance(message, AIMessageChunk) else "\n\n"
                                current_query_steps.append(separator + (message.content or ""))
                                rendered_content = ""
                                prev_type = None
                                for step in current_query_steps:
                                    if step.startswith("\n\n"):
                                        if prev_type is AIMessageChunk:
                                            rendered_content += step[2:]
                                            step = step[2:]
                                        else:
                                            rendered_content += step
                                    else:
                                        rendered_content += step
                                    prev_type = AIMessageChunk if step and is_message_visible(message, is_final=False) and isinstance(message, AIMessageChunk) else AIMessage
                                thinking_container.markdown(rendered_content)
                            previous_message_type = type(message)

                if current_query_steps:
                    st.session_state.intermediate_steps_history.append(current_query_steps)
                if len(st.session_state.intermediate_steps_history) > MAX_HISTORY // 2:
                    st.session_state.intermediate_steps_history = st.session_state.intermediate_steps_history[-MAX_HISTORY // 2:]