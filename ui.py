import streamlit as st
import json
import uuid
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from parameters import users
import setup
import graph
import logging
import plotly.io as pio

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
                match st.session_state.test_mode:
                    case 'Tool':
                        st.session_state.graph = graph.build_tool_graph(
                            llm=st.session_state.llm,
                            db=st.session_state.db,
                            table_relationship_graph=st.session_state.table_relationship_graph,
                            user_id=st.session_state.selected_user_id,
                            global_hierarchy_access=st.session_state.global_hierarchy_access
                        )
                    case 'Supervisor':
                        st.session_state.graph = graph.build_supervisor_graph(
                            llm=st.session_state.llm,
                            db=st.session_state.db,
                            table_relationship_graph=st.session_state.table_relationship_graph,
                            user_id=st.session_state.selected_user_id,
                            global_hierarchy_access=st.session_state.global_hierarchy_access
                        )
                st.rerun()
            else:
                st.error("Please select a user")

@st.dialog("Select Test")
def test_mode_modal():
    """Renders the test mode selection modal using Streamlit dialog."""
    with st.form("test_mode_form", enter_to_submit=False):
        test_modes = ["Tool", "Supervisor"]
        default_mode = st.session_state.test_mode
        
        selected_test_mode = st.segmented_control(
            label="Select Test Mode",
            options=test_modes,
            selection_mode="single",
            default=default_mode,
            key="test_mode_select",
            label_visibility="visible",
            help="Choose the component to test: Tool, Sub-agent, or Supervisor."
        )
        
        if st.form_submit_button("Select"):
            if selected_test_mode:
                if selected_test_mode != st.session_state.test_mode:
                    match selected_test_mode:
                        case "Tool":
                            st.session_state.graph = graph.build_tool_graph(
                                llm=st.session_state.llm,
                                db=st.session_state.db,
                                table_relationship_graph=st.session_state.table_relationship_graph,
                                user_id=st.session_state.selected_user_id,
                                global_hierarchy_access=st.session_state.global_hierarchy_access
                            )
                        case "Supervisor":
                            st.session_state.graph = graph.build_supervisor_graph(
                                llm=st.session_state.llm,
                                db=st.session_state.db,
                                table_relationship_graph=st.session_state.table_relationship_graph,
                                user_id=st.session_state.selected_user_id,
                                global_hierarchy_access=st.session_state.global_hierarchy_access
                            )
                st.session_state.test_mode = selected_test_mode
                st.toast(f"Test mode set to {selected_test_mode}", icon=":material/check_circle:")
                st.rerun()
            else:
                st.error("Please select a test mode")

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
                st.button(label="Select Test", icon=":material/build:", key="test_mode_button", help="Select test mode", on_click=test_mode_modal, use_container_width=True)

    with st.empty():
        st.chat_input(
            placeholder="Setting up, please wait...",
            disabled=True,
            key="initial_chat_input"
        )

def parse_message_components(content, additional_kwargs=None):
    """Extract message type, content, and additional kwargs from prefixed message."""
    prefixes = [
        '[ACTION]:', 'action',
        '[ACTION_INPUT]:', 'action_input',
        '[OBSERVATION]:', 'observation',
        '[THOUGHT]:', 'thought',
        '[FINAL]:', 'final'
    ]
    
    for prefix, msg_type in zip(prefixes[::2], prefixes[1::2]):
        if content.startswith(prefix):
            return msg_type, content[len(prefix):], additional_kwargs or {}
    
    return None, content, additional_kwargs or {}

def render_message_content(msg: AIMessage, msg_type: str, clean_content: str, additional_kwargs: dict):
    """Render message content based on its type, handling artifacts for final and observation messages."""
    if msg_type in ['final', 'observation']:
        st.markdown(clean_content)
        artifacts = additional_kwargs.get("artifacts", [])
        if artifacts and isinstance(artifacts, list):
            for artifact in artifacts:
                artifact_type = artifact.get('type')
                if artifact_type == 'Plotly object':
                    try:
                        fig = pio.from_json(artifact['content'])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Failed to render Plotly figure: {str(e)}")
                        st.error("Error rendering plot")
                elif artifact_type == 'CSV':
                    st.download_button(
                        label="Download Time Series Data (CSV)",
                        data=artifact['content'],
                        file_name=artifact['filename'],
                        mime='text/csv',
                        key=f"csv_download_{artifact['artifact_id']}"
                    )
    else:
        # Return content for intermediate steps to be rendered in expander
        return clean_content

def render_chat_content() -> None:
    """Renders chat messages, history, and enabled chat input after setup is complete."""
    if not st.session_state.setup_complete:
        return
        
    handle_clear_chat()

    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    current_query_steps = []
    query_index = 0

    for msg in st.session_state.get('messages', []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
            if query_index < len(st.session_state.intermediate_steps_history):
                with st.expander(f"Intermediate Steps for Query {query_index + 1}", expanded=False):
                    for step in st.session_state.intermediate_steps_history[query_index]:
                        st.markdown(step, unsafe_allow_html=True)
            query_index += 1
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                msg_type, clean_content, additional_kwargs = parse_message_components(
                    msg.content, msg.additional_kwargs
                )
                content = render_message_content(msg, msg_type, clean_content, additional_kwargs)
                if content and msg_type != 'final':
                    current_query_steps.append(content)
    
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
                response_container = st.empty()

                current_query_steps = []

                for state_update in st.session_state.graph.stream(initial_state, stream_mode="messages", config=config):
                    logger.debug(f'Raw state update in ui.py: {state_update[:100]}')

                    # Handle case where state_update is a tuple
                    if isinstance(state_update, tuple):
                        # Assume the first element is the State dictionary or message
                        state = state_update[0]
                        if isinstance(state, dict):
                            messages = state.get('messages', [])
                        else:
                            # If state is a message (e.g., AIMessage), wrap it in a list
                            messages = [state]
                    else:
                        # Assume state_update is a State dictionary
                        messages = state_update.get('messages', [])
                    
                    for message in messages:
                        if isinstance(message, (AIMessage, AIMessageChunk)):
                            content = message.content
                            msg_type, clean_content, additional_kwargs = parse_message_components(
                                content, message.additional_kwargs
                            )

                            logger.debug(f'Extracted content: {content[:100]}')
                            logger.debug(f'Message type: {msg_type}')
                            
                            new_message = AIMessage(
                                content=content,
                                additional_kwargs=additional_kwargs
                            )
                            st.session_state.messages.append(new_message)

                            if msg_type == 'final':
                                response_container.markdown(clean_content)
                                render_message_content(new_message, msg_type, clean_content, additional_kwargs)
                            else:
                                current_query_steps.append(clean_content)
                                thinking_container.markdown("\n".join(current_query_steps))
                                if msg_type == 'observation':
                                    render_message_content(new_message, msg_type, clean_content, additional_kwargs)

                if current_query_steps:
                    st.session_state.intermediate_steps_history.append(current_query_steps)
                if len(st.session_state.intermediate_steps_history) > MAX_HISTORY // 2:
                    st.session_state.intermediate_steps_history = st.session_state.intermediate_steps_history[-MAX_HISTORY // 2:]