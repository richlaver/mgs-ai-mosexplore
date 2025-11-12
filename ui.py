import streamlit as st
import plotly.io as pio
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk

import json
import uuid
import logging
import re

from classes import AgentState, Context
from parameters import users, table_info
from graph import build_graph
import setup
from tools.artefact_toolkit import ReadArtefactsTool, DeleteArtefactsTool
from modal_management import deploy_app, warm_up_container, stop_app, is_app_deployed, is_container_warm
from utils.chat_history import filter_messages_only_final

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
                st.toast("Logged in as admin", icon=":material/login:")
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

@st.dialog("Sandbox")
def sandbox_modal():
    st.markdown(f"App Deployed: **{'Yes' if st.session_state.app_deployed else 'No'}**")
    st.markdown(f"Container Warm: **{'Yes' if st.session_state.container_warm else 'No'}**")
    st.info("Note: Keeping a container warm costs ~$55/month. Stop when not in use to save costs.")

    spin_up_disabled = st.session_state.app_deployed and st.session_state.container_warm
    kill_disabled = not (st.session_state.app_deployed and st.session_state.container_warm)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Spin-up", disabled=spin_up_disabled, key="spin_up", icon=":material/rocket_launch:"):
            if not st.session_state.app_deployed:
                deploy_app()
            if st.session_state.app_deployed and not st.session_state.container_warm:
                warm_up_container()
            st.rerun()
    with col2:
        if st.button("Kill", disabled=kill_disabled, key="kill", icon=":material/block:"):
            stop_app()
            st.rerun()

@st.dialog("Delete All Artefacts")
def delete_artefacts_modal():
    st.markdown(
        """
        <div style="text-align: center;">
            <h3 style="color: #d32f2f;">Do you really want to irreversibly delete all artefacts?<br></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", type="secondary", use_container_width=True):
            pass
    with col2:
        delete_button = st.button(
            "Delete Forever",
            type="primary",
            icon=":material/delete_forever:",
            use_container_width=True,
            key="delete_forever_btn"
        )
        if delete_button:
            with st.spinner("Deleting all artefacts... This may take some time..."):
                try:
                    delete_tool = DeleteArtefactsTool(
                        blob_db=st.session_state.blob_db,
                        metadata_db=st.session_state.metadata_db
                    )
                    result = delete_tool._run()

                    if result.get("error"):
                        st.error(f"Failed to delete some artefacts: {result['error']}")
                    else:
                        st.toast("All artefacts deleted", icon=":material/delete_forever:")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                finally:
                    st.rerun()

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
    if "app_deployed" not in st.session_state:
        st.session_state.app_deployed = is_app_deployed()
        # if not st.session_state.app_deployed:
        #     st.session_state.sandbox_mode = "Local"
        logger.info(f"app_deployed after setting session state variable: {st.session_state.app_deployed}")
    if "container_warm" not in st.session_state:
        st.session_state.container_warm = is_container_warm()
        logger.info(f"container_warm after setting session state variable: {st.session_state.container_warm}")
        
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
            padding-left: 1rem;
            padding-right: 1rem;
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
            max-width: 1100px;
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

    def _rebuild_graph():
        st.session_state.num_completions_before_response = max(
            1,
            min(
                st.session_state.get('num_completions_before_response', 1),
                st.session_state.num_parallel_executions,
            ),
        )
        st.session_state.graph = build_graph(
            llm=st.session_state.llm,
            db=st.session_state.db,
            blob_db=st.session_state.blob_db,
            metadata_db=st.session_state.metadata_db,
            table_info=table_info,
            table_relationship_graph=st.session_state.table_relationship_graph,
            thread_id=st.session_state.thread_id,
            user_id=st.session_state.selected_user_id,
            global_hierarchy_access=st.session_state.global_hierarchy_access,
            remote_sandbox=st.session_state.sandbox_mode == "Remote",
            num_parallel_executions=st.session_state.num_parallel_executions,
            num_completions_before_response=st.session_state.num_completions_before_response,
            agent_type=st.session_state.agent_type,
        )

    with st.sidebar:
        st.divider()
        st.slider(
            label="No. of Parallel-Running Agents",
            min_value=1,
            max_value=3,
            step=1,
            format="%i",
            key="num_parallel_executions",
            help="Use more agents to improve **accuracy** with marginal latency penalty",
            on_change=_rebuild_graph,
        )
        # Uncomment to enable completions before response slider
        # if st.session_state.num_parallel_executions > 1:
        #     st.slider(
        #         label="No. of Completions Before Responding",
        #         min_value=1,
        #         max_value=st.session_state.num_parallel_executions,
        #         step=1,
        #         format="%i",
        #         key="num_completions_before_response",
        #         help="Await completions to improve **accuracy** but prolong **latency**",
        #         on_change=_rebuild_graph,
        #     )
        st.selectbox(
            label="Agent Type",
            options=["Auto", "CodeAct", "ReAct", "Tool-Calling"],
            key="agent_type",
            help="**CodeAct**: writes and executes code.  \n**ReAct**: reasons and uses tools.  \n**Tool-Calling**: parallel tool execution",
            on_change=_rebuild_graph,
        )
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
                st.toggle(label="Developer View", key="developer_view", help="Toggle to show or hide internal LLM responses")
                st.button(label="Switch User", icon=":material/account_circle:", key="user_button", help="Change user", on_click=user_modal, use_container_width=True)
                st.button(
                    label="Sandbox App",
                    icon=":material/developer_board:",
                    key="sandbox_button",
                    help="Manage Sandbox app and containers",
                    on_click=sandbox_modal,
                    use_container_width=True
                )
                st.button(
                    "Delete Artefacts",
                    icon=":material/delete_forever:",
                    key="delete_artefacts_button",
                    help="Delete all generated plots and CSVs",
                    on_click=delete_artefacts_modal,
                    use_container_width=True
                )

    with st.empty():
        st.chat_input(
            placeholder="Setting up, please wait...",
            disabled=True,
            key="initial_chat_input"
        )

def is_message_visible(message: AIMessage | AIMessageChunk, is_final: bool) -> bool:
    additional_kwargs = message.additional_kwargs or {}
    if is_final:
        return additional_kwargs.get('is_final') is True
    else:
        if st.session_state.get('developer_view', False):
            return True
        if additional_kwargs.get('is_messenger') is True:
            return True
    return False

def postprocess_state_update(state_update) -> AIMessage | AIMessageChunk | None:
    node_icon_map = {
        'history_summariser': 'summarize',
        'context_orchestrator': 'search',
        'query_classifier': 'shuffle',
        'codeact_coder_branch': 'lightbulb',
        'codeact_executor_branch': 'sprint',
        'react_branch': 'conversion_path',
        'reporter': 'edit_square'
    }
    if isinstance(state_update, tuple) and len(state_update) == 2:
        message, metadata = state_update
        if isinstance(message, (AIMessage, AIMessageChunk)) and isinstance(metadata, dict):
            # Augment metadata
            langgraph_node = metadata.get('langgraph_node', "")
            if not hasattr(message, "additional_kwargs"):
                message.additional_kwargs = {}

            stage = message.additional_kwargs.get('stage')
            message.additional_kwargs["is_final"] = stage == 'final'

            message.additional_kwargs["is_messenger"] = "messenger" in langgraph_node
            if "branch" in langgraph_node:
                message.additional_kwargs["thinking_container"] = "parallel"
            elif "reporter" in langgraph_node:
                message.additional_kwargs["thinking_container"] = "postparallel"
            else:
                message.additional_kwargs["thinking_container"] = "preparallel"
            suffix_match = re.search(r'_branch_(\d+)(?=_|$)', langgraph_node)
            message.additional_kwargs["branch_id"] = int(suffix_match.group(1)) if suffix_match else None

            # Prepend icon
            if message.additional_kwargs["is_messenger"] or stage == "execution_output":
                for node_key, icon_name in node_icon_map.items():
                    if node_key in langgraph_node:
                        if not message.content.startswith(':material/'):
                            message.content = f":material/{icon_name}: " + message.content.lstrip()
                        break

            return message
    return None

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
            logger.info("artefact_id from `render_message_content`: %s", artefact_id)
            if artefact_id:
                read_tool = ReadArtefactsTool(blob_db=st.session_state.blob_db, metadata_db=st.session_state.metadata_db)
                result = read_tool._run(metadata_only=False, artefact_ids=[artefact_id])
                logger.info("Result from ReadArtefactsTool in `render_message_content`: %s", result)
                if result['success'] and result['artefacts']:
                    artefact = result['artefacts'][0]
                    blob = artefact['blob']
                    try:
                        fig_json = json.loads(blob.decode('utf-8'))
                        fig = pio.from_json(json.dumps(fig_json))
                        with st.container():
                            st.plotly_chart(fig, use_container_width=True, key=f"plot_{artefact_id}")
                    except Exception as e:
                        logger.error(f"Failed to render Plotly figure: {str(e)}")
                        st.error("Error rendering plot")
        elif process == 'csv':
            artefact_id = additional_kwargs.get('artefact_id')
            if artefact_id:
                read_tool = ReadArtefactsTool(blob_db=st.session_state.blob_db, metadata_db=st.session_state.metadata_db)
                result = read_tool._run(metadata_only=False, artefact_ids=[artefact_id])
                if result['success'] and result['artefacts']:
                    artefact = result['artefacts'][0]
                    blob = artefact['blob'].decode('utf-8')
                    desc = artefact['metadata']['description_text']
                    prompt = f"Generate a short, descriptive filename for this CSV based on the description: {desc}. Do not include the .csv extension."
                    filename_response = st.session_state.llm.invoke(prompt)
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

    handle_clear_chat()

    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

    i = 0
    query_index = 0
    while i < len(st.session_state.get('messages', [])):
        message = st.session_state.messages[i]
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
            
            with st.chat_message("assistant"):
                if query_index < len(st.session_state.intermediate_steps_history):
                    query_steps = st.session_state.intermediate_steps_history[query_index]
                    with st.expander(f"Intermediate Steps for Query {query_index + 1}", expanded=False):                    
                        if query_steps['preparallel']:
                            preparallel_content = ''.join(query_steps['preparallel'])
                            st.markdown(preparallel_content)
                        
                        parallels = query_steps['parallels']
                        if parallels:
                            num_branches = len(parallels)
                            parallel_thinking_cols = st.columns(num_branches)
                            for branch_id, steps in enumerate(parallels):
                                with parallel_thinking_cols[branch_id]:
                                    if steps:
                                        parallel_content = ''.join(steps)
                                        st.markdown(parallel_content)
                        
                        if query_steps['postparallel']:
                            postparallel_content = ''.join(query_steps['postparallel'])
                            st.markdown(postparallel_content)
                
                i += 1
                while i < len(st.session_state.messages) and isinstance(st.session_state.messages[i], (AIMessage, AIMessageChunk)):
                    final_message = st.session_state.messages[i]
                    render_message_content(final_message)
                    i += 1
                i -= 1
            
            query_index += 1
        i += 1
    
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
                "user_id": st.session_state.selected_user_id
            }
        }
        messages_summary = filter_messages_only_final(st.session_state.messages)
        initial_state = AgentState(
            messages=messages_summary,
            context=Context(retrospective_query="")
        )

        with st.spinner("Generating..."):
            with st.chat_message("assistant"):
                with st.expander(f"Intermediate Steps for Query {query_index + 1}", expanded=True):
                    # Define thinking containers
                    preparallel_thinking_container = st.empty()
                    parallel_thinking_containers = []
                    parallel_thinking_cols = st.columns(st.session_state.num_parallel_executions)
                    for col_idx in range(st.session_state.num_parallel_executions):
                        with parallel_thinking_cols[col_idx]:
                            parallel_thinking_container = st.empty()
                            parallel_thinking_containers.append(parallel_thinking_container)
                    postparallel_thinking_container = st.empty()

                    current_query_steps = {
                        'preparallel': [],
                        'parallels': [[] for _ in range(st.session_state.num_parallel_executions)],
                        'postparallel': []
                    }
                    st.session_state.intermediate_steps_history.append(current_query_steps)

                    preparallel_current_query_steps = current_query_steps['preparallel']
                    postparallel_current_query_steps = current_query_steps['postparallel']
                    parallel_current_query_steps = current_query_steps['parallels']

                    preparallel_previous_message_type = None
                    parallel_previous_message_types = [None for _ in range(st.session_state.num_parallel_executions)]
                    postparallel_previous_message_type = None

                for state_update in st.session_state.graph.stream(initial_state, stream_mode="messages", config=config):
                    logger.debug(f'Raw state update in ui.py: {state_update[:100]}')
                    message = postprocess_state_update(state_update)

                    if is_message_visible(message=message, is_final=True):
                        # Render final response
                        st.session_state.messages.append(message)
                        render_message_content(message)
                    elif is_message_visible(message=message, is_final=False):
                        # Render intermediate steps
                        match message.additional_kwargs.get("thinking_container"):
                            case "preparallel":
                                separator = "" if preparallel_previous_message_type is AIMessageChunk and isinstance(message, AIMessageChunk) else "\n\n"
                                preparallel_current_query_steps.append(separator + (message.content or ""))
                                rendered_content = ""
                                prev_type = None
                                for step in preparallel_current_query_steps:
                                    rendered_content += step
                                    prev_type = AIMessageChunk if step and is_message_visible(message, is_final=False) and isinstance(message, AIMessageChunk) else AIMessage
                                preparallel_thinking_container.markdown(rendered_content)
                                preparallel_previous_message_type = type(message)
                            case "parallel":
                                branch_id = message.additional_kwargs.get("branch_id", 0)
                                separator = "" if parallel_previous_message_types[branch_id] is AIMessageChunk and isinstance(message, AIMessageChunk) else "\n\n"
                                parallel_current_query_steps[branch_id].append(separator + (message.content or ""))
                                rendered_content = ""
                                prev_type = None
                                for step in parallel_current_query_steps[branch_id]:
                                    rendered_content += step
                                    prev_type = AIMessageChunk if step and is_message_visible(message, is_final=False) and isinstance(message, AIMessageChunk) else AIMessage
                                parallel_thinking_containers[branch_id].markdown(rendered_content)
                                parallel_previous_message_types[branch_id] = type(message)
                            case "postparallel":
                                separator = "" if postparallel_previous_message_type is AIMessageChunk and isinstance(message, AIMessageChunk) else "\n\n"
                                postparallel_current_query_steps.append(separator + (message.content or ""))
                                rendered_content = ""
                                prev_type = None
                                for step in postparallel_current_query_steps:
                                    rendered_content += step
                                    prev_type = AIMessageChunk if step and is_message_visible(message, is_final=False) and isinstance(message, AIMessageChunk) else AIMessage
                                postparallel_thinking_container.markdown(rendered_content)
                                postparallel_previous_message_type = type(message)

                if len(st.session_state.intermediate_steps_history) > MAX_HISTORY // 2:
                    st.session_state.intermediate_steps_history = st.session_state.intermediate_steps_history[-MAX_HISTORY // 2:]