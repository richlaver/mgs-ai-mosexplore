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
from utils.project_selection import (
    list_projects,
    get_selected_project_key,
)
from utils.context_data import (
    ensure_project_context,
    configure_context_api_from_secrets,
    register_project_configs,
    set_default_project_key,
)
from tools.artefact_toolkit import ReadArtefactsTool, DeleteArtefactsTool
from modal_management import deploy_app, stop_app, is_app_deployed, warm_up_container
from utils.chat_history import filter_messages_only_final
from utils.run_cancellation import (
    RunCancelledError,
    RunCancellationController,
    activate_controller,
    reset_controller,
)

logger = logging.getLogger(__name__)


def _normalize_project_key(project_key: str) -> str:
    if not project_key:
        return ""
    if project_key.startswith("project_data."):
        return project_key
    return f"project_data.{project_key}"


def _get_project_users(project_key: str) -> list[dict]:
    if not project_key:
        return []
    normalized = _normalize_project_key(project_key)
    matched: list[dict] = []
    for entry in users:
        entry_key = entry.get("project_key")
        if not entry_key:
            continue
        entry_normalized = _normalize_project_key(str(entry_key))
        if entry_key == project_key or entry_normalized == normalized:
            matched.extend(entry.get("users", []))
    return matched


def _sync_selected_user_for_project(project_key: str, force_reset: bool = False) -> list[dict]:
    project_users = _get_project_users(project_key)
    if not project_users:
        return project_users
    user_ids = [user.get("id") for user in project_users]
    if force_reset or st.session_state.get("selected_user_id") not in user_ids:
        st.session_state.selected_user_id = project_users[0].get("id")
    if st.session_state.get("user_selector_id") not in user_ids:
        st.session_state.user_selector_id = st.session_state.selected_user_id
    return project_users

def is_code_like(text: str) -> bool:
    """Heuristically determine if a text chunk looks like code.

    Rules:
    - Explicit fenced start (``` or ```python) counts as code.
    - Prefer syntax signals over generic English keywords to avoid false positives.
    - Use multiple signals to classify ambiguous cases.
    """

    if not text:
        return False
    t = text.lstrip()
    if t.startswith("```"):
        return True

    lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False

    strong_signals = [
        re.compile(r"^\s*(def|class)\s+\w+\s*\(?.*\):"),     # def foo(...): or class Bar:
        re.compile(r"^\s*import\s+[A-Za-z0-9_., ]+"),           # import x, y
        re.compile(r"^\s*from\s+[A-Za-z0-9_\.]+\s+import\s+"),# from x.y import z
        re.compile(r"^\s*(try|except|with|for|while|if|elif|else)\b.*:\s*$"), # block starters ending with ':'
        re.compile(r"^\s*@[A-Za-z_][A-Za-z0-9_]*\b"),           # decorators
        re.compile(r"^\s*#[^\n]*$"),                            # comment line
    ]

    weak_signals = [
        re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+$"),   # assignment
        re.compile(r"\breturn\b|\byield\b|\blambda\b"),      # less common in prose
        re.compile(r"\basync\b|\bawait\b"),                    # async/await
        re.compile(r"\bprint\s*\("),                           # print(
        re.compile(r"[(){}\[\]]"),                              # brackets presence
    ]

    strong_hits = 0
    weak_hits = 0
    structural_hits = 0

    for ln in lines[:6]:
        for rx in strong_signals:
            if rx.search(ln):
                strong_hits += 1
                break
        for rx in weak_signals:
            if rx.search(ln):
                weak_hits += 1
        # structural cues: indentation, line ending with ':', or punctuation density typical in code
        if ln.startswith(" ") or ln.startswith("\t"):
            structural_hits += 1
        if ln.endswith(":"):
            structural_hits += 1
        if sum(ln.count(ch) for ch in ",.;:(){}[]") >= 3:
            structural_hits += 1

    if strong_hits >= 1:
        return True
    if weak_hits >= 2 and structural_hits >= 1:
        return True
    return False


def _set_active_run_controller(controller: RunCancellationController, token) -> None:
    st.session_state.active_run_controller = controller
    st.session_state.active_run_controller_token = token


def _clear_active_run_controller() -> None:
    token = st.session_state.pop("active_run_controller_token", None)
    if token is not None:
        try:
            reset_controller(token)
        except Exception:
            pass
    st.session_state.pop("active_run_controller", None)
    st.session_state.pop("stop_button_pending", None)


def _cancel_active_run(reason: str) -> None:
    controller: RunCancellationController | None = st.session_state.get("active_run_controller")
    if controller:
        try:
            controller.cancel(reason)
        finally:
            st.session_state.stop_button_pending = False

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

@st.dialog("Sandbox")
def sandbox_modal():
    st.markdown(f"App Deployed: **{'Yes' if st.session_state.app_deployed else 'No'}**")
    st.info("Note: Keeping a container warm costs ~$55/month. Stop when not in use to save costs.")

    spin_up_disabled = st.session_state.app_deployed
    kill_disabled = not st.session_state.app_deployed

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Spin-up", disabled=spin_up_disabled, key="spin_up", icon=":material/rocket_launch:"):
            if not st.session_state.app_deployed:
                deploy_app()
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
    configure_context_api_from_secrets(st.secrets)
    register_project_configs(st.secrets.get("project_data", {}))
    set_default_project_key(st.session_state.get("selected_project_key"))
    if "app_deployed" not in st.session_state:
        st.session_state.app_deployed = is_app_deployed()
        logger.info(f"app_deployed after setting session state variable: {st.session_state.app_deployed}")
        
    st.markdown(
        """
        <style>
        .stAppHeader {
            background-color: rgba(255, 255, 255, 0.0);
            visibility: visible;
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 12rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stChatMessage,
        [data-testid="stChatMessage"] {
            margin-bottom: 0.75rem;
        }
        .stChatMessage:last-of-type,
        [data-testid="stChatMessage"]:last-of-type {
            margin-bottom: 6rem;
        }
        .chat-messages {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            padding-bottom: 10px;
            margin-bottom: 60px;
            margin-top: 60px;
        }
        .stChatInput {
            position: fixed;
            bottom: 10px;
            width: calc(100% - 60px - 336px);
            # max-width: calc(1100px - 120px);
            margin: 0 auto;
            background-color: white;
            z-index: 1000;
        }
        .st-key-stop_active_run_button {
            position: fixed !important;
            bottom: 10px;
            right: 12px;
            # right: max(12px, calc((100vw - min(100vw, 1100px)) / 2));
            width: auto !important;
            height: auto !important;
            padding: 0 !important;
            z-index: 1001;
            display: flex;
            justify-content: flex-end;
            align-items: flex-end;
        }
        .st-key-stop_active_run_button [data-testid="stTooltipHoverTarget"] {
            width: auto !important;
            display: flex;
            justify-content: flex-end;
        }
        .st-key-stop_active_run_button button[data-testid="stBaseButton-secondary"] {
            padding: 6px;
            min-width: auto;
            width: 40px;
            height: 40px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin: 0;
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
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                    window.newMessageAdded = false;
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

    def _majority_threshold(count: int) -> int:
        return max(1, (count // 2) + 1)

    parallel_plans = {
        "Economy": {"agents": 3, "label": "Economy — 3 agents"},
        "Reliable": {"agents": 5, "label": "Reliable — 5 agents"},
        "Performance": {"agents": 7, "label": "Performance — 7 agents"},
    }

    completion_strategies = {
        "Intelligent": {
            "label": "Intelligent — consistency-based",
            "resolve": _majority_threshold,
        },
        "Quick": {
            "label": "Quick — first completion",
            "resolve": lambda n: 1,
        },
        "Balanced": {
            "label": "Balanced — majority completion",
            "resolve": _majority_threshold,
        },
        "Max": {
            "label": "Max — all agents complete",
            "resolve": lambda n: max(1, n),
        },
    }
    def _rebuild_graph():
        selected_plan = st.session_state.get("parallel_plan", "Economy")
        plan = parallel_plans.get(selected_plan, parallel_plans["Economy"])
        st.session_state.num_parallel_executions = plan["agents"]

        selected_strategy = st.session_state.get("completion_strategy", "Intelligent")
        strategy = completion_strategies.get(selected_strategy, completion_strategies["Intelligent"])
        st.session_state.num_completions_before_response = strategy["resolve"](st.session_state.num_parallel_executions)

        min_successful = min(
            st.session_state.get("min_successful_responses", 3),
            st.session_state.num_parallel_executions,
        )
        st.session_state.min_successful_responses = min_successful
        st.session_state.min_explained_variance = float(st.session_state.get("min_explained_variance", 0.7))

        st.session_state.graph = build_graph(
            llms=st.session_state.llms,
            db=st.session_state.db,
            blob_db=st.session_state.blob_db,
            metadata_db=st.session_state.metadata_db,
            table_info=table_info,
            table_relationship_graph=st.session_state.table_relationship_graph,
            thread_id=st.session_state.thread_id,
            user_id=st.session_state.selected_user_id,
            global_hierarchy_access=st.session_state.global_hierarchy_access,
            num_parallel_executions=st.session_state.num_parallel_executions,
            num_completions_before_response=st.session_state.num_completions_before_response,
            response_mode=selected_strategy,
            min_successful_responses=st.session_state.min_successful_responses,
            min_explained_variance=st.session_state.min_explained_variance,
            selected_project_key=st.session_state.get("selected_project_key"),
        )
        try:
            if st.session_state.get("app_deployed"):
                warm_up_container()
        except Exception:
            pass

    with st.sidebar:
        st.divider()
        projects = list_projects()
        if projects:
            display_names = [p["display_name"] for p in projects]
            key_by_display = {p["display_name"]: p["key"] for p in projects}
            current_project_key = get_selected_project_key()
            current_display_name = next((p["display_name"] for p in projects if p["key"] == current_project_key), display_names[0])

            def _on_project_change():
                selected_display = st.session_state.project_selector_display
                selected_key = key_by_display.get(selected_display)
                if selected_key and selected_key != st.session_state.get("selected_project_key"):
                    st.session_state.selected_project_key = selected_key
                    set_default_project_key(selected_key)
                    try:
                        setup.set_db_env()
                        st.session_state.db = setup.get_db()
                    except Exception as e:
                        st.error(f"Failed to switch project database: {e}")
                        return
                    _sync_selected_user_for_project(selected_key, force_reset=True)
                    try:
                        st.session_state.modal_secrets = setup.build_modal_secrets()
                    except Exception:
                        pass
                    try:
                        st.session_state.global_hierarchy_access = setup.get_global_hierarchy_access(db=st.session_state.db)
                    except Exception:
                        pass
                    try:
                        st.session_state.table_relationship_graph = setup.build_relationship_graph()
                    except Exception:
                        pass
                    try:
                        ensure_project_context(selected_key, force_refresh=True, strict=True)
                    except Exception as e:
                        logger.error("Failed to refresh project contexts for %s: %s", selected_key, e)
                    if all(k in st.session_state for k in ["llms", "db", "blob_db", "metadata_db", "thread_id", "selected_user_id", "global_hierarchy_access", "num_parallel_executions"]):
                        try:
                            logger.info("[Project Switch] Rebuilding agent graph for project=%s", selected_key)
                            st.session_state.graph = build_graph(
                                llms=st.session_state.llms,
                                db=st.session_state.db,
                                blob_db=st.session_state.blob_db,
                                metadata_db=st.session_state.metadata_db,
                                table_info=table_info,
                                table_relationship_graph=st.session_state.table_relationship_graph,
                                thread_id=st.session_state.thread_id,
                                user_id=st.session_state.selected_user_id,
                                global_hierarchy_access=st.session_state.global_hierarchy_access,
                                num_parallel_executions=st.session_state.num_parallel_executions,
                                num_completions_before_response=st.session_state.get("num_completions_before_response", 1),
                                response_mode=st.session_state.get("completion_strategy", "Intelligent"),
                                min_successful_responses=st.session_state.get("min_successful_responses", 3),
                                min_explained_variance=st.session_state.get("min_explained_variance", 0.7),
                                selected_project_key=selected_key,
                            )
                            logger.info("[Project Switch] Graph rebuilt for project=%s", selected_key)
                        except Exception as e:
                            st.error(f"Failed to rebuild agent graph after project switch: {e}")
                    st.toast(f"Switched to project: {selected_display}", icon=":material/database:")

            def _on_user_change():
                selected_user_id = st.session_state.get("user_selector_id")
                if selected_user_id and selected_user_id != st.session_state.get("selected_user_id"):
                    st.session_state.selected_user_id = selected_user_id
                    try:
                        st.session_state.global_hierarchy_access = setup.get_global_hierarchy_access(db=st.session_state.db)
                    except Exception:
                        pass
                    if all(k in st.session_state for k in ["llms", "db", "blob_db", "metadata_db", "thread_id", "selected_user_id", "global_hierarchy_access", "num_parallel_executions"]):
                        try:
                            st.session_state.graph = build_graph(
                                llms=st.session_state.llms,
                                db=st.session_state.db,
                                blob_db=st.session_state.blob_db,
                                metadata_db=st.session_state.metadata_db,
                                table_info=table_info,
                                table_relationship_graph=st.session_state.table_relationship_graph,
                                thread_id=st.session_state.thread_id,
                                user_id=st.session_state.selected_user_id,
                                global_hierarchy_access=st.session_state.global_hierarchy_access,
                                num_parallel_executions=st.session_state.num_parallel_executions,
                                num_completions_before_response=st.session_state.get("num_completions_before_response", 1),
                                response_mode=st.session_state.get("completion_strategy", "Intelligent"),
                                min_successful_responses=st.session_state.get("min_successful_responses", 3),
                                min_explained_variance=st.session_state.get("min_explained_variance", 0.7),
                                selected_project_key=st.session_state.get("selected_project_key"),
                            )
                        except Exception as e:
                            st.error(f"Failed to rebuild agent graph after user switch: {e}")

            st.selectbox(
                label="Project",
                options=display_names,
                index=display_names.index(current_display_name) if current_display_name in display_names else 0,
                key="project_selector_display",
                help="Select which project database to query.",
                on_change=_on_project_change,
            )
            previous_user_id = st.session_state.get("selected_user_id")
            project_users = _sync_selected_user_for_project(current_project_key)
            if project_users:
                user_ids = [user.get("id") for user in project_users]
                user_label_by_id = {user.get("id"): user.get("display_name") for user in project_users}
                if st.session_state.get("user_selector_id") not in user_ids:
                    st.session_state.user_selector_id = st.session_state.get("selected_user_id", user_ids[0])
                if st.session_state.get("selected_user_id") not in user_ids:
                    st.session_state.selected_user_id = user_ids[0]
                if st.session_state.get("selected_user_id") != previous_user_id:
                    try:
                        st.session_state.global_hierarchy_access = setup.get_global_hierarchy_access(db=st.session_state.db)
                    except Exception:
                        pass
                st.selectbox(
                    label="User",
                    options=user_ids,
                    format_func=lambda user_id: user_label_by_id.get(user_id, str(user_id)),
                    key="user_selector_id",
                    help="Select which MissionOS user to query as.",
                    on_change=_on_user_change,
                )
        st.toggle(
            label="Show Sandbox Logs",
            key="show_sandbox_stream_logs",
            help="Toggle streaming of sandbox stdout/stderr in intermediate steps.",
        )
        plan_keys = list(parallel_plans.keys())
        # Uncomment to re-enable subscription plan selection
        # st.selectbox(
        #     label="Subscription Plan",
        #     options=plan_keys,
        #     format_func=lambda key: parallel_plans[key]["label"],
        #     key="parallel_plan",
        #     help="Economy: 3 agents • Reliable: 5 agents • Performance: 7 agents",
        #     on_change=_rebuild_graph,
        # )
        strategy_keys = list(completion_strategies.keys())
        # Uncomment to re-enable response mode selection
        # st.selectbox(
        #     label="Response Mode",
        #     options=strategy_keys,
        #     format_func=lambda key: completion_strategies[key]["label"],
        #     key="completion_strategy",
        #     help="Intelligent: consistency-based • Quick: first completion • Balanced: majority completion • Max: all agents complete",
        #     on_change=_rebuild_graph,
        # )
        if st.session_state.get("completion_strategy", "Intelligent") == "Intelligent":
            selected_plan_key = st.session_state.get("parallel_plan", "Performance")
            selected_plan = parallel_plans.get(selected_plan_key, parallel_plans["Performance"])
            max_success_needed = selected_plan.get("agents", 7)
            st.session_state.min_successful_responses = min(max_success_needed, st.session_state.min_successful_responses)
            st.slider(
                label="Minimum No. of Successful Responses to Await",
                min_value=1,
                max_value=max_success_needed,
                step=1,
                key="min_successful_responses",
                help="Stop early when consistency is stable after at least this many successful responses.",
                on_change=_rebuild_graph,
            )
            st.slider(
                label="Minimum Explained Variance",
                min_value=0.5,
                max_value=1.0,
                key="min_explained_variance",
                help="Regularised Explained Variance threshold for Intelligent mode termination.",
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

    chat_col, stop_col = st.columns([9, 1], vertical_alignment="bottom")
    with chat_col:
        st.chat_input(
            placeholder="Setting up, please wait...",
            disabled=True,
            key="initial_chat_input"
        )

def is_message_visible(message: AIMessage | AIMessageChunk, is_final: bool) -> bool:
    if isinstance(message, AIMessageChunk) and getattr(message, "usage_metadata", None):
        return False
    if message.content.strip() == "":
        return False
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
        'execution_initializer': 'build',
        'codeact_coder_branch': 'lightbulb',
        'codeact_executor_branch': 'sprint',
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
                    filename_response = st.session_state.llms['FAST'].invoke(prompt)
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

    chat_col, stop_col = st.columns([9, 1], vertical_alignment="bottom")

    question = None
    with chat_col:
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

        i = 0
        query_index = 0
        while i < len(st.session_state.get('messages', [])):
            message = st.session_state.messages[i]
            if isinstance(message, HumanMessage):
                with chat_col.chat_message("user"):
                    st.markdown(message.content)

                with chat_col.chat_message("assistant"):
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

        active_controller = st.session_state.get("active_run_controller")
        controller_busy = bool(active_controller and not getattr(active_controller, "is_cancelled", lambda: False)())
        pending_stop_button = st.session_state.get("stop_button_pending", False)
        input_disabled = controller_busy
        chat_placeholder = (
            "Ask a query about project data:"
            if not input_disabled
            else "Start a new chat to continue querying."
        )

        question = st.chat_input(
            placeholder=chat_placeholder,
            key="active_chat_input",
            disabled=input_disabled,
        )
    if question:
        st.session_state.stop_button_pending = True
        pending_stop_button = True

    with stop_col:
        stop_disabled = not (pending_stop_button or controller_busy)
        if st.button(
            label="",
            icon=":material/stop_circle:",
            key="stop_active_run_button",
            type="secondary",
            disabled=stop_disabled,
            help="Stop current response",
            use_container_width=True,
        ):
            _cancel_active_run("User pressed stop button")
            st.toast("Stopping current response...", icon=":material/stop_circle:")

    if question:
        user_message = HumanMessage(content=question, additional_kwargs={"type": "query"})
        st.session_state.messages.append(user_message)
        st.session_state.new_message_added = True
        with chat_col.chat_message("user"):
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
        logger.info(f"Type of Context(retrospective_query=''): {type(Context(retrospective_query=''))}")
        logger.info(f"Context(retrospective_query=''): {Context(retrospective_query='')}")
        initial_state = AgentState(
            messages=messages_summary,
            context=Context(retrospective_query="")
        )
        logger.info(f"Initial state: {initial_state}")

        controller = RunCancellationController(
            run_id=f"{st.session_state.thread_id}:{uuid.uuid4()}",
            user_id=st.session_state.selected_user_id,
        )
        token = activate_controller(controller)
        _set_active_run_controller(controller, token)

        stream = None
        try:
            with chat_col:
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
                            preparallel_previous_chunk_id = None
                            parallel_previous_message_types = [None for _ in range(st.session_state.num_parallel_executions)]
                            parallel_previous_chunk_ids = [None for _ in range(st.session_state.num_parallel_executions)]
                            parallel_code_block_open = [False for _ in range(st.session_state.num_parallel_executions)]
                            postparallel_previous_message_type = None
                            postparallel_previous_chunk_id = None

                        stream = st.session_state.graph.stream(initial_state, stream_mode="messages", config=config)
                        for state_update in stream:
                            if controller.is_cancelled():
                                raise RunCancelledError("Run cancelled by user")
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
                                        mc = (message.content or "")
                                        is_chunk = isinstance(message, AIMessageChunk)
                                        separator = ""
                                        if is_chunk:
                                            if preparallel_previous_chunk_id is not None and message.id != preparallel_previous_chunk_id:
                                                separator = "\n"
                                        elif preparallel_previous_message_type is AIMessageChunk:
                                            separator = "\n\n"
                                        else:
                                            separator = "\n\n"

                                        preparallel_current_query_steps.append(separator + mc)
                                        rendered_content = ""
                                        for step in preparallel_current_query_steps:
                                            rendered_content += step
                                        preparallel_thinking_container.markdown(rendered_content)
                                        preparallel_previous_message_type = type(message)
                                        preparallel_previous_chunk_id = message.id if isinstance(message, AIMessageChunk) else None
                                    case "parallel":
                                        branch_id = message.additional_kwargs.get("branch_id", 0)
                                        mc = (message.content or "")
                                        is_chunk = isinstance(message, AIMessageChunk)
                                        prev_chunk_id = parallel_previous_chunk_ids[branch_id]
                                        parts = []

                                        if is_chunk:
                                            is_import_line = bool(
                                                re.match(r"^\s*import\b", mc) or re.match(r"^\s*from\b.+\bimport\b", mc)
                                            )
                                            if is_import_line and not parallel_code_block_open[branch_id]:
                                                parts.append("\n\n```python\n")
                                                parallel_code_block_open[branch_id] = True
                                            if mc.startswith("```") and not parallel_code_block_open[branch_id]:
                                                parallel_code_block_open[branch_id] = True
                                            if parallel_code_block_open[branch_id] and prev_chunk_id is not None and message.id != prev_chunk_id:
                                                last_step = parallel_current_query_steps[branch_id][-1] if parallel_current_query_steps[branch_id] else ""
                                                logger.info(f"Last step before closing code block for branch {branch_id} for AIMessageChunk: {last_step}")
                                                if not last_step.rstrip().endswith("```"):
                                                    parts.append("\n```\n\n")
                                                else:
                                                    parts.append("\n\n")
                                                if is_import_line and not mc.startswith("```"):
                                                    parts.append("\n\n```python\n")
                                        else:
                                            if parallel_code_block_open[branch_id]:
                                                last_step = parallel_current_query_steps[branch_id][-1] if parallel_current_query_steps[branch_id] else ""
                                                logger.info(f"Last step before closing code block for branch {branch_id} for AIMessage: {last_step}")
                                                if not last_step.rstrip().endswith("```"):
                                                    parts.append("\n```\n\n")
                                                else:
                                                    parts.append("\n\n")
                                                parallel_code_block_open[branch_id] = False
                                            else:
                                                parts.append("\n\n")

                                        if mc != "":
                                            parts.append(mc)
                                        if parts:
                                            logger.info(f"parts for parallel branch {branch_id}: {parts}")
                                            parallel_current_query_steps[branch_id].append("".join(parts))
                                        rendered_content = ""
                                        for step in parallel_current_query_steps[branch_id]:
                                            rendered_content += step
                                        parallel_thinking_containers[branch_id].markdown(rendered_content)
                                        parallel_previous_message_types[branch_id] = type(message)
                                        parallel_previous_chunk_ids[branch_id] = message.id if isinstance(message, AIMessageChunk) else None
                                    case "postparallel":
                                        mc = (message.content or "")
                                        is_chunk = isinstance(message, AIMessageChunk)
                                        separator = ""
                                        if is_chunk:
                                            if postparallel_previous_chunk_id is not None and message.id != postparallel_previous_chunk_id:
                                                separator = "\n"
                                        elif postparallel_previous_message_type is AIMessageChunk:
                                            separator = "\n\n"
                                        else:
                                            separator = "\n\n"

                                        postparallel_current_query_steps.append(separator + mc)
                                        rendered_content = ""
                                        for step in postparallel_current_query_steps:
                                            rendered_content += step
                                        postparallel_thinking_container.markdown(rendered_content)
                                        postparallel_previous_message_type = type(message)
                                        postparallel_previous_chunk_id = message.id if isinstance(message, AIMessageChunk) else None

                    if len(st.session_state.intermediate_steps_history) > MAX_HISTORY // 2:
                        st.session_state.intermediate_steps_history = st.session_state.intermediate_steps_history[-MAX_HISTORY // 2:]
        except GeneratorExit:
            logger.info("Stream generator closed by Streamlit runtime; treating as cancelled.")
            st.info("Response stopped.", icon=":material/stop_circle:")
        except RunCancelledError:
            st.info("Response cancelled.", icon=":material/stop_circle:")
        except Exception as exc:
            logger.exception("Unexpected error during stream: %s", exc)
            st.error("Unexpected error while generating the response. Please retry.")
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            _clear_active_run_controller()