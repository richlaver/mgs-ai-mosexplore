import streamlit as st
import plotly.io as pio
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk

import json
import uuid
import logging
import re
import time
import threading
import queue
from typing import Any

from classes import AgentState, Context
from parameters import users, table_info
from graph import build_graph, drain_stream_messages, clear_stream_message_queue
import setup
import setup_modal
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

STREAM_MESSAGE_EMIT_INTERVAL_SECONDS = 0.1
CODE_YIELD_STEP_PREFIX_RE = re.compile(r"^\s*Step\s+\d+[^:]*:\s*")


def _strip_code_yield_step_prefix(content: str, metadata: dict) -> str:
    if not isinstance(content, str) or st.session_state.get("developer_view", False):
        return content
    origin = metadata.get("origin")
    if not isinstance(origin, dict) or origin.get("process") != "code_yield":
        return content
    return CODE_YIELD_STEP_PREFIX_RE.sub("", content, count=1)


def _fence_sandbox_sql(content: str, metadata: dict) -> str:
    if not isinstance(content, str):
        return content
    origin = metadata.get("origin")
    if not isinstance(origin, dict) or origin.get("process") != "sandbox_log":
        return content
    if "```sql" in content:
        return content
    sql_start_match = re.search(
        r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|MERGE|REPLACE)\b",
        content,
    )
    if not sql_start_match:
        return content
    start = sql_start_match.start()
    end = content.find(";", start)
    if end == -1:
        end = len(content)
    else:
        end += 1
    before = content[:start]
    sql = content[start:end]
    after = content[end:]
    return f"{before}\n```sql\n{sql}\n```\n{after}"


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


@st.dialog("Template Sandbox")
def template_sandbox_modal():
    st.markdown("Building E2B sandbox template...")
    with st.spinner("Building template (this can take a few minutes)..."):
        try:
            setup.build_e2b_sandbox_template()
        except Exception as exc:
            logger.error("E2B template build failed: %s", exc)
            st.error(f"E2B template build failed: {exc}")
        finally:
            st.rerun()

def new_chat() -> None:
    """Clear chat history and start a new chat by resetting session state variables."""
    st.session_state.clear_chat = True
    try:
        clear_stream_message_queue()
    except Exception:
        pass

def handle_clear_chat() -> None:
    """Handle clearing the chat if the clear_chat flag is set."""
    if st.session_state.get('clear_chat', False):
        st.session_state.clear_chat = False
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        try:
            clear_stream_message_queue()
        except Exception:
            pass

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
            margin-bottom: 1rem;
        }
        .chat-messages {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            padding-bottom: 10px;
            margin-bottom: 10px;
            margin-top: 10px;
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
        [data-testid="stExpander"] > details,
        [data-testid="stExpander"] > details > summary,
        [data-testid="stExpander"] > details > div {
            background-color: #FFE4CC;
            border: none !important;
            box-shadow: none !important;
            border-radius: 5px;
            padding: 10px;
        }
        [data-testid="stStatus"] {
            background-color: #FFE4CC;
            border: none !important;
            box-shadow: none !important;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
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
            stream_sandbox_logs=st.session_state.get("sandbox_logging", True),
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
                        st.session_state.modal_secrets = setup_modal.build_modal_secrets()
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
                    try:
                        setup.refresh_instrument_selection_cache(
                            selected_key,
                            st.session_state.llms["FAST"],
                        )
                    except Exception as e:
                        logger.warning("Failed to refresh instrument selection cache for %s: %s", selected_key, e)
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
                                stream_sandbox_logs=st.session_state.get("sandbox_logging", True),
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
                                stream_sandbox_logs=st.session_state.get("sandbox_logging", True),
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
                st.toggle(
                    label="Sandbox Logging",
                    key="sandbox_logging",
                    help="Toggle sandbox log streaming",
                    on_change=_rebuild_graph,
                )
                st.button(
                    label="Sandbox App",
                    icon=":material/developer_board:",
                    key="sandbox_button",
                    help="Manage Sandbox app and containers",
                    on_click=sandbox_modal,
                    use_container_width=True
                )
                if st.button(
                    label="Template Sandbox",
                    icon=":material/construction:",
                    key="template_sandbox_button",
                    help="Build the E2B sandbox template",
                    use_container_width=True,
                ):
                    template_sandbox_modal()
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

def _render_final_message(message: AIMessage) -> None:
    metadata = message.additional_kwargs or {}
    artefacts = metadata.get("artefacts")
    if not artefacts:
        if message.content:
            with st.container():
                st.markdown(message.content)
        return

    artefact_list = artefacts if isinstance(artefacts, list) else [artefacts]
    for artefact in artefact_list:
        if not isinstance(artefact, dict):
            continue
        artefact_type = artefact.get("type")
        artefact_id = artefact.get("id")
        if not artefact_id:
            continue
        if artefact_type == "plot":
            read_tool = ReadArtefactsTool(
                blob_db=st.session_state.blob_db,
                metadata_db=st.session_state.metadata_db,
            )
            result = read_tool._run(metadata_only=False, artefact_ids=[artefact_id])
            if result['success'] and result['artefacts']:
                artefact_entry = result['artefacts'][0]
                blob = artefact_entry['blob']
                try:
                    fig_json = json.loads(blob.decode('utf-8'))
                    fig = pio.from_json(json.dumps(fig_json))
                    with st.container():
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{artefact_id}")
                except Exception as e:
                    logger.error(f"Failed to render Plotly figure: {str(e)}")
                    st.error("Error rendering plot")
        elif artefact_type == "csv":
            read_tool = ReadArtefactsTool(
                blob_db=st.session_state.blob_db,
                metadata_db=st.session_state.metadata_db,
            )
            result = read_tool._run(metadata_only=False, artefact_ids=[artefact_id])
            if result['success'] and result['artefacts']:
                artefact_entry = result['artefacts'][0]
                blob = artefact_entry['blob'].decode('utf-8')
                desc = artefact_entry['metadata']['description_text']
                prompt = (
                    "Generate a short, descriptive filename for this CSV based on the "
                    f"description: {desc}. Do not include the .csv extension."
                )
                filename_response = st.session_state.llms['FAST'].invoke(prompt)
                filename = filename_response.content.strip() + '.csv'
                with st.container():
                    st.download_button(
                        label=f"Download {filename}",
                        data=blob,
                        file_name=filename,
                        mime='text/csv',
                        key=f"csv_download_{artefact_id}",
                    )

def _run_graph_stream_worker(stream_queue: queue.Queue, graph, initial_state, config, controller) -> None:
    stream = None
    try:
        stream = graph.stream(initial_state, stream_mode="messages", config=config)
        for state_update in stream:
            if controller.is_cancelled():
                break
            stream_queue.put(("update", state_update))
    except Exception as exc:
        stream_queue.put(("error", exc))
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        stream_queue.put(("done", None))

def render_chat_content() -> None:
    """Renders chat messages, history, and enabled chat input after setup is complete."""
    if not st.session_state.setup_complete:
        return

    handle_clear_chat()

    chat_col, stop_col = st.columns([9, 1], vertical_alignment="bottom")

    question = None
    with chat_col:
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

        messages = st.session_state.get("messages", [])

        def _consume_assistant_messages(start_index: int) -> int:
            idx = start_index
            while idx < len(messages) and isinstance(messages[idx], (AIMessage, AIMessageChunk)):
                if isinstance(messages[idx], AIMessage):
                    _render_final_message(messages[idx])
                idx += 1
            return idx

        i = 0
        query_index = 0
        while i < len(messages):
            message = messages[i]
            if isinstance(message, HumanMessage):
                with chat_col.chat_message("user"):
                    st.markdown(message.content)
                with chat_col.chat_message("assistant"):
                    i = _consume_assistant_messages(i + 1)
                query_index += 1
                continue
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

        try:
            with chat_col:
                with st.chat_message("assistant"):
                    status_container_placeholder = st.empty()
                    status_container = status_container_placeholder.container()
                    status_by_stage: dict[int, Any] = {}
                    status_child_slots: dict[int, Any] = {}
                    parallel_status_child_slots: dict[int, list[Any]] = {}
                    parallel_status_child_buffers: dict[int, list[list[str]]] = {}
                    current_parent_stage: int | None = None

                    def _complete_status(stage: int | None) -> None:
                        if stage is None:
                            return
                        status = status_by_stage.get(stage)
                        if status:
                            try:
                                status.update(state="complete", expanded=False)
                            except Exception:
                                try:
                                    status.update(state="complete")
                                except Exception:
                                    pass

                    def _complete_all_statuses() -> None:
                        for stage in list(status_by_stage.keys()):
                            _complete_status(stage)

                    def _clear_status_container(force: bool = False) -> None:
                        try:
                            status_container_placeholder.empty()
                        except Exception:
                            pass
                        try:
                            st.markdown(
                                """
                                <style>
                                [data-testid="stStatus"],
                                [data-testid="stExpander"]:has([data-testid^="stExpanderIcon"]) {
                                    display: none !important;
                                    visibility: hidden !important;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            pass
                        status_by_stage.clear()
                        status_child_slots.clear()
                        parallel_status_child_slots.clear()
                        parallel_status_child_buffers.clear()

                    def _get_stage(metadata: dict) -> int | None:
                        origin = metadata.get("origin")
                        if not isinstance(origin, dict):
                            return None
                        stage = origin.get("thinking_stage")
                        return stage if isinstance(stage, int) else None

                    def _render_parent_message(content: str, metadata: dict) -> None:
                        nonlocal current_parent_stage
                        stage = _get_stage(metadata)
                        if stage is None:
                            stage = 0
                        if current_parent_stage is not None and current_parent_stage != stage:
                            _complete_status(current_parent_stage)
                        current_parent_stage = stage
                        if stage in status_by_stage:
                            status_by_stage[stage].update(label=content)
                        else:
                            with status_container:
                                status = st.status(content, expanded=True, state="running")
                            status_by_stage[stage] = status
                            status_child_slots[stage] = status.empty()

                    def _render_child_message(content: str, metadata: dict) -> None:
                        stage = _get_stage(metadata)
                        if stage is None:
                            stage = 0
                        content = _strip_code_yield_step_prefix(content, metadata)
                        content = _fence_sandbox_sql(content, metadata)
                        if stage not in status_by_stage:
                            _render_parent_message(f"Stage {stage}", {"origin": {"thinking_stage": stage}})
                        status = status_by_stage.get(stage)

                        branch_id = metadata.get("origin", {}).get("branch_id")
                        developer_view = st.session_state.get("developer_view", False)
                        can_render_parallel = (
                            developer_view
                            and branch_id is not None
                            and isinstance(branch_id, int)
                            and stage in (2, 3)
                        )

                        if can_render_parallel:
                            parallel_slots = parallel_status_child_slots.get(stage)
                            parallel_buffers = parallel_status_child_buffers.get(stage)
                            if parallel_slots is None:
                                cols = status.columns(st.session_state.num_parallel_executions)
                                parallel_slots = []
                                for col in cols:
                                    parallel_slots.append(col.empty())
                                parallel_status_child_slots[stage] = parallel_slots
                                parallel_buffers = [[] for _ in range(st.session_state.num_parallel_executions)]
                                parallel_status_child_buffers[stage] = parallel_buffers

                            if parallel_buffers is None:
                                parallel_buffers = [[] for _ in range(len(parallel_slots))]
                                parallel_status_child_buffers[stage] = parallel_buffers

                            if 0 <= branch_id < len(parallel_slots):
                                parallel_buffers[branch_id].append(content)
                                with parallel_slots[branch_id].container():
                                    for message in parallel_buffers[branch_id]:
                                        with st.container():
                                            st.markdown(message)
                                return

                        child_slot = status_child_slots.get(stage)
                        if child_slot is None:
                            child_slot = status.empty()
                            status_child_slots[stage] = child_slot
                        child_slot.markdown(content)

                    clear_stream_message_queue()

                    stream_queue: queue.Queue = queue.Queue()
                    worker = threading.Thread(
                        target=_run_graph_stream_worker,
                        args=(stream_queue, st.session_state.graph, initial_state, config, controller),
                        daemon=True,
                    )
                    worker.start()

                    pending_payload = None
                    pending_emit_at = None
                    stream_done = False

                    while True:
                        if controller.is_cancelled():
                            raise RunCancelledError("Run cancelled by user")

                        try:
                            kind, payload = stream_queue.get(timeout=0.2)
                            if kind == "update":
                                # logger.debug(f"Raw state update in ui.py: {str(payload)[:100]}")
                                logger.debug(f"Raw state update in ui.py: {str(payload)}")
                                if isinstance(payload, tuple) and len(payload) == 2:
                                    message, metadata = payload
                                else:
                                    message, metadata = None, None
                                skip_message = False
                                if isinstance(metadata, dict):
                                    context_payload = metadata.get("context")
                                    if isinstance(context_payload, dict) and context_payload.get("ls_provider"):
                                        skip_message = True
                                    if metadata.get("ls_provider"):
                                        skip_message = True

                                if not skip_message and not isinstance(message, (AIMessage, AIMessageChunk)):
                                    skip_message = True
                                if not skip_message and isinstance(message, AIMessageChunk) and getattr(message, "usage_metadata", None):
                                    skip_message = True

                                if not skip_message:
                                    msg_metadata = message.additional_kwargs or {}
                                    if not st.session_state.get("developer_view", False):
                                        if msg_metadata.get("level") != "info":
                                            skip_message = True

                                if not skip_message:
                                    if msg_metadata.get("is_final") is True:
                                        _complete_all_statuses()
                                        if not st.session_state.get("developer_view", False):
                                            _clear_status_container()
                                        st.session_state.messages.append(message)
                                        _render_final_message(message)
                                    elif msg_metadata.get("is_child") is True:
                                        _render_child_message(message.content, msg_metadata)
                                    else:
                                        _render_parent_message(message.content, msg_metadata)
                                        pending_payload = None
                                        pending_emit_at = None
                            elif kind == "error":
                                raise payload
                            elif kind == "done":
                                stream_done = True
                        except queue.Empty:
                            pass

                        queued_messages = drain_stream_messages()
                        for queued in queued_messages:
                            logger.debug(f"Raw queued message in ui.py: {str(queued)}")
                            queued_metadata = queued.get("additional_kwargs") or {}
                            skip_queued = False
                            if not st.session_state.get("developer_view", False):
                                if queued_metadata.get("level") != "info":
                                    skip_queued = True

                            if not skip_queued:
                                if queued_metadata.get("is_final") is True:
                                    _complete_all_statuses()
                                    if not st.session_state.get("developer_view", False):
                                        _clear_status_container()
                                    final_message = AIMessage(content=queued.get("content") or "", additional_kwargs=queued_metadata)
                                    _render_final_message(final_message)
                                    pending_payload = None
                                    pending_emit_at = None
                                elif queued_metadata.get("origin", {}).get("branch_id") is not None:
                                    _render_child_message(queued.get("content") or "", queued_metadata)
                                    pending_payload = None
                                    pending_emit_at = None
                                elif queued_metadata.get("is_child") is True:
                                    pending_payload = queued
                                    pending_emit_at = (queued.get("timestamp") or time.monotonic()) + STREAM_MESSAGE_EMIT_INTERVAL_SECONDS
                                else:
                                    _render_parent_message(queued.get("content") or "", queued_metadata)
                                    pending_payload = None
                                    pending_emit_at = None

                        if pending_payload and pending_emit_at is not None and time.monotonic() >= pending_emit_at:
                            pending_metadata = pending_payload.get("additional_kwargs") or {}
                            _render_child_message(pending_payload.get("content") or "", pending_metadata)
                            pending_payload = None
                            pending_emit_at = None

                        if stream_done and stream_queue.empty() and pending_payload is None:
                            break

                    if pending_payload:
                        pending_metadata = pending_payload.get("additional_kwargs") or {}
                        _render_child_message(pending_payload.get("content") or "", pending_metadata)
                    
        except GeneratorExit:
            logger.info("Stream generator closed by Streamlit runtime; treating as cancelled.")
            st.info("Response stopped.", icon=":material/stop_circle:")
        except RunCancelledError:
            st.info("Response cancelled.", icon=":material/stop_circle:")
        except Exception as exc:
            logger.exception("Unexpected error during stream: %s", exc)
            st.error("Unexpected error while generating the response. Please retry.")
        finally:
            _clear_active_run_controller()