import streamlit as st
import modal
import subprocess
import time
import logging
import json
from typing import Optional

import modal_sandbox_remote
from graph import build_graph
from parameters import table_info

logger = logging.getLogger(__name__)

app_ref = modal_sandbox_remote.app

MOCK_TABLE_INFO = []
MOCK_TABLE_REL_GRAPH = {}
MOCK_THREAD_ID = "1"
MOCK_USER_ID = 1
MOCK_GLOBAL_ACCESS = False
MOCK_CODE = """
def execute_strategy():
    yield {"type": "warmup", "content": "Container warmed"}
"""

def make_local_sandbox_mode():
    current_sandbox_mode = st.session_state.get("sandbox_mode")
    if current_sandbox_mode != "Local":
        st.session_state.sandbox_mode = "Local"
        st.session_state.update({'graph': build_graph(
            llms=st.session_state.llms,
            db=st.session_state.db,
            blob_db=st.session_state.blob_db,
            metadata_db=st.session_state.metadata_db,
            table_info=table_info,
            table_relationship_graph=st.session_state.table_relationship_graph, 
            thread_id=st.session_state.thread_id,
            user_id=st.session_state.selected_user_id,
            global_hierarchy_access=st.session_state.global_hierarchy_access,
            remote_sandbox=False,
            num_parallel_executions=st.session_state.num_parallel_executions,
            num_completions_before_response=st.session_state.num_completions_before_response,
            agent_type=st.session_state.agent_type,
            selected_project_key=st.session_state.get("selected_project_key"),
        )})

# Check if app is deployed using Modal API
def is_app_deployed():
    try:
        app = modal.App.lookup("mgs-code-sandbox", environment_name=None)
        st.session_state.app_id = app.app_id
        logger.info(f"App status: {app}")
        logger.info(f"App ID: {app.app_id}")
        logger.info(f"App Description: {app.description}")
        return app is not None
    except modal.exception.ExecutionError:
        # make_local_sandbox_mode()
        return False
    except Exception as e:
        # make_local_sandbox_mode()
        # Comment the line below to fail silently when running Streamlit locally
        # st.error(f"Failed to check app status: {str(e)}")
        return False

 

# Deploy app
def deploy_app():
    with st.spinner("Deploying Modal app... This may take 1-5 minutes."):
        try:
            deployment = app_ref.deploy()
            logger.info(f"Deployment result: {deployment}")
            st.session_state.app_deployed = True
            st.session_state.app_id = deployment.app_id if hasattr(deployment, 'app_id') else "unknown"
            st.toast(f"App deployed successfully! App ID: {st.session_state.app_id}", icon=":material/rocket_launch:")
            st.session_state.container_warm = False
            st.session_state.container_warm_count = 0
            warm_up_container()
        except Exception as e:
            st.session_state.app_deployed = False
            st.session_state.app_id = None
            st.toast(f"Deployment failed: {str(e)}", icon=":material/error:")
            raise

 

# Stop app
def stop_app():
    app_id = st.session_state.get("app_id", None)
    with st.spinner("Stopping app and all containers..."):
        try:
            subprocess.run(["modal", "app", "stop", app_id], check=True)
            st.session_state.app_deployed = False
            st.session_state.container_warm = False
            st.session_state.container_warm_count = 0
            st.session_state.app_id = None
            # make_local_sandbox_mode()
            st.toast("App and containers stopped successfully.", icon=":material/block:")
        except Exception as e:
            st.toast(f"Stop failed: {str(e)}. Use CLI: modal app stop {app_id}", icon=":material/error:")
            raise


def restart_container_for_project_change():
    """Stop, reload module to refresh secrets, redeploy, and warm the container.

    Use this when the selected project changes so the remote sandbox picks up new DB creds.
    """
    import importlib
    global app_ref
    logger.info("[Modal] Restart requested due to project change")

    try:
        if is_app_deployed():
            logger.info("[Modal] Stopping existing app before redeploy...")
            stop_app()
    except Exception as e:
        logger.warning("[Modal] Ignoring stop error during restart: %s", e)

    try:
        logger.info("[Modal] Reloading modal_sandbox_remote module to refresh secrets")
        importlib.reload(modal_sandbox_remote)
        app_ref = modal_sandbox_remote.app
    except Exception as e:
        logger.error("[Modal] Failed to reload modal_sandbox_remote: %s", e)
        raise

    try:
        deploy_app()
        logger.info("[Modal] Restart complete: app deployed and container warm")
    except Exception as e:
        logger.error("[Modal] Restart failed: %s", e)
        raise


def _list_app_containers(app_id: Optional[str]) -> list[dict]:
    """Return running containers for the given app id using the Modal CLI."""

    if not app_id:
        return []
    try:
        result = subprocess.run(
            ["modal", "container", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        containers = json.loads(result.stdout)
        return [c for c in containers if c.get("App ID") == app_id]
    except Exception as exc:
        logger.warning("[Modal] Failed to list containers for app %s: %s", app_id, exc)
        return []


def _stop_extra_containers(app_id: Optional[str], keep: int) -> int:
    """Stop containers above the desired count; returns how many were stopped."""

    if keep < 0:
        keep = 0
    containers = _list_app_containers(app_id)
    if not containers:
        return 0

    # Prefer keeping the oldest containers (likely already warm/busy).
    containers.sort(key=lambda c: c.get("Start Time", ""))
    extra = containers[keep:]
    stopped = 0
    for container in extra:
        container_id = container.get("Container ID")
        if not container_id:
            continue
        try:
            subprocess.run(["modal", "container", "stop", container_id], check=True)
            stopped += 1
            logger.info("[Modal] Stopped extra container: %s", container_id)
        except Exception as exc:
            logger.warning("[Modal] Failed to stop container %s: %s", container_id, exc)
    return stopped


def _stop_all_containers(app_id: Optional[str]) -> int:
    """Stop all running containers for the app; returns how many were stopped."""

    containers = _list_app_containers(app_id)
    stopped = 0
    for container in containers:
        container_id = container.get("Container ID")
        if not container_id:
            continue
        try:
            subprocess.run(["modal", "container", "stop", container_id], check=True)
            stopped += 1
            logger.info("[Modal] Stopped container: %s", container_id)
        except Exception as exc:
            logger.warning("[Modal] Failed to stop container %s: %s", container_id, exc)
    return stopped


def _update_autoscaler_for_all(target: int) -> None:
    """Set per-class autoscaler targets with priority A -> B -> C."""

    class_names = ["SandboxExecutorA", "SandboxExecutorB", "SandboxExecutorC"]

    def _desired_class_counts(n: int) -> dict[str, int]:
        alloc = {"SandboxExecutorA": 0, "SandboxExecutorB": 0, "SandboxExecutorC": 0}
        remaining = max(n, 0)
        for name in class_names:
            if remaining <= 0:
                break
            alloc[name] = 1
            remaining -= 1
        return alloc

    allocations = _desired_class_counts(target)

    for name, count in allocations.items():
        try:
            cls = modal.Cls.from_name("mgs-code-sandbox", name)
            obj = cls()
            obj.update_autoscaler(  # type: ignore[attr-defined]
                min_containers=count,
                max_containers=count,
                buffer_containers=0,
            )
            logger.info("[Modal] Autoscaler updated | class=%s min=%s max=%s", name, count, count)
        except Exception as exc:
            logger.warning("[Modal] Failed to update autoscaler for %s: %s", name, exc)


def warm_up_container():
    """Execute a trivial remote sandbox strategy to trigger container spin-up and mark warm status.

    Only runs if the app is deployed. Sets st.session_state.container_warm accordingly.
    """
    if not st.session_state.get("app_deployed"):
        logger.info("Skipping warm_up_container: app not deployed")
        return
    app_id = st.session_state.get("app_id")
    target = int(st.session_state.get("num_parallel_executions", 1) or 0)

    _update_autoscaler_for_all(target)

    actual_containers = _list_app_containers(app_id)
    current_warm = len(actual_containers)
    st.session_state.container_warm_count = current_warm
    st.session_state.container_warm = current_warm > 0

    if target == current_warm:
        logger.info("Skipping warm_up_container: target=%s matches current=%s", target, current_warm)
        return

    if target < current_warm:
        with st.spinner("Reducing warm containers..."):
            _stop_all_containers(app_id)
            refreshed = _list_app_containers(app_id)
            st.session_state.container_warm_count = len(refreshed)
            st.session_state.container_warm = st.session_state.container_warm_count > 0
            logger.info("[Modal] Stopped all containers to enforce ordered allocation | before=%s after=%s target=%s", current_warm, st.session_state.container_warm_count, target)
        # Reset warm count for deterministic re-warm below.
        current_warm = 0

    with st.spinner("Warming up containers..."):
        try:
            for slot in range(current_warm, target):
                for output in modal_sandbox_remote.execute_remote_sandbox(
                    code=MOCK_CODE,
                    table_info=MOCK_TABLE_INFO,
                    table_relationship_graph=MOCK_TABLE_REL_GRAPH,
                    thread_id=MOCK_THREAD_ID,
                    user_id=MOCK_USER_ID,
                    global_hierarchy_access=MOCK_GLOBAL_ACCESS,
                    selected_project_key=st.session_state.get("selected_project_key"),
                    container_slot=slot,
                ):
                    logger.info(f"Container[{slot}] output from warm_up_container: {output}")
                    if output.get("type") == "error" and "not deployed" in (output.get("content", "") or "").lower():
                        st.error("Cannot warm container: App not deployed.")
                        st.session_state.container_warm = False
                        st.session_state.container_warm_count = current_warm
                        return
            refreshed = _list_app_containers(app_id)
            new_count = len(refreshed)
            st.session_state.container_warm = new_count > 0
            st.session_state.container_warm_count = new_count
            if new_count >= target:
                st.toast(f"{target} container(s) warmed successfully.", icon=":material/mode_heat:")
            else:
                st.toast(f"Warmed {new_count}/{target} container(s). Check Modal console.", icon=":material/warning:")
        except Exception as e:
            st.session_state.container_warm = False
            st.session_state.container_warm_count = current_warm
            st.toast(f"Warm-up failed: {str(e)}", icon=":material/error:")
            logger.exception("Warm-up failed")
            raise