import streamlit as st
import modal
import subprocess
import time
import logging

import modal_sandbox_remote
from graph import build_graph
from parameters import table_info

logger = logging.getLogger(__name__)

MOCK_TABLE_INFO = []
MOCK_TABLE_REL_GRAPH = {}
MOCK_THREAD_ID = "1"
MOCK_USER_ID = 1
MOCK_GLOBAL_ACCESS = False
MOCK_CODE = """
def execute_strategy():
    yield {"type": "warmup", "content": "Container warmed"}
"""

app_ref = modal_sandbox_remote.app

def make_local_sandbox_mode():
    current_sandbox_mode = st.session_state.get("sandbox_mode")
    if current_sandbox_mode != "Local":
        st.session_state.sandbox_mode = "Local"
        st.session_state.update({'graph': build_graph(
            llm=st.session_state.llm,
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

# Check if container is warm (response time < 3s)
def is_container_warm():
    logger.info("Checking if container is warm (max 3s)...")
    deployed = is_app_deployed()
    logger.info(f"is_app_deployed: {deployed}")
    if not deployed:
        return False

    start_time = time.time()
    TIMEOUT = 3.0
    warm_detected = False

    try:
        for output in modal_sandbox_remote.execute_remote_sandbox(
            code=MOCK_CODE,
            table_info=MOCK_TABLE_INFO,
            table_relationship_graph=MOCK_TABLE_REL_GRAPH,
            thread_id=MOCK_THREAD_ID,
            user_id=MOCK_USER_ID,
            global_hierarchy_access=MOCK_GLOBAL_ACCESS,
        ):
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT:
                logger.warning(f"Warm check exceeded {TIMEOUT}s (elapsed={elapsed:.2f}); aborting and returning False")
                return False
            logger.info(f"Container output from is_container_warm: {output}")
            out_type = output.get("type")
            content = output.get("content", "") or ""
            normalized = content.lower()
            if out_type == "error" and ("not deployed" in normalized or "is not deployed" in normalized):
                logger.info("Detected 'not deployed' error in sandbox output; returning False")
                return False
            if out_type in ("warmup", "progress", "final"):
                warm_detected = True
        total_elapsed = time.time() - start_time
        logger.info(f"Container warm check completed in {total_elapsed:.2f}s warm_detected={warm_detected}")
        if total_elapsed > TIMEOUT:
            logger.info("Total elapsed exceeded timeout; returning False")
            return False
        return warm_detected and total_elapsed < TIMEOUT
    except Exception as e:
        logger.exception(f"Warm check failed: {e}")
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
        except Exception as e:
            st.session_state.app_deployed = False
            st.session_state.app_id = None
            st.toast(f"Deployment failed: {str(e)}", icon=":material/error:")
            raise

# Warm up container
def warm_up_container():
    if not is_app_deployed():
        st.error("Cannot warm container: App 'mgs-code-sandbox' is not deployed. Use 'Spin-up' to deploy.")
        return
    with st.spinner("Warming up container..."):
        try:
            for output in modal_sandbox_remote.execute_remote_sandbox(
                code=MOCK_CODE,
                table_info=MOCK_TABLE_INFO,
                table_relationship_graph=MOCK_TABLE_REL_GRAPH,
                thread_id=MOCK_THREAD_ID,
                user_id=MOCK_USER_ID,
                global_hierarchy_access=MOCK_GLOBAL_ACCESS,
            ):
                logger.info(f"Container output from warm_up_container: {output}")
                if output.get("type") == "error" and "App 'mgs-code-sandbox' is not deployed" in output.get("content", ""):
                    st.error("Cannot warm container: App not deployed.")
                    st.session_state.container_warm = False
                    return
            st.session_state.container_warm = True
            st.toast("Container warmed successfully.", icon=":material/mode_heat:")
        except Exception as e:
            st.session_state.container_warm = False
            st.toast(f"Warm-up failed: {str(e)}", icon=":material/error:")
            raise

# Stop app
def stop_app():
    app_id = st.session_state.get("app_id", None)
    with st.spinner("Stopping app and all containers..."):
        try:
            subprocess.run(["modal", "app", "stop", app_id], check=True)
            st.session_state.app_deployed = False
            st.session_state.container_warm = False
            st.session_state.app_id = None
            # make_local_sandbox_mode()
            st.toast("App and containers stopped successfully.", icon=":material/block:")
        except Exception as e:
            st.toast(f"Stop failed: {str(e)}. Use CLI: modal app stop {app_id}", icon=":material/error:")
            raise