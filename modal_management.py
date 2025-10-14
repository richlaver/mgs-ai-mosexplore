import streamlit as st
import modal
import subprocess
import time
import logging
import modal_sandbox

logger = logging.getLogger(__name__)

MOCK_TABLE_INFO = []
MOCK_TABLE_REL_GRAPH = {}
MOCK_THREAD_ID = 1
MOCK_USER_ID = 1
MOCK_GLOBAL_ACCESS = False
MOCK_CODE = """
def execute_strategy():
    yield {"type": "warmup", "content": "Container warmed"}
"""

app_ref = modal_sandbox.app

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
        return False
    except Exception as e:
        st.error(f"Failed to check app status: {str(e)}")
        return False

# Check if container is warm (response time < 3s)
def is_container_warm():
    logger.info(f"Checking if container is warm...")
    logger.info(f"is_app_deployed: {is_app_deployed()}")
    if not is_app_deployed():
        return False
    start_time = time.time()
    try:
        for output in modal_sandbox.run_with_live_logs(
            code=MOCK_CODE,
            table_info=MOCK_TABLE_INFO,
            table_relationship_graph=MOCK_TABLE_REL_GRAPH,
            thread_id=MOCK_THREAD_ID,
            user_id=MOCK_USER_ID,
            global_hierarchy_access=MOCK_GLOBAL_ACCESS,
        ):
            logger.info(f"Container output from is_container_warm: {output}")
            if output.get("type") == "error" and "App 'mgs-code-sandbox' is not deployed" in output.get("content", ""):
                return False
        elapsed = time.time() - start_time
        logger.info(f"Container response time: {elapsed} seconds")
        return elapsed < 3
    except Exception:
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
            for output in modal_sandbox.run_with_live_logs(
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
            st.toast("App and containers stopped successfully.", icon=":material/block:")
        except Exception as e:
            st.toast(f"Stop failed: {str(e)}. Use CLI: modal app stop {app_id}", icon=":material/error:")
            raise