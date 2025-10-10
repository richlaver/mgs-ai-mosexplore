import sys
import uuid
import modal
import logging
import json
import os
import time
from typing import Generator, List, Dict, Tuple, Optional, Callable
import threading

import streamlit as st

from setup import build_modal_secrets

# Define the Modal App and the container Image for the sandbox (Modal 1.0 style).
# Local code/assets are added to the Image (mounts on decorators are deprecated).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "langchain_community",
        "langchain-google-vertexai",
        "sqlalchemy",
        "langchain-core",
        "langgraph",
        "sqlparse",
        "pydantic",
        "typing_extensions",
        "mysql-connector-python",
        "geoalchemy2",
        "numpy",
        "pandas",
        "shapely",
        "scipy",
        "scikit-learn",
        "b2sdk",
        "psycopg2-binary"
    )
    # Make local modules/files available inside the container at runtime
    .add_local_dir("agents", remote_path="/root/agents")
    .add_local_file("classes.py", remote_path="/root/classes.py")
    .add_local_file("parameters.py", remote_path="/root/parameters.py")
    .add_local_file("timeseries_plot_sandbox_agent.py", remote_path="/root/timeseries_plot_sandbox_agent.py")
    .add_local_file("map_plot_sandbox_agent.py", remote_path="/root/map_plot_sandbox_agent.py")
    .add_local_file("artefact_management.py", remote_path="/root/artefact_management.py")
    .add_local_file("artefact_tools.py", remote_path="/root/artefact_tools.py")
)

app = modal.App("mgs-code-sandbox")

# Module-level logger
logger = logging.getLogger("modal_sandbox")

def _ensure_basic_logging():
    """Ensure a reasonable default logging config if none exists."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

@app.function(
    image=image,
    secrets=build_modal_secrets(),
)
def run_sandboxed_code(
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    user_id: int,
    global_hierarchy_access: bool,
) -> Generator[dict, None, None]:
    """
    Executes the provided Python code string in a secure, remote Modal sandbox.

    This function runs in a containerized environment on Modal's servers, ensuring that the dynamically generated code is completely isolated from the main application and local system.

    Args:
        code: The Python code string to execute. Must define an 'execute_strategy' function that yields dicts (JSON-like outputs).
              Required signature: execute_strategy() taking no arguments.

    Yields:
        Dictionaries representing the JSON outputs from the executed code, or error dicts if failed.
    """
    import sys
    import os
    import psycopg2
    import b2sdk.v1 as b2
    from tools.artefact_toolkit import WriteArtefactTool, ReadArtefactsTool, DeleteArtefactsTool
    _ensure_basic_logging()
    run_logger = logging.getLogger("modal_sandbox.run")
    run_logger.info("Starting sandbox execution | user_id=%s global_access=%s code_len=%s table_info=%s rel_graph_keys=%s",
                    user_id, global_hierarchy_access, len(code) if isinstance(code, str) else None,
                    len(table_info) if isinstance(table_info, list) else None,
                    list(table_relationship_graph.keys()) if isinstance(table_relationship_graph, dict) else None)
    from langchain_community.utilities import SQLDatabase
    from agents.extraction_sandbox_agent import extraction_sandbox_agent
    from agents.timeseries_plot_sandbox_agent import timeseries_plot_sandbox_agent
    from agents.map_plot_sandbox_agent import map_plot_sandbox_agent
    from tools.sql_security_toolkit import GeneralSQLQueryTool
    from langchain_google_vertexai import ChatVertexAI
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer
    from geoalchemy2 import Geometry
    from parameters import include_tables

    # Ensure added local files are importable inside the container
    sys.path.append("/root")

    try:
        # Set up Google credentials from the Modal secret
        try:
            credentials_json = os.environ["GOOGLE_CREDENTIALS_JSON"]
            run_logger.info("Found GOOGLE_CREDENTIALS_JSON in environment (len=%s)", len(credentials_json))
        except KeyError:
            run_logger.error("GOOGLE_CREDENTIALS_JSON not found in environment")
            yield {"type": "error", "content": "Error: Google credentials secret not found. Ensure 'google-credentials-json' is set via Modal CLI."}
            return

        temp_credentials_path = "google_credentials.json"
        with open(temp_credentials_path, "w") as f:
            f.write(credentials_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path
        run_logger.info("Wrote temporary Google credentials to %s", temp_credentials_path)

        # Parse project_id from credentials
        project_id = json.loads(credentials_json).get("project_id")
        if not project_id:
            raise ValueError("project_id not found in GOOGLE_CREDENTIALS_JSON")
        run_logger.info("Parsed project_id: %s", project_id)

        # Set up metadata_db (PostgreSQL RDS)
        try:
            required_rds_env = ["ARTEFACT_METADATA_RDS_HOST", "ARTEFACT_METADATA_RDS_USER", "ARTEFACT_METADATA_RDS_PASS", "ARTEFACT_METADATA_RDS_DATABASE"]
            missing_rds_env = [k for k in required_rds_env if k not in os.environ or not os.environ.get(k)]
            rds_port = os.environ.get("ARTEFACT_METADATA_RDS_PORT", "5432")
            if missing_rds_env:
                run_logger.error("Missing RDS env vars: %s", missing_rds_env)
                raise KeyError("Missing required RDS environment variables: " + ", ".join(missing_rds_env))

            metadata_db = psycopg2.connect(
                host=os.environ["ARTEFACT_METADATA_RDS_HOST"],
                user=os.environ["ARTEFACT_METADATA_RDS_USER"],
                password=os.environ["ARTEFACT_METADATA_RDS_PASS"],
                database=os.environ["ARTEFACT_METADATA_RDS_DATABASE"],
                port=rds_port
            )
            run_logger.info("Connected to PostgreSQL RDS metadata_db")
        except Exception as e:
            run_logger.error("Failed to connect to RDS: %s", e)
            yield {"type": "error", "content": f"Error connecting to metadata database: {str(e)}"}
            return

        # Set up blob_db (Backblaze B2)
        try:
            required_b2_env = ["ARTEFACT_BLOB_B2_KEY_ID", "ARTEFACT_BLOB_B2_KEY", "ARTEFACT_BLOB_B2_BUCKET"]
            missing_b2_env = [k for k in required_b2_env if k not in os.environ or not os.environ.get(k)]
            if missing_b2_env:
                run_logger.error("Missing B2 env vars: %s", missing_b2_env)
                raise KeyError("Missing required B2 environment variables: " + ", ".join(missing_b2_env))

            info = b2.InMemoryAccountInfo()
            b2_api = b2.B2Api(info)
            b2_api.authorize_account("production", os.environ["ARTEFACT_BLOB_B2_KEY_ID"], os.environ["ARTEFACT_BLOB_B2_KEY"])
            blob_db = b2_api.get_bucket_by_name(os.environ["ARTEFACT_BLOB_B2_BUCKET"])
            run_logger.info("Authorized and got B2 bucket: %s", os.environ["ARTEFACT_BLOB_B2_BUCKET"])
        except Exception as e:
            run_logger.error("Failed to set up B2: %s", e)
            yield {"type": "error", "content": f"Error setting up Backblaze B2: {str(e)}"}
            return

        # Reconstruct the necessary environment within the sandbox
        llm = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0.1)
        run_logger.info("Initialized ChatVertexAI model=%s", "gemini-2.0-flash-001")

        # Initialize the SQL database connection (MySQL, mirroring setup.get_db)
        try:
            required_env = ["DB_HOST", "DB_USER", "DB_PASS", "DB_NAME"]
            missing_env = [k for k in required_env if k not in os.environ or not os.environ.get(k)]
            db_port = os.environ.get("DB_PORT", "3306")
            if missing_env:
                run_logger.error("Missing DB env vars: %s | Present: %s", missing_env, [k for k in required_env if k in os.environ])
                raise KeyError("Missing required DB environment variables: " + ", ".join(missing_env))

            db_host = os.environ["DB_HOST"]
            db_user = os.environ["DB_USER"]
            db_pass = os.environ["DB_PASS"]
            db_name = os.environ["DB_NAME"]

            db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            run_logger.info("Creating SQLAlchemy engine for MySQL (host=%s port=%s db=%s user=%s)", db_host, db_port, db_name, db_user)
            engine = create_engine(db_uri, echo=False)

            metadata = MetaData()
            Table(
                '3d_condours', metadata,
                Column('id', Integer, primary_key=True),
                Column('contour_bound', Geometry)
            )
            run_logger.info("Initialized SQLAlchemy metadata and registered custom Geometry table")
            db = SQLDatabase(
                engine=engine,
                metadata=metadata,
                include_tables=include_tables,
                sample_rows_in_table_info=3,
                lazy_table_reflection=True,
            )
            run_logger.info("Created SQLDatabase (include_tables=%s)", include_tables)
        except KeyError:
            yield {"type": "error", "content": (
                "Error: Database credentials secret not found. "
                "Ensure 'mysql-credentials' secret includes DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT."
            )}
            return

        extraction_tool = extraction_sandbox_agent(
            llm=llm,
            db=db,
            table_info=table_info,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access,
        )
        run_logger.info("Initialized extraction_tool")

        general_sql_query_tool = GeneralSQLQueryTool(
            db=db,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )

        write_artefact_tool = WriteArtefactTool(blob_db=blob_db, metadata_db=metadata_db)

        thread_id = str(uuid.uuid4())
        user_id = 1
        timeseries_plot_sandbox_tool = timeseries_plot_sandbox_agent(
            llm=llm,
            sql_tool=general_sql_query_tool,
            write_artefact_tool=write_artefact_tool,
            thread_id=thread_id,
            user_id=user_id
        )
        map_plot_sandbox_tool = map_plot_sandbox_agent(
            llm=llm,
            sql_tool=general_sql_query_tool,
            write_artefact_tool=write_artefact_tool,
            thread_id=thread_id,
            user_id=user_id
        )
        run_logger.info("Initialized plotting tools")

        # Initialize artefact management tools
        write_artefact_tool = WriteArtefactTool(blob_db=blob_db, metadata_db=metadata_db)
        read_artefacts_tool = ReadArtefactsTool(blob_db=blob_db, metadata_db=metadata_db)
        delete_artefacts_tool = DeleteArtefactsTool(blob_db=blob_db, metadata_db=metadata_db)
        run_logger.info("Initialized artefact management tools")

        if not code.strip().startswith("from __future__ import annotations"):
            code = "from __future__ import annotations\n" + code

        # Prepare the local namespace for exec()
        local_namespace = {
            "extraction_sandbox_agent": extraction_tool,
            "timeseries_plot_sandbox_agent": timeseries_plot_sandbox_tool,
            "map_plot_sandbox_agent": map_plot_sandbox_tool,
            "write_artefact_tool": write_artefact_tool,
            "read_artefacts_tool": read_artefacts_tool,
            "delete_artefacts_tool": delete_artefacts_tool,
            "llm": llm,
            "db": db,
            "datetime": __import__("datetime"),
        }
        run_logger.info("Executing user code via exec()")
        
        # Execute the user's code, which defines the `execute_strategy` function
        exec(code, local_namespace)
        
        # Validate the execute_strategy function
        if "execute_strategy" not in local_namespace:
            run_logger.error("User code did not define 'execute_strategy'")
            yield {"type": "error", "content": "Error: Code must define an 'execute_strategy' function."}
            return
        execute_strategy = local_namespace["execute_strategy"]
        if not callable(execute_strategy):
            run_logger.error("'execute_strategy' is not callable")
            yield {"type": "error", "content": "Error: 'execute_strategy' must be a callable function."}
            return

        # Execute the strategy and yield the outputs (zero-arg required)
        run_logger.info("Running execute_strategy() with zero arguments ...")
        try:
            iterator = execute_strategy()
        except TypeError as te:
            yield {"type": "error", "content": (
                "Error: execute_strategy must be defined with zero arguments. "
                f"Caught TypeError when calling execute_strategy(): {te}"
            )}
            return

        for output in iterator:
            yield output
        run_logger.info("execute_strategy completed")

    except Exception as e:
        # Catch any exception during setup or execution and yield it
        import traceback
        error_message = f"An error occurred in the sandbox: {e}\n{traceback.format_exc()}"
        run_logger.exception("Sandbox execution failed: %s", e)
        yield {"type": "error", "content": error_message}
    finally:
        # Clean up the temporary credentials file
        if 'temp_credentials_path' in locals() and os.path.exists(temp_credentials_path):
            os.remove(temp_credentials_path)
            run_logger.info("Removed temporary credentials file %s", temp_credentials_path)
        # Close connections
        if 'metadata_db' in locals() and metadata_db:
            metadata_db.close()
            run_logger.info("Closed metadata_db connection")


def run_with_live_logs(
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    user_id: int,
    global_hierarchy_access: bool,
) -> Generator[dict, None, None]:
    call = run_sandboxed_code.spawn(
        code=code,
        table_info=table_info,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )

    def poll_logs():
        last_log = ""
        while True:
            current_log = call.get_logs()
            if current_log is not None:
                new_log = current_log[len(last_log):]
                print(new_log, end="")
                last_log = current_log
            if call.finished():
                break
            time.sleep(0.1)
        # Final log check after finished
        current_log = call.get_logs()
        if current_log is not None:
            new_log = current_log[len(last_log):]
            print(new_log, end="")

    log_thread = threading.Thread(target=poll_logs)
    log_thread.start()

    for output in call:
        yield output

    log_thread.join()

def run_sandboxed_code_with_local_logs(
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    user_id: int,
    global_hierarchy_access: bool,
) -> Generator[dict, None, None]:
    """
    Executes the provided Python code string locally with real-time logging to the terminal.

    Args:
        code: The Python code string to execute. Must define an 'execute_strategy' function that yields dicts.
        table_info: List of dictionaries containing table metadata.
        table_relationship_graph: Dictionary representing table relationships.
        user_id: Integer representing the user ID.
        global_hierarchy_access: Boolean indicating global hierarchy access.

    Yields:
        Dictionaries representing the JSON outputs from the executed code, or error dicts if failed.
    """
    # Configure logger for local execution
    local_logger = logging.getLogger("modal_sandbox.local")
    if not local_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        local_logger.addHandler(handler)
        local_logger.setLevel(logging.INFO)

    local_logger.info("Starting local sandbox execution | user_id=%s global_access=%s", user_id, global_hierarchy_access)

    # Call the original run_sandboxed_code as a generator
    for output in run_sandboxed_code.local(
        code=code,
        table_info=table_info,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    ):
        yield output

    local_logger.info("Local sandbox execution completed")