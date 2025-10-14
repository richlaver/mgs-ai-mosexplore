import sys
import modal
import logging
import json
from typing import Generator, List, Dict
import os
import psycopg2
from psycopg2 import OperationalError
import b2sdk.v1 as b2
from langchain_google_vertexai import ChatVertexAI
from sqlalchemy import create_engine, MetaData, Table, Column, Integer
from geoalchemy2 import Geometry
from langchain_community.utilities import SQLDatabase
from parameters import include_tables

from setup import build_modal_secrets

app = modal.App("mgs-code-sandbox")
logger = logging.getLogger("modal_sandbox")
root = logging.getLogger()
if not root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
        "plotly",
        "pyproj",
        "b2sdk",
        "psycopg2-binary",
        "requests",
        "streamlit"
    )
    .add_local_file("classes.py", remote_path="/root/classes.py")
    .add_local_file("parameters.py", remote_path="/root/parameters.py")
    .add_local_file("agents/extraction_sandbox_agent.py", remote_path="/root/agents/extraction_sandbox_agent.py")
    .add_local_file("agents/timeseries_plot_sandbox_agent.py", remote_path="/root/agents/timeseries_plot_sandbox_agent.py")
    .add_local_file("agents/map_plot_sandbox_agent.py", remote_path="/root/agents/map_plot_sandbox_agent.py")
    .add_local_file("tools/artefact_toolkit.py", remote_path="/root/tools/artefact_toolkit.py")
    .add_local_file("tools/create_output_toolkit.py", remote_path="/root/tools/create_output_toolkit.py")
    .add_local_file("tools/sql_security_toolkit.py", remote_path="/root/tools/sql_security_toolkit.py")
    .add_local_file("artefact_management.py", remote_path="/root/artefact_management.py")
    .add_local_file("setup.py", remote_path="/root/setup.py")
)

@app.cls(
    image=image,
    secrets=build_modal_secrets(),
    min_containers=1,
    timeout=600
)
@modal.concurrent(max_inputs=10)
class SandboxExecutor:
    @modal.enter()
    def enter(self):
        logger.info("Running container enter hook")

        credentials_json = os.environ["GOOGLE_CREDENTIALS_JSON"]
        logger.info("Found GOOGLE_CREDENTIALS_JSON in environment (len=%s)", len(credentials_json))
        temp_credentials_path = "google_credentials.json"
        with open(temp_credentials_path, "w") as f:
            f.write(credentials_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path
        logger.info("Wrote temporary Google credentials to %s", temp_credentials_path)

        self.project_id = json.loads(credentials_json).get("project_id")
        if not self.project_id:
            raise ValueError("project_id not found in GOOGLE_CREDENTIALS_JSON")
        logger.info("Parsed project_id: %s", self.project_id)

        required_rds_env = ["ARTEFACT_METADATA_RDS_HOST", "ARTEFACT_METADATA_RDS_USER", "ARTEFACT_METADATA_RDS_PASS", "ARTEFACT_METADATA_RDS_DATABASE"]
        missing_rds_env = [k for k in required_rds_env if k not in os.environ or not os.environ.get(k)]
        rds_port = os.environ.get("ARTEFACT_METADATA_RDS_PORT", "5432")
        if missing_rds_env:
            logger.error("Missing RDS env vars: %s", missing_rds_env)
            raise KeyError("Missing required RDS environment variables: " + ", ".join(missing_rds_env))

        self.metadata_db = psycopg2.connect(
            host=os.environ["ARTEFACT_METADATA_RDS_HOST"],
            user=os.environ["ARTEFACT_METADATA_RDS_USER"],
            password=os.environ["ARTEFACT_METADATA_RDS_PASS"],
            database=os.environ["ARTEFACT_METADATA_RDS_DATABASE"],
            port=rds_port
        )
        logger.info("Connected to PostgreSQL RDS metadata_db")

        required_b2_env = ["ARTEFACT_BLOB_B2_KEY_ID", "ARTEFACT_BLOB_B2_KEY", "ARTEFACT_BLOB_B2_BUCKET"]
        missing_b2_env = [k for k in required_b2_env if k not in os.environ or not os.environ.get(k)]
        if missing_b2_env:
            logger.error("Missing B2 env vars: %s", missing_b2_env)
            raise KeyError("Missing required B2 environment variables: " + ", ".join(missing_b2_env))

        info = b2.InMemoryAccountInfo()
        b2_api = b2.B2Api(info)
        b2_api.authorize_account("production", os.environ["ARTEFACT_BLOB_B2_KEY_ID"], os.environ["ARTEFACT_BLOB_B2_KEY"])
        self.blob_db = b2_api.get_bucket_by_name(os.environ["ARTEFACT_BLOB_B2_BUCKET"])
        logger.info("Authorized and got B2 bucket: %s", os.environ["ARTEFACT_BLOB_B2_BUCKET"])

        self.llm = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0.1)
        logger.info("Initialized ChatVertexAI model=%s", "gemini-2.0-flash-001")

        required_env = ["DB_HOST", "DB_USER", "DB_PASS", "DB_NAME"]
        missing_env = [k for k in required_env if k not in os.environ or not os.environ.get(k)]
        db_port = os.environ.get("DB_PORT", "3306")
        if missing_env:
            logger.error("Missing DB env vars: %s | Present: %s", missing_env, [k for k in required_env if k in os.environ])
            raise KeyError("Missing required DB environment variables: " + ", ".join(missing_env))

        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USER"]
        db_pass = os.environ["DB_PASS"]
        db_name = os.environ["DB_NAME"]

        db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        logger.info("Creating SQLAlchemy engine for MySQL (host=%s port=%s db=%s user=%s)", db_host, db_port, db_name, db_user)
        engine = create_engine(db_uri, echo=False, pool_pre_ping=True)
        metadata = MetaData()
        Table(
            '3d_condours', metadata,
            Column('id', Integer, primary_key=True),
            Column('contour_bound', Geometry)
        )
        logger.info("Initialized SQLAlchemy metadata and registered custom Geometry table")
        self.db = SQLDatabase(
            engine=engine,
            metadata=metadata,
            include_tables=include_tables,
            sample_rows_in_table_info=3,
            lazy_table_reflection=True,
        )
        logger.info("Created SQLDatabase (include_tables=%s)", include_tables)

        sys.path.append("/root")

    @modal.exit()
    def exit(self):
        logger.info("Running container exit hook")
        if hasattr(self, 'metadata_db') and self.metadata_db:
            self.metadata_db.close()
            logger.info("Closed PostgreSQL connection")

    @modal.method()
    def run_sandboxed_code(
        self,
        code: str,
        table_info: List[Dict],
        table_relationship_graph: Dict[str, List[tuple]],
        thread_id: int,
        user_id: int,
        global_hierarchy_access: bool,
    ) -> Generator[dict, None, None]:
        """
        Executes the provided Python code string in a secure, remote Modal sandbox.
        """
        from tools.artefact_toolkit import WriteArtefactTool
        run_logger = logging.getLogger("modal_sandbox.run")
        run_logger.info("Starting sandbox execution | user_id=%s global_access=%s code_len=%s table_info=%s rel_graph_keys=%s",
                        user_id, global_hierarchy_access, len(code) if isinstance(code, str) else None,
                        len(table_info) if isinstance(table_info, list) else None,
                        list(table_relationship_graph.keys()) if isinstance(table_relationship_graph, dict) else None)
        from agents.extraction_sandbox_agent import extraction_sandbox_agent
        from agents.timeseries_plot_sandbox_agent import timeseries_plot_sandbox_agent
        from agents.map_plot_sandbox_agent import map_plot_sandbox_agent
        from tools.sql_security_toolkit import GeneralSQLQueryTool

        try:
            try:
                if self.metadata_db.closed != 0:
                    run_logger.info("PostgreSQL connection closed; reconnecting")
                    raise OperationalError("Connection closed")
                with self.metadata_db.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                run_logger.info("PostgreSQL connection is active")
            except (OperationalError, psycopg2.InterfaceError) as e:
                run_logger.warning("PostgreSQL connection failed: %s; reconnecting", e)
                self.metadata_db.close()
                self.metadata_db = psycopg2.connect(
                    host=os.environ["ARTEFACT_METADATA_RDS_HOST"],
                    user=os.environ["ARTEFACT_METADATA_RDS_USER"],
                    password=os.environ["ARTEFACT_METADATA_RDS_PASS"],
                    database=os.environ["ARTEFACT_METADATA_RDS_DATABASE"],
                    port=os.environ.get("ARTEFACT_METADATA_RDS_PORT", "5432")
                )
                run_logger.info("Reconnected to PostgreSQL RDS metadata_db")

            extraction_tool = extraction_sandbox_agent(
                llm=self.llm,
                db=self.db,
                table_info=table_info,
                table_relationship_graph=table_relationship_graph,
                user_id=user_id,
                global_hierarchy_access=global_hierarchy_access,
            )
            run_logger.info("Initialized extraction_tool")

            general_sql_query_tool = GeneralSQLQueryTool(
                db=self.db,
                table_relationship_graph=table_relationship_graph,
                user_id=user_id,
                global_hierarchy_access=global_hierarchy_access
            )

            write_artefact_tool = WriteArtefactTool(blob_db=self.blob_db, metadata_db=self.metadata_db)

            timeseries_plot_sandbox_tool = timeseries_plot_sandbox_agent(
                llm=self.llm,
                sql_tool=general_sql_query_tool,
                write_artefact_tool=write_artefact_tool,
                thread_id=thread_id,
                user_id=user_id
            )
            map_plot_sandbox_tool = map_plot_sandbox_agent(
                llm=self.llm,
                sql_tool=general_sql_query_tool,
                write_artefact_tool=write_artefact_tool,
                thread_id=thread_id,
                user_id=user_id
            )
            run_logger.info("Initialized plotting tools")

            if not code.strip().startswith("from __future__ import annotations"):
                code = "from __future__ import annotations\n" + code

            local_namespace = {
                "extraction_sandbox_agent": extraction_tool,
                "timeseries_plot_sandbox_agent": timeseries_plot_sandbox_tool,
                "map_plot_sandbox_agent": map_plot_sandbox_tool,
                "llm": self.llm,
                "db": self.db,
                "datetime": __import__("datetime"),
            }
            run_logger.info("Executing user code via exec()")
            
            exec(code, local_namespace)
            
            if "execute_strategy" not in local_namespace:
                run_logger.error("User code did not define 'execute_strategy'")
                yield {"type": "error", "content": "Error: Code must define an 'execute_strategy' function."}
                return
            execute_strategy = local_namespace["execute_strategy"]
            if not callable(execute_strategy):
                run_logger.error("'execute_strategy' is not callable")
                yield {"type": "error", "content": "Error: 'execute_strategy' must be a callable function."}
                return

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
            import traceback
            error_message = f"An error occurred in the sandbox: {e}\n{traceback.format_exc()}"
            run_logger.exception("Sandbox execution failed: %s", e)
            yield {"type": "error", "content": error_message}

def run_with_live_logs(
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: int,
    user_id: int,
    global_hierarchy_access: bool,
) -> Generator[dict, None, None]:
    with modal.enable_output():
        # executor = SandboxExecutor()
        executor_class = modal.Cls.from_name("mgs-code-sandbox", "SandboxExecutor")
        executor = executor_class()
        outputs = executor.run_sandboxed_code.remote_gen(
            code=code,
            table_info=table_info,
            table_relationship_graph=table_relationship_graph,
            thread_id=thread_id,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access,
        )
        for output in outputs:
            yield output

def run_sandboxed_code_with_local_logs(
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: int,
    user_id: int,
    global_hierarchy_access: bool,
) -> Generator[dict, None, None]:
    local_logger = logging.getLogger("modal_sandbox.local")
    if not local_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        local_logger.addHandler(handler)
        local_logger.setLevel(logging.INFO)

    local_logger.info("Starting local sandbox execution | user_id=%s global_access=%s", user_id, global_hierarchy_access)

    executor = SandboxExecutor()
    for output in executor.run_sandboxed_code.local_gen(
        code=code,
        table_info=table_info,
        table_relationship_graph=table_relationship_graph,
        thread_id=thread_id,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    ):
        yield output

    local_logger.info("Local sandbox execution completed")