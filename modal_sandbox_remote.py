import sys
import asyncio
import inspect
import modal
import logging
import json
import os
import psycopg2
from psycopg2 import OperationalError
import b2sdk.v1 as b2
from langchain_google_vertexai import ChatVertexAI
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, event
from geoalchemy2 import Geometry
from langchain_community.utilities import SQLDatabase
from typing import Generator, List, Dict, AsyncGenerator, Optional
from parameters import include_tables
from setup import build_modal_secrets

app = modal.App("mgs-code-sandbox")
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.13")
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
        "sqlglot",
        "streamlit",
        "tabulate",
    )
    .add_local_file("classes.py", remote_path="/root/classes.py")
    .add_local_file("parameters.py", remote_path="/root/parameters.py")
    .add_local_file("agents/extraction_sandbox_agent.py", remote_path="/root/agents/extraction_sandbox_agent.py")
    .add_local_file("agents/timeseries_plot_sandbox_agent.py", remote_path="/root/agents/timeseries_plot_sandbox_agent.py")
    .add_local_file("agents/map_plot_sandbox_agent.py", remote_path="/root/agents/map_plot_sandbox_agent.py")
    .add_local_file("agents/review_level_agents.py", remote_path="/root/agents/review_level_agents.py")
    .add_local_file("tools/artefact_toolkit.py", remote_path="/root/tools/artefact_toolkit.py")
    .add_local_file("tools/create_output_toolkit.py", remote_path="/root/tools/create_output_toolkit.py")
    .add_local_file("tools/review_level_toolkit.py", remote_path="/root/tools/review_level_toolkit.py")
    .add_local_file("tools/sql_security_toolkit.py", remote_path="/root/tools/sql_security_toolkit.py")
    .add_local_file("artefact_management.py", remote_path="/root/artefact_management.py")
    .add_local_file("setup.py", remote_path="/root/setup.py")
    .add_local_file("utils/project_selection.py", remote_path="/root/utils/project_selection.py")
)


def _get_int_env(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Environment variable %s=%s is not an int", name, raw)
        return default

def _enter_impl(self):
    logger.info("Running container enter hook")

    # Set up Google credentials
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

    # Set up metadata_db (PostgreSQL RDS)
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

    # Set up blob_db (Backblaze B2)
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

    # Initialize LLMs
    self.llms = {
        "FAST": ChatVertexAI(model="gemini-2.5-flash-lite", temperature=0.1),
        "BALANCED": ChatVertexAI(model="gemini-2.5-flash", temperature=0.1),
        "THINKING": ChatVertexAI(model="gemini-2.5-pro", temperature=0.1),
    }
    logger.info("Initialized ChatVertexAI models: FAST, BALANCED, THINKING")

    self.project_configs = {}
    pdj = os.environ.get("PROJECT_DATA_JSON")
    if pdj:
        try:
            parsed = json.loads(pdj)
            if isinstance(parsed, dict):
                for key, cfg in parsed.items():
                    if isinstance(cfg, dict):
                        self.project_configs[f"project_data.{key}"] = cfg
            logger.info("Loaded %d project configs from PROJECT_DATA_JSON", len(self.project_configs))
        except Exception as e:
            logger.warning("Failed to parse PROJECT_DATA_JSON: %s", e)
    else:
        logger.info("PROJECT_DATA_JSON not present; will rely on default DB_* env")

    self._db_cache = {}

    sys.path.append("/root")


def _attach_session_configuration(engine, tmp_size: Optional[int], heap_size: Optional[int]):
    if not tmp_size and not heap_size:
        return

    @event.listens_for(engine, "connect")
    def _set_session_limits(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            if tmp_size:
                cursor.execute("SET SESSION tmp_table_size = %s", (tmp_size,))
            if heap_size:
                cursor.execute("SET SESSION max_heap_table_size = %s", (heap_size,))
        finally:
            cursor.close()


def _exit_impl(self):
    logger.info("Running container exit hook")
    if hasattr(self, 'metadata_db') and self.metadata_db:
        self.metadata_db.close()
        logger.info("Closed PostgreSQL connection")


async def _run_impl(self,
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    selected_project_key: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    from tools.artefact_toolkit import WriteArtefactTool
    run_logger = logging.getLogger("modal_sandbox.run")
    run_logger.info("Starting sandbox execution | user_id=%s global_access=%s code_len=%s table_info=%s rel_graph_keys=%s",
                    user_id, global_hierarchy_access, len(code) if isinstance(code, str) else None,
                    len(table_info) if isinstance(table_info, list) else None,
                    list(table_relationship_graph.keys()) if isinstance(table_relationship_graph, dict) else None)
    from agents.extraction_sandbox_agent import extraction_sandbox_agent
    from agents.timeseries_plot_sandbox_agent import timeseries_plot_sandbox_agent
    from agents.map_plot_sandbox_agent import map_plot_sandbox_agent
    from agents.review_level_agents import (
        review_by_value_agent,
        review_by_time_agent,
        review_schema_agent,
        breach_instr_agent
    )
    from tools.create_output_toolkit import CSVSaverTool
    from tools.sql_security_toolkit import GeneralSQLQueryTool

    session_tmp_table_size = _get_int_env("SANDBOX_SESSION_TMP_TABLE_SIZE")
    session_max_heap_table_size = _get_int_env("SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE")
    try:
        # Verify PostgreSQL connection
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

        def _get_db_for_project(project_key: Optional[str]) -> SQLDatabase:
            if project_key and project_key in self._db_cache:
                return self._db_cache[project_key]
            cfg = None
            if project_key and project_key in self.project_configs:
                cfg = self.project_configs.get(project_key)
            if cfg and all(k in cfg for k in ("db_host", "db_user", "db_pass", "db_name")):
                host = str(cfg.get("db_host"))
                user = str(cfg.get("db_user"))
                pwd = str(cfg.get("db_pass"))
                name = str(cfg.get("db_name"))
                port = str(cfg.get("port", "3306"))
                uri = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"
                run_logger.info("Creating project SQLAlchemy engine (project=%s host=%s db=%s port=%s)", project_key, host, name, port)
                engine = create_engine(uri, echo=False, pool_pre_ping=True)
                metadata = MetaData()
                Table('3d_condours', metadata, Column('id', Integer, primary_key=True), Column('contour_bound', Geometry))
                _attach_session_configuration(engine, session_tmp_table_size, session_max_heap_table_size)
                db = SQLDatabase(
                    engine=engine,
                    metadata=metadata,
                    include_tables=include_tables,
                    sample_rows_in_table_info=3,
                    lazy_table_reflection=True,
                )
                self._db_cache[project_key] = db
                return db
            raise RuntimeError("PROJECT_DATA_JSON missing or project config not found/invalid for key: %s" % project_key)

        db = _get_db_for_project(selected_project_key)
        # Initialize tools
        extraction_tool = extraction_sandbox_agent(
            llm=self.llms['BALANCED'],
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

        write_artefact_tool = WriteArtefactTool(blob_db=self.blob_db, metadata_db=self.metadata_db)

        timeseries_plot_sandbox_tool = timeseries_plot_sandbox_agent(
            llm=self.llms['BALANCED'],
            sql_tool=general_sql_query_tool,
            write_artefact_tool=write_artefact_tool,
            thread_id=thread_id,
            user_id=user_id
        )
        map_plot_sandbox_tool = map_plot_sandbox_agent(
            llm=self.llms['BALANCED'],
            sql_tool=general_sql_query_tool,
            write_artefact_tool=write_artefact_tool,
            thread_id=thread_id,
            user_id=user_id
        )
        review_by_value_tool = review_by_value_agent(
            llm=self.llms['BALANCED'],
            db=db,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        review_by_time_tool = review_by_time_agent(
            llm=self.llms['BALANCED'],
            db=db,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        review_schema_tool = review_schema_agent(
            llm=self.llms['BALANCED'],
            db=db,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        breach_instr_tool = breach_instr_agent(
            llm=self.llms['BALANCED'],
            db=db,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        csv_saver_tool = CSVSaverTool(
            write_artefact_tool=write_artefact_tool,
            thread_id=thread_id,
            user_id=user_id,
        )
        run_logger.info("Initialized tools")

        tool_limit = _get_int_env("SANDBOX_MAX_CONCURRENT_TOOL_CALLS", 8) or 8
        tool_limit = max(1, tool_limit)
        tool_semaphore = asyncio.Semaphore(tool_limit)
        run_logger.info("Tool call concurrency limited to %s", tool_limit)

        async def ainvoke(tool, prompt: str, timeout: float | None = None):
            try:
                async with tool_semaphore:
                    meth = getattr(tool, "ainvoke", None)
                    if callable(meth):
                        if inspect.iscoroutinefunction(meth):
                            coro = meth(prompt)
                            return await (asyncio.wait_for(coro, timeout) if timeout else coro)
                        fut = asyncio.to_thread(meth, prompt)
                        res = await (asyncio.wait_for(fut, timeout) if timeout else fut)
                        if inspect.isawaitable(res):
                            return await (asyncio.wait_for(res, timeout) if timeout else res)
                        return res

                    call = getattr(tool, "invoke", None)
                    if call is None or not callable(call):
                        raise AttributeError("Tool has neither ainvoke nor invoke")
                    fut = asyncio.to_thread(call, prompt)
                    return await (asyncio.wait_for(fut, timeout) if timeout else fut)
            except Exception as e:
                run_logger.exception("ainvoke failed for tool %s: %s", getattr(tool, 'name', type(tool).__name__), e)
                raise

        # Prepare and execute code
        if not code.strip().startswith("from __future__ import annotations"):
            code = "from __future__ import annotations\n" + code

        local_namespace = {
            "extraction_sandbox_agent": extraction_tool,
            "timeseries_plot_sandbox_agent": timeseries_plot_sandbox_tool,
            "map_plot_sandbox_agent": map_plot_sandbox_tool,
            "review_by_value_agent": review_by_value_tool,
            "review_by_time_agent": review_by_time_tool,
            "review_schema_agent": review_schema_tool,
            "breach_instr_agent": breach_instr_tool,
            "csv_saver_tool": csv_saver_tool,
            "llm": self.llms['BALANCED'],
            "db": db,
            # Avoid importing datetime because code generation agent gets confused between the datetime module and the datetime class
            # "datetime": __import__("datetime"),
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "ainvoke": ainvoke,
            "asyncio": asyncio,
            "PARALLEL_ENABLED": True,
        }
        run_logger.info("[%s] Executing user code via exec()", type(self).__name__)

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

        run_logger.info("[%s] Running execute_strategy() with zero arguments ...", type(self).__name__)

        async def _yield_outputs(obj):
            """Yield outputs from any kind of result (async gen, coroutine, sync iterable, dict)."""
            try:
                if obj is None:
                    return
                # Async generator
                if inspect.isasyncgen(obj):
                    async for item in obj:
                        yield item
                    return
                # Coroutine -> await and recurse
                if inspect.iscoroutine(obj):
                    awaited = await obj
                    async for item in _yield_outputs(awaited):
                        yield item
                    return
                # Single dict payload
                if isinstance(obj, dict):
                    yield obj
                    return
                # Synchronous iterable (avoid iterating strings/bytes)
                if isinstance(obj, (str, bytes, bytearray)):
                    return
                try:
                    iterator = iter(obj)
                except TypeError:
                    return
                else:
                    for item in iterator:
                        yield item
                    return
            except Exception as e:
                run_logger.exception("Error while yielding outputs: %s", e)
                raise

        try:
            result = execute_strategy()
        except TypeError as te:
            yield {"type": "error", "content": (
                "Error: execute_strategy must be defined with zero arguments. "
                f"Caught TypeError when calling execute_strategy(): {te}"
            )}
            return

        async for output in _yield_outputs(result):
            yield output
        run_logger.info("[%s] execute_strategy completed", type(self).__name__)

    except Exception as e:
        import traceback
        error_message = f"An error occurred in the sandbox: {e}\n{traceback.format_exc()}"
        run_logger.exception("Sandbox execution failed: %s", e)
        yield {"type": "error", "content": error_message}
@app.cls(
    image=image,
    secrets=build_modal_secrets(),
    min_containers=1,
    timeout=600
)
@modal.concurrent(max_inputs=1)
class SandboxExecutorA:
    @modal.enter()
    def enter(self):
        _enter_impl(self)

    @modal.exit()
    def exit(self):
        _exit_impl(self)

    @modal.method()
    async def run_sandboxed_code(
        self,
        code: str,
        table_info: List[Dict],
        table_relationship_graph: Dict[str, List[tuple]],
        thread_id: str,
        user_id: int,
        global_hierarchy_access: bool,
        selected_project_key: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        async for out in _run_impl(
            self,
            code,
            table_info,
            table_relationship_graph,
            thread_id,
            user_id,
            global_hierarchy_access,
            selected_project_key,
        ):
            yield out


@app.cls(
    image=image,
    secrets=build_modal_secrets(),
    min_containers=0,
    timeout=600
)
@modal.concurrent(max_inputs=1)
class SandboxExecutorB:
    @modal.enter()
    def enter(self):
        _enter_impl(self)

    @modal.exit()
    def exit(self):
        _exit_impl(self)

    @modal.method()
    async def run_sandboxed_code(
        self,
        code: str,
        table_info: List[Dict],
        table_relationship_graph: Dict[str, List[tuple]],
        thread_id: str,
        user_id: int,
        global_hierarchy_access: bool,
        selected_project_key: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:  # type: ignore[override]
        async for out in _run_impl(
            self,
            code,
            table_info,
            table_relationship_graph,
            thread_id,
            user_id,
            global_hierarchy_access,
            selected_project_key,
        ):
            yield out


@app.cls(
    image=image,
    secrets=build_modal_secrets(),
    # Keep C container optionally warm via warm-up (do not force always-on)
    min_containers=0,
    timeout=600
)
@modal.concurrent(max_inputs=1)
class SandboxExecutorC:
    @modal.enter()
    def enter(self):
        _enter_impl(self)

    @modal.exit()
    def exit(self):
        _exit_impl(self)

    @modal.method()
    async def run_sandboxed_code(
        self,
        code: str,
        table_info: List[Dict],
        table_relationship_graph: Dict[str, List[tuple]],
        thread_id: str,
        user_id: int,
        global_hierarchy_access: bool,
        selected_project_key: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:  # type: ignore[override]
        async for out in _run_impl(
            self,
            code,
            table_info,
            table_relationship_graph,
            thread_id,
            user_id,
            global_hierarchy_access,
            selected_project_key,
        ):
            yield out

def execute_remote_sandbox(
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    selected_project_key: Optional[str] = None,
    container_slot: Optional[int] = None,
) -> Generator[dict, None, None]:
    with modal.enable_output():
        slot = 0 if container_slot is None else int(container_slot)
        slot_mod = slot % 3
        class_name = "SandboxExecutorA" if slot_mod == 0 else ("SandboxExecutorB" if slot_mod == 1 else "SandboxExecutorC")
        executor_class = modal.Cls.from_name("mgs-code-sandbox", class_name)
        executor = executor_class()
        outputs = executor.run_sandboxed_code.remote_gen(
            code=code,
            table_info=table_info,
            table_relationship_graph=table_relationship_graph,
            thread_id=thread_id,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access,
            selected_project_key=selected_project_key,
        )
        for output in outputs:
            yield output