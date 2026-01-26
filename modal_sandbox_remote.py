import sys
import asyncio
import inspect
import contextlib
import io
import modal
import logging
import json
import os
import queue
import threading
import psycopg2
from psycopg2 import OperationalError
import b2sdk.v1 as b2
from langchain_google_vertexai import ChatVertexAI
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, event
from geoalchemy2 import Geometry
from langchain_community.utilities import SQLDatabase
from datetime import datetime, timezone
from time import perf_counter
from typing import Generator, List, Dict, AsyncGenerator, Optional, Any
from parameters import include_tables
from setup import build_modal_secrets
from contextlib import suppress
from modal._functions import _FunctionCall

try:  # The utils module may not exist inside the remote container image
    from utils.run_cancellation import get_active_run_controller
except ModuleNotFoundError:  # pragma: no cover - remote fallback
    def get_active_run_controller():
        return None

# Ten times the default MySQL limits
MYSQL_DEFAULT_SESSION_TABLE_LIMIT_BYTES = 16 * 1024 * 1024
SANDBOX_SESSION_TABLE_LIMIT_BYTES = MYSQL_DEFAULT_SESSION_TABLE_LIMIT_BYTES * 10
SANDBOX_EXECUTOR_CLASS_NAMES = [
    "SandboxExecutorA",
    "SandboxExecutorB",
    "SandboxExecutorC",
    "SandboxExecutorD",
    "SandboxExecutorE",
    "SandboxExecutorF",
    "SandboxExecutorG",
]

# Temporary debug flag: surface INFO logs to client to trace stalls. Set back to False after diagnosing.
STREAM_INFO_LOGS = True


def _ensure_basic_logging():
    """Attach a stdout handler so every logger (including nested tools) surfaces output."""
    root = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stdout_handler = None
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler.stream, "name", None) == "stdout":
            stdout_handler = handler
            break

    if stdout_handler is None:
        stdout_handler = logging.StreamHandler(sys.stdout)
        root.addHandler(stdout_handler)

    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.INFO)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)

    for namespace in (
        "modal_sandbox",
        "tools",
        "agents",
        "tools.sql_security_toolkit",
    ):
        logging.getLogger(namespace).setLevel(logging.INFO)


app = modal.App("mgs-code-sandbox")
logger = logging.getLogger(__name__)

def _normalize_api_endpoint(raw: str) -> str:
    if not raw:
        return ""
    return raw.replace("https://", "").replace("http://", "").rstrip("/")


VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
VERTEX_ENDPOINT = _normalize_api_endpoint(
    os.environ.get(
        "VERTEX_ENDPOINT",
        "aiplatform.googleapis.com"
        if VERTEX_LOCATION == "global"
        else f"{VERTEX_LOCATION}-aiplatform.googleapis.com",
    )
)

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
    .add_local_file("utils/run_cancellation.py", remote_path="/root/utils/run_cancellation.py")
    .add_local_file("utils/cancelable_llm.py", remote_path="/root/utils/cancelable_llm.py")
    .add_local_file("utils/cancelable_sql.py", remote_path="/root/utils/cancelable_sql.py")
)


@app.function(
    image=image,
    secrets=build_modal_secrets(),
    timeout=60,
)
def sandbox_timezone_probe() -> dict:
    now = datetime.now().astimezone()
    raw_offset = now.strftime("%z") or "+0000"
    normalized = f"{raw_offset[:3]}:{raw_offset[3:]}" if len(raw_offset) == 5 else raw_offset
    return {
        "iso": now.isoformat(),
        "tzname": now.tzname(),
        "offset": normalized,
    }


def _get_int_env(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Environment variable %s=%s is not an int", name, raw)
        return default


def _ensure_sandbox_session_limits():
    """Ensure sandbox session tmp/max heap table limits default to 10x MySQL defaults."""
    default_value = str(SANDBOX_SESSION_TABLE_LIMIT_BYTES)
    limit_mib = SANDBOX_SESSION_TABLE_LIMIT_BYTES // (1024 * 1024)
    if "SANDBOX_SESSION_TMP_TABLE_SIZE" not in os.environ:
        os.environ["SANDBOX_SESSION_TMP_TABLE_SIZE"] = default_value
        logger.info(
            "Defaulting SANDBOX_SESSION_TMP_TABLE_SIZE to %s bytes (~%s MiB)",
            default_value,
            limit_mib,
        )
    if "SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE" not in os.environ:
        os.environ["SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE"] = default_value
        logger.info(
            "Defaulting SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE to %s bytes (~%s MiB)",
            default_value,
            limit_mib,
        )


class _SandboxStream(io.TextIOBase):
    """Tee a text stream into an asyncio queue while preserving default behavior."""

    def __init__(self, original: io.TextIOBase, queue: "asyncio.Queue[Optional[str]]", filter_fn=None):
        self._original = original
        self._queue = queue
        self._buffer: str = ""
        self._filter = filter_fn

    def writable(self) -> bool:  # pragma: no cover - interface hook
        return True

    def write(self, data: str) -> int:  # pragma: no cover - exercised indirectly
        if not data:
            return 0
        self._original.write(data)
        normalized = data.replace("\r\n", "\n").replace("\r", "\n")
        self._buffer += normalized
        self._flush_lines()
        return len(data)

    def flush(self) -> None:  # pragma: no cover - interface hook
        self._original.flush()

    def flush_buffer(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""

    def _flush_lines(self) -> None:
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)

    def _emit_line(self, line: str) -> None:
        trimmed = line.rstrip()
        if not trimmed:
            return
        if self._filter and not self._filter(trimmed):
            return
        try:
            self._queue.put_nowait(trimmed)
        except asyncio.QueueFull:  # pragma: no cover - unbounded queue by default
            pass


def _stdout_error_filter(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if STREAM_INFO_LOGS:
        return True
    upper = stripped.upper()
    if any(marker in upper for marker in ("[WARNING]", "[ERROR]", "[CRITICAL]", "[EXCEPTION]")):
        return True
    lower = stripped.lower()
    if "traceback" in lower or "exception" in lower:
        return True
    return "error" in lower


class _QueueLoggingHandler(logging.Handler):
    """Bridge logging records into the sandbox stdout/stderr queues."""

    def __init__(self, stdout_queue: "asyncio.Queue[Optional[str]]", stderr_queue: "asyncio.Queue[Optional[str]]"):
        super().__init__(level=logging.INFO)
        self._stdout_queue = stdout_queue
        self._stderr_queue = stderr_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin adapter
        try:
            msg = self.format(record)
        except Exception:  # pragma: no cover - fallback to default logging error handling
            self.handleError(record)
            return

        target_queue = self._stderr_queue if record.levelno >= logging.WARNING else self._stdout_queue
        if not msg or target_queue is None:
            return
        if target_queue is self._stdout_queue and not _stdout_error_filter(msg):
            return
        try:
            target_queue.put_nowait(msg)
        except asyncio.QueueFull:  # pragma: no cover - queues are unbounded in practice
            pass


def _make_step_payload(content: str, typ: str, step: int = 0, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Dict | str]:
    safe_step = step if isinstance(step, int) and step >= 0 else 0
    payload = {
        "content": (content or ""),
        "metadata": {
            "type": typ,
            "step": safe_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    if extra_metadata:
        payload["metadata"].update(extra_metadata)
    return payload


def _make_control_payload(event: str, **data: Any) -> Dict[str, Any]:
    control_payload = {"event": event}
    for key, value in data.items():
        if value is not None:
            control_payload[key] = value
    return {"__control__": control_payload}


def _format_error_payload(
    message: Optional[str] = None,
    step: int = 0,
    origin: str = "sandbox",
    *,
    line: Optional[str] = None,
) -> Dict[str, Dict | str]:
    safe_step = step if isinstance(step, int) and step >= 0 else 0
    chosen_text = message if message not in (None, "") else line
    safe_message = (chosen_text or "").strip()
    safe_origin = (origin or "sandbox").strip() or "sandbox"
    typ = "error" if safe_origin == "stderr" else "progress"
    prefix = f"[{safe_origin.upper()}][Step {safe_step}]"
    content = f"{prefix} {safe_message}" if safe_message else prefix
    return _make_step_payload(content=content, typ=typ, step=safe_step, extra_metadata={"origin": safe_origin})


def _drain_stream_lines(queue: "asyncio.Queue[Optional[str]]") -> List[str]:
    lines: List[str] = []
    while True:
        try:
            line = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if line:
            lines.append(line)
    return lines

def _enter_impl(self):
    _ensure_basic_logging()
    _ensure_sandbox_session_limits()
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
        port=rds_port,
        connect_timeout=10,
        options="-c statement_timeout=50000",
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
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
    llm_common = {
        "location": VERTEX_LOCATION,
        "api_endpoint": VERTEX_ENDPOINT,
    }
    self.llms = {
        "FAST": ChatVertexAI(model="gemini-2.5-flash-lite", temperature=0.1, **llm_common),
        "BALANCED": ChatVertexAI(model="gemini-2.5-flash", temperature=0.1, **llm_common),
        "THINKING": ChatVertexAI(model="gemini-2.5-pro", temperature=0.1, **llm_common),
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
                        from utils.project_selection import make_project_key
                        derived_key = make_project_key(str(cfg.get("db_host", "")), str(cfg.get("db_name", "")))
                        resolved_key = derived_key or str(key)
                        self.project_configs[f"project_data.{resolved_key}"] = cfg
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
    _ensure_basic_logging()
    start_ts = perf_counter()

    def log_stage(label: str) -> None:
        elapsed = perf_counter() - start_ts
        run_logger.info("[stage] %s | elapsed=%.3fs", label, elapsed)

    from tools.artefact_toolkit import WriteArtefactTool
    run_logger = logging.getLogger("modal_sandbox.run")
    run_logger.info("Starting sandbox execution | user_id=%s global_access=%s code_len=%s table_info=%s rel_graph_keys=%s",
                    user_id, global_hierarchy_access, len(code) if isinstance(code, str) else None,
                    len(table_info) if isinstance(table_info, list) else None,
                    list(table_relationship_graph.keys()) if isinstance(table_relationship_graph, dict) else None)
    log_stage("after_start_log")

    function_call_id = getattr(modal, "current_function_call_id", lambda: None)()
    input_id = getattr(modal, "current_input_id", lambda: None)()
    if function_call_id:
        run_logger.info(
            "[Sandbox] current_function_call_id=%s current_input_id=%s",
            function_call_id,
            input_id,
        )
        yield _make_control_payload(
            event="modal_call_metadata",
            function_call_id=function_call_id,
            input_id=input_id,
        )
    from agents.extraction_sandbox_agent import extraction_sandbox_agent
    from agents.timeseries_plot_sandbox_agent import timeseries_plot_sandbox_agent
    from agents.map_plot_sandbox_agent import map_plot_sandbox_agent
    from agents.review_level_agents import (
        review_by_value_agent,
        review_by_time_agent,
        review_schema_agent,
        breach_instr_agent,
        review_changes_across_period_agent,
    )
    from tools.create_output_toolkit import CSVSaverTool
    from tools.sql_security_toolkit import GeneralSQLQueryTool

    session_tmp_table_size = _get_int_env("SANDBOX_SESSION_TMP_TABLE_SIZE", SANDBOX_SESSION_TABLE_LIMIT_BYTES)
    session_max_heap_table_size = _get_int_env("SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE", SANDBOX_SESSION_TABLE_LIMIT_BYTES)
    try:
        log_stage("before_metadata_ping")
        async def _ping_metadata_db() -> None:
            def _do_ping() -> None:
                with self.metadata_db.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()

            return await asyncio.wait_for(asyncio.to_thread(_do_ping), timeout=10)

        # Verify PostgreSQL connection
        try:
            if self.metadata_db.closed != 0:
                run_logger.info("PostgreSQL connection closed; reconnecting")
                raise OperationalError("Connection closed")
            await _ping_metadata_db()
            run_logger.info("PostgreSQL connection is active")
        except (OperationalError, psycopg2.InterfaceError, asyncio.TimeoutError) as e:
            run_logger.warning("PostgreSQL connection failed or timed out: %s; reconnecting", e)
            self.metadata_db.close()
            self.metadata_db = psycopg2.connect(
                host=os.environ["ARTEFACT_METADATA_RDS_HOST"],
                user=os.environ["ARTEFACT_METADATA_RDS_USER"],
                password=os.environ["ARTEFACT_METADATA_RDS_PASS"],
                database=os.environ["ARTEFACT_METADATA_RDS_DATABASE"],
                port=os.environ.get("ARTEFACT_METADATA_RDS_PORT", "5432"),
                connect_timeout=10,
                options="-c statement_timeout=50000",
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=3,
            )
            run_logger.info("Reconnected to PostgreSQL RDS metadata_db")
            log_stage("after_metadata_reconnect")
            await _ping_metadata_db()
            run_logger.info("PostgreSQL reconnection validated")
        log_stage("after_metadata_ping")

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
                connect_args = {"connection_timeout": 10}
                run_logger.info(
                    "Creating project SQLAlchemy engine (project=%s host=%s db=%s port=%s connect_timeout=%ss pool_timeout=%ss pool_recycle=%ss)",
                    project_key,
                    host,
                    name,
                    port,
                    connect_args["connection_timeout"],
                    30,
                    1800,
                )
                engine = create_engine(
                    uri,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    pool_timeout=30,
                    connect_args=connect_args,
                )
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
        log_stage("after_db_lookup")
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
        review_changes_across_period_tool = review_changes_across_period_agent(
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
        log_stage("after_tool_init")

        tool_limit = _get_int_env("SANDBOX_MAX_CONCURRENT_TOOL_CALLS", 8) or 8
        tool_limit = max(1, tool_limit)
        tool_semaphore = asyncio.Semaphore(tool_limit)
        run_logger.info("Tool call concurrency limited to %s", tool_limit)

        stdout_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        stderr_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        stdout_stream = _SandboxStream(sys.stdout, stdout_queue, filter_fn=_stdout_error_filter)
        stderr_stream = _SandboxStream(sys.stderr, stderr_queue)
        stdout_ctx = contextlib.redirect_stdout(stdout_stream)
        stderr_ctx = contextlib.redirect_stderr(stderr_stream)

        queue_logging_handler = _QueueLoggingHandler(stdout_queue, stderr_queue)
        queue_logging_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(queue_logging_handler)

        def _collect_stream_payloads(flush: bool = False) -> List[Dict[str, Dict | str]]:
            if flush:
                stdout_stream.flush_buffer()
                stderr_stream.flush_buffer()
            payloads: List[Dict[str, Dict | str]] = []
            for origin, queue in (("stderr", stderr_queue), ("stdout", stdout_queue)):
                for line in _drain_stream_lines(queue):
                    payloads.append(_format_error_payload(message=line, step=0, origin=origin))
            return payloads

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

        try:
            with stdout_ctx, stderr_ctx:
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
                    "review_changes_across_period_agent": review_changes_across_period_tool,
                    "general_sql_query_tool": general_sql_query_tool,
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
                log_stage("before_exec")

                exec(code, local_namespace)

                if "execute_strategy" not in local_namespace:
                    run_logger.error("User code did not define 'execute_strategy'")
                    yield _make_step_payload(
                        content="Error: Code must define an 'execute_strategy' function.",
                        typ="error",
                    )
                    for payload in _collect_stream_payloads(flush=True):
                        yield payload
                    return
                execute_strategy = local_namespace["execute_strategy"]
                if not callable(execute_strategy):
                    run_logger.error("'execute_strategy' is not callable")
                    yield _make_step_payload(
                        content="Error: 'execute_strategy' must be a callable function.",
                        typ="error",
                    )
                    for payload in _collect_stream_payloads(flush=True):
                        yield payload
                    return

                run_logger.info("[%s] Running execute_strategy() with zero arguments ...", type(self).__name__)
                log_stage("before_execute_strategy")

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
                    yield _make_step_payload(
                        content=(
                            "Error: execute_strategy must be defined with zero arguments. "
                            f"Caught TypeError when calling execute_strategy(): {te}"
                        ),
                        typ="error",
                    )
                    for payload in _collect_stream_payloads(flush=True):
                        yield payload
                    return

                async for output in _yield_outputs(result):
                    for payload in _collect_stream_payloads():
                        yield payload
                    yield output
                for payload in _collect_stream_payloads(flush=True):
                    yield payload
                run_logger.info("[%s] execute_strategy completed", type(self).__name__)
                log_stage("after_execute_strategy")
        finally:
            root_logger.removeHandler(queue_logging_handler)
            queue_logging_handler.close()

    except Exception as e:
        import traceback
        error_message = f"An error occurred in the sandbox: {e}\n{traceback.format_exc()}"
        run_logger.exception("Sandbox execution failed: %s", e)
        yield _make_step_payload(content=error_message, typ="error")
        for payload in _collect_stream_payloads(flush=True):
            yield payload
@app.cls(
    image=image,
    secrets=build_modal_secrets(),
    min_containers=1,
    timeout=500
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
    timeout=500
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
    timeout=500
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


@app.cls(
    image=image,
    secrets=build_modal_secrets(),
    min_containers=0,
    timeout=500
)
@modal.concurrent(max_inputs=1)
class SandboxExecutorD:
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
    min_containers=0,
    timeout=500
)
@modal.concurrent(max_inputs=1)
class SandboxExecutorE:
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
    min_containers=0,
    timeout=500
)
@modal.concurrent(max_inputs=1)
class SandboxExecutorF:
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
    min_containers=0,
    timeout=500
)
@modal.concurrent(max_inputs=1)
class SandboxExecutorG:
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

    first_payload_timeout = _get_int_env("SANDBOX_FIRST_PAYLOAD_TIMEOUT_SECONDS", 20) or 20

    def _extract_modal_call(gen, slot_label: int, class_label: str) -> Optional[_FunctionCall]:
        to_visit: list[Any] = [gen]
        seen: set[int] = set()

        def _maybe_queue(value: Any) -> None:
            if value is None:
                return
            if not (inspect.isgenerator(value) or inspect.isasyncgen(value)):
                return
            ident = id(value)
            if ident in seen:
                return
            to_visit.append(value)

        while to_visit:
            current = to_visit.pop()
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)

            frame = (
                getattr(current, "ag_frame", None)
                or getattr(current, "gi_frame", None)
                or getattr(current, "cr_frame", None)
            )
            locals_map = getattr(frame, "f_locals", {}) if frame else {}
            invocation = locals_map.get("invocation")
            if invocation is not None:
                function_call_id = getattr(invocation, "function_call_id", None)
                client = getattr(invocation, "client", None)
                if function_call_id and client:
                    try:
                        fc = _FunctionCall._new_hydrated(function_call_id, client, None)
                        fc._is_generator = True
                        logger.info(
                            "[RemoteSandbox] Hydrated FunctionCall for cancellation | slot=%s class=%s call_id=%s",
                            slot_label,
                            class_label,
                            function_call_id,
                        )
                        return fc
                    except Exception as exc:
                        logger.warning(
                            "[RemoteSandbox] Failed to hydrate FunctionCall for cancellation | slot=%s class=%s error=%s",
                            slot_label,
                            class_label,
                            exc,
                        )
                        return None

            for value in locals_map.values():
                _maybe_queue(value)

            for attr in ("gen", "_gen", "_agen", "_async_gen"):
                _maybe_queue(getattr(current, attr, None))

            inner = getattr(current, "gi_yieldfrom", None)
            if inner is not None:
                _maybe_queue(inner)

        return None

    def _run_coro_blocking(coro):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    with modal.enable_output():
        slot = 0 if container_slot is None else int(container_slot)
        slot_mod = slot % len(SANDBOX_EXECUTOR_CLASS_NAMES)
        class_name = SANDBOX_EXECUTOR_CLASS_NAMES[slot_mod]
        logger.info(
            "[RemoteSandbox] Starting execute_remote_sandbox | slot=%s class=%s thread_id=%s user_id=%s",
            slot,
            class_name,
            thread_id,
            user_id,
        )
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
        modal_function_call = _extract_modal_call(outputs, slot, class_name)
        controller = get_active_run_controller()
        stream_handle = None
        modal_handle = None
        pending_modal_cancel = False

        def _hydrate_modal_call_by_id(function_call_id: Optional[str], origin: str) -> Optional[_FunctionCall]:
            if not function_call_id:
                logger.warning(
                    "[RemoteSandbox] Control payload missing function_call_id | slot=%s class=%s origin=%s",
                    slot,
                    class_name,
                    origin,
                )
                return None
            try:
                fc = modal.FunctionCall.from_id(function_call_id)
            except Exception as exc:
                logger.warning(
                    "[RemoteSandbox] Failed to hydrate Modal FunctionCall from id | slot=%s class=%s origin=%s error=%s",
                    slot,
                    class_name,
                    origin,
                    exc,
                )
                return None
            if hasattr(fc, "_is_generator"):
                setattr(fc, "_is_generator", True)
            logger.info(
                "[RemoteSandbox] Hydrated FunctionCall via %s | slot=%s class=%s call_id=%s",
                origin,
                slot,
                class_name,
                getattr(fc, "object_id", "unknown"),
            )
            return fc  # type: ignore[return-value]

        def _cancel_modal_call() -> None:
            nonlocal pending_modal_cancel
            if modal_function_call is None:
                pending_modal_cancel = True
                logger.warning(
                    "[RemoteSandbox] Cancellation requested but FunctionCall handle not ready | slot=%s class=%s",
                    slot,
                    class_name,
                )
                return
            pending_modal_cancel = False
            logger.warning(
                "[RemoteSandbox] Cancellation requested, sending Modal cancel | slot=%s class=%s call_id=%s",
                slot,
                class_name,
                getattr(modal_function_call, "object_id", "unknown"),
            )
            try:
                result = modal_function_call.cancel(terminate_containers=True)
                if inspect.isawaitable(result):
                    _run_coro_blocking(result)
            except Exception:
                logger.exception(
                    "[RemoteSandbox] Failed to cancel Modal FunctionCall | slot=%s class=%s",
                    slot,
                    class_name,
                )

        def _ensure_modal_handle(candidate: Optional[_FunctionCall], origin: str) -> None:
            nonlocal modal_function_call
            if candidate is None:
                return
            modal_function_call = candidate
            if controller and controller.is_cancelled():
                logger.info(
                    "[RemoteSandbox] Run already cancelled; invoking Modal cancel immediately | slot=%s class=%s origin=%s",
                    slot,
                    class_name,
                    origin,
                )
                _cancel_modal_call()
            elif pending_modal_cancel:
                logger.info(
                    "[RemoteSandbox] Deferred Modal cancellation will now run | slot=%s class=%s origin=%s",
                    slot,
                    class_name,
                    origin,
                )
                _cancel_modal_call()

        _ensure_modal_handle(modal_function_call, "frame_introspection")

        if controller:
            logger.info(
                "[RemoteSandbox] Registering cancellation callback | run_id=%s user_id=%s",
                getattr(controller, "run_id", "?"),
                getattr(controller, "user_id", "?"),
            )

            def _close_remote_stream() -> None:
                logger.warning(
                    "[RemoteSandbox] Cancellation requested, closing Modal stream | slot=%s class=%s",
                    slot,
                    class_name,
                )
                with suppress(Exception):
                    outputs.close()

            stream_handle = controller.register_generic(_close_remote_stream, label="modal_sandbox_stream_close")
            modal_handle = controller.register_generic(_cancel_modal_call, label="modal_sandbox_call_cancel")
        else:
            logger.info("[RemoteSandbox] No active RunCancellationController detected for slot %s", slot)

        def _maybe_process_control_message(output: Any) -> bool:
            if not isinstance(output, dict):
                return False
            control = output.get("__control__")
            if not isinstance(control, dict):
                return False
            event = control.get("event")
            if event == "modal_call_metadata":
                fc = _hydrate_modal_call_by_id(control.get("function_call_id"), "control_payload")
                _ensure_modal_handle(fc, "control_payload")
                return True
            return False

        if modal_function_call is None:
            logger.info(
                "[RemoteSandbox] Waiting for Modal control metadata to enable cancellation | slot=%s class=%s",
                slot,
                class_name,
            )

        def _stream_with_timeout(gen):
            q: queue.Queue[tuple[str, Any | None, BaseException | None]] = queue.Queue()

            def _pump() -> None:
                try:
                    for item in gen:
                        q.put(("item", item, None))
                except BaseException as exc:
                    q.put(("error", None, exc))
                finally:
                    q.put(("eof", None, None))

            t = threading.Thread(target=_pump, daemon=True)
            t.start()

            got_first = False
            try:
                while True:
                    try:
                        kind, item, err = q.get(timeout=first_payload_timeout if not got_first else 1.0)
                    except queue.Empty:
                        if not got_first:
                            raise TimeoutError(f"No sandbox output within {first_payload_timeout}s (likely not started)")
                        # Keep checking cancellation regularly after first item
                        if controller:
                            if controller.is_cancelled():
                                logger.info(
                                    "[RemoteSandbox] Detected cancellation signal while waiting for stream | slot=%s class=%s",
                                    slot,
                                    class_name,
                                )
                            controller.raise_if_cancelled("modal_sandbox_stream")
                        continue

                    if kind == "item":
                        got_first = True
                        yield item
                    elif kind == "error":
                        assert err is not None
                        raise err
                    elif kind == "eof":
                        break
            finally:
                with suppress(Exception):
                    gen.close()
                t.join(timeout=2)

        try:
            for output in _stream_with_timeout(outputs):
                if _maybe_process_control_message(output):
                    continue
                if controller:
                    if controller.is_cancelled():
                        logger.info(
                            "[RemoteSandbox] Detected cancellation signal while streaming | slot=%s class=%s",
                            slot,
                            class_name,
                        )
                    controller.raise_if_cancelled("modal_sandbox_stream")
                yield output
        finally:
            if controller:
                if stream_handle:
                    controller.unregister(stream_handle)
                if modal_handle:
                    controller.unregister(modal_handle)
                logger.info(
                    "[RemoteSandbox] Unregistered cancellation callbacks | slot=%s class=%s",
                    slot,
                    class_name,
                )


class _RunSandboxedCodeAdapter:
    """Backward compatible shim exposing .remote/.remote_gen helpers."""

    def remote(self, **kwargs):
        return execute_remote_sandbox(**kwargs)

    def remote_gen(self, **kwargs):  # pragma: no cover - alias for modal API compatibility
        return execute_remote_sandbox(**kwargs)


run_sandboxed_code = _RunSandboxedCodeAdapter()