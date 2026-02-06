import asyncio
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from e2b_code_interpreter import AsyncSandbox
from utils.run_cancellation import get_active_run_controller

logger = logging.getLogger(__name__)

_PAYLOAD_PREFIX = "__E2B_PAYLOAD__:"

_sandbox_pool_lock = threading.Lock()
_sandbox_pools: Dict[str, "_SandboxPool"] = {}

_async_loop: Optional[asyncio.AbstractEventLoop] = None
_async_loop_thread: Optional[threading.Thread] = None
_async_loop_ready = threading.Event()
_async_loop_lock = threading.Lock()


def _ensure_async_loop() -> asyncio.AbstractEventLoop:
    global _async_loop
    global _async_loop_thread
    with _async_loop_lock:
        if _async_loop and _async_loop.is_running():
            return _async_loop

        _async_loop_ready.clear()

        def _loop_worker() -> None:
            global _async_loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _async_loop = loop
            _async_loop_ready.set()
            loop.run_forever()

        _async_loop_thread = threading.Thread(target=_loop_worker, daemon=True)
        _async_loop_thread.start()
        _async_loop_ready.wait(timeout=5)

        if _async_loop is None:
            raise RuntimeError("Failed to initialize E2B async event loop")

        return _async_loop


class _PooledSandbox:
    def __init__(self, sandbox: AsyncSandbox, idle_timeout: float) -> None:
        self.sandbox = sandbox
        self._idle_timeout = idle_timeout
        self._idle_event = threading.Event()
        self._idle_thread = threading.Thread(target=self._idle_worker, daemon=True)
        self._idle_thread.start()

    def _idle_worker(self) -> None:
        if not self._idle_event.wait(timeout=self._idle_timeout):
            logger.info("[E2B Pool] Idle timeout reached; killing sandbox")
            _kill_sandbox_sync(self.sandbox)

    def mark_in_use(self) -> None:
        self._idle_event.set()


class _SandboxPool:
    def __init__(
        self,
        *,
        run_id: str,
        template_name: str,
        envs: Dict[str, str],
        idle_timeout: float,
        max_retries: int,
        stagger_seconds: float,
    ) -> None:
        self.run_id = run_id
        self.template_name = template_name
        self.envs = envs
        self.idle_timeout = idle_timeout
        self.max_retries = max_retries
        self.stagger_seconds = max(0.0, stagger_seconds)
        self._sandboxes: List[_PooledSandbox] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self, count: int) -> None:
        logger.info("[E2B Pool] Starting prestart worker run_id=%s count=%s", self.run_id, count)
        threading.Thread(target=self._start_worker, args=(count,), daemon=True).start()

    def _start_worker(self, count: int) -> None:
        google_creds_text = _load_google_credentials_text()
        for idx in range(max(0, count)):
            if self._stop_event.is_set():
                break
            logger.info("[E2B Pool] Creating sandbox %s/%s run_id=%s", idx + 1, count, self.run_id)
            sandbox = _create_sandbox_with_retries(
                template_name=self.template_name,
                envs=self.envs,
                max_retries=self.max_retries,
            )
            if sandbox is not None:
                logger.info("[E2B Pool] Warming sandbox run_id=%s", self.run_id)
                _warmup_sandbox_sync(sandbox, google_creds_text)
                pooled = _PooledSandbox(sandbox, idle_timeout=self.idle_timeout)
                with self._lock:
                    self._sandboxes.append(pooled)
                logger.info("[E2B Pool] Sandbox ready run_id=%s pool_size=%s", self.run_id, len(self._sandboxes))
            else:
                logger.warning("[E2B Pool] Sandbox create failed run_id=%s", self.run_id)
            if self.stagger_seconds and idx < count - 1:
                time.sleep(self.stagger_seconds)

    def acquire(self) -> Optional[AsyncSandbox]:
        with self._lock:
            if not self._sandboxes:
                logger.info("[E2B Pool] No prestarted sandbox available run_id=%s", self.run_id)
                return None
            pooled = self._sandboxes.pop(0)
            logger.info("[E2B Pool] Acquired sandbox run_id=%s remaining=%s", self.run_id, len(self._sandboxes))
        pooled.mark_in_use()
        return pooled.sandbox

    def kill_all(self) -> None:
        self._stop_event.set()
        with self._lock:
            sandboxes = [p.sandbox for p in self._sandboxes]
            self._sandboxes = []
        logger.info("[E2B Pool] Killing %s prestarted sandboxes run_id=%s", len(sandboxes), self.run_id)
        for sandbox in sandboxes:
            _kill_sandbox_sync(sandbox)


def _create_sandbox_with_retries(*, template_name: str, envs: Dict[str, str], max_retries: int) -> Optional[AsyncSandbox]:
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            return _create_sandbox_sync(template_name=template_name, envs=envs)
        except Exception:
            logger.warning("[E2B Pool] Sandbox create attempt %s/%s failed", attempt + 1, attempts)
            if attempt >= attempts - 1:
                break
            time.sleep(0.5 * (attempt + 1))
    return None


def _create_sandbox_sync(*, template_name: str, envs: Dict[str, str]) -> AsyncSandbox:
    loop = _ensure_async_loop()

    logger.info("[E2B] Creating sandbox (template=%s)", template_name)

    async def _create() -> AsyncSandbox:
        return await AsyncSandbox.create(
            template=template_name,
            timeout=500,
            envs=envs,
            request_timeout=500,
        )

    future = asyncio.run_coroutine_threadsafe(_create(), loop)
    return future.result()


def _warmup_sandbox_sync(sandbox: AsyncSandbox, google_creds_text: Optional[str]) -> None:
    loop = _ensure_async_loop()

    warmup_code = r'''
import os
import sys

repo_root = os.environ.get("MGS_REPO_ROOT", "/root")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if os.path.isdir(repo_root):
    try:
        os.chdir(repo_root)
    except Exception:
        pass

try:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sqlalchemy  # noqa: F401
    import psycopg2  # noqa: F401
    import b2sdk.v1  # noqa: F401
    import geoalchemy2  # noqa: F401
    import langchain_community  # noqa: F401
    import langchain_google_vertexai  # noqa: F401
except Exception:
    pass

try:
    import parameters  # noqa: F401
    import table_info  # noqa: F401
    import agents.extraction_sandbox_agent  # noqa: F401
    import agents.timeseries_plot_sandbox_agent  # noqa: F401
    import agents.map_plot_sandbox_agent  # noqa: F401
    import agents.review_level_agents  # noqa: F401
    import tools.sql_security_toolkit  # noqa: F401
    import tools.artefact_toolkit  # noqa: F401
    import tools.create_output_toolkit  # noqa: F401
    import utils.project_selection  # noqa: F401
except Exception:
    pass

print("[warmup] ok")
'''

    async def _warmup() -> None:
        if google_creds_text:
            try:
                await sandbox.files.write("/home/user/google_credentials.json", google_creds_text)
            except Exception:
                pass
        try:
            await sandbox.run_code(warmup_code)
        except Exception:
            pass

    future = asyncio.run_coroutine_threadsafe(_warmup(), loop)
    try:
        future.result(timeout=30)
    except Exception:
        pass


def _kill_sandbox_sync(sandbox: AsyncSandbox) -> None:
    loop = _ensure_async_loop()

    logger.info("[E2B] Killing sandbox")

    async def _kill() -> None:
        try:
            await sandbox.kill()
        except Exception:
            pass

    future = asyncio.run_coroutine_threadsafe(_kill(), loop)
    try:
        future.result(timeout=10)
    except Exception:
        pass


def start_sandbox_pool(
    *,
    run_id: str,
    count: int,
    envs: Dict[str, str],
    template_name: Optional[str] = None,
    idle_timeout: float = 120.0,
    max_retries: int = 5,
    stagger_seconds: float = 0.2,
    controller: Any | None = None,
) -> None:
    if not run_id:
        return
    template = template_name or os.environ.get("E2B_TEMPLATE_NAME", "mos-explore-sandbox")
    logger.info("[E2B Pool] Initializing pool run_id=%s template=%s", run_id, template)
    pool = _SandboxPool(
        run_id=run_id,
        template_name=template,
        envs=envs,
        idle_timeout=idle_timeout,
        max_retries=max_retries,
        stagger_seconds=stagger_seconds,
    )
    with _sandbox_pool_lock:
        existing = _sandbox_pools.pop(run_id, None)
        if existing:
            logger.info("[E2B Pool] Replacing existing pool run_id=%s", run_id)
            existing.kill_all()
        _sandbox_pools[run_id] = pool
    pool.start(count)
    if controller is not None:
        controller.register_generic(lambda: shutdown_sandbox_pool(run_id), label="e2b_pool")


def shutdown_sandbox_pool(run_id: str) -> None:
    if not run_id:
        return
    with _sandbox_pool_lock:
        pool = _sandbox_pools.pop(run_id, None)
    if pool:
        logger.info("[E2B Pool] Shutdown requested run_id=%s", run_id)
        pool.kill_all()


def build_sandbox_envs_for_pool(
    *,
    user_id: int,
    global_hierarchy_access: bool,
    thread_id: str,
    selected_project_key: Optional[str],
) -> Dict[str, str]:
    envs = _build_sandbox_envs(
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
        thread_id=thread_id,
        selected_project_key=selected_project_key,
    )
    envs["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/user/google_credentials.json"
    return envs


def _make_step_payload(content: str, typ: str, *, origin: str = "sandbox") -> Dict[str, Any]:
    origin_value = (origin or "sandbox").strip().lower()
    if origin_value == "stderr":
        level = "error"
    elif origin_value == "stdout":
        level = "debug"
    else:
        level = "error" if str(typ).lower() == "error" else "debug"

    return {
        "content": content or "",
        "additional_kwargs": {
            "level": level,
            "is_final": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": {
                "process": "sandbox_log",
                "thinking_stage": None,
                "branch_id": None,
            },
            "is_child": True,
            "artefacts": [],
        },
    }


def _iter_log_chunks(data: Any) -> List[str]:
    if data is None:
        return []
    if isinstance(data, (list, tuple)):
        chunks: List[str] = []
        for item in data:
            if item is None:
                continue
            text = str(item)
            if text.strip():
                chunks.append(text)
        return chunks
    text = str(data)
    if not text.strip():
        return []
    return [text]


def _process_stream_chunk(text: str, origin: str, stream_sandbox_logs: bool) -> List[Dict[str, Any]]:
    if not text:
        return []
    payloads: List[Dict[str, Any]] = []
    non_payload_lines: List[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line:
            continue
        if line.startswith(_PAYLOAD_PREFIX):
            payload_text = line[len(_PAYLOAD_PREFIX):].strip()
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = _make_step_payload(payload_text, "error", origin=origin)
            payloads.append(payload)
        else:
            non_payload_lines.append(line)

    if non_payload_lines and stream_sandbox_logs:
        message = "\n".join(non_payload_lines)
        payloads.append(
            _make_step_payload(
                f"[{origin}] {message}",
                "error" if origin == "stderr" else "progress",
                origin=origin,
            )
        )
    return payloads


def _build_runner_code() -> str:
    return r'''
import asyncio
import contextlib
import inspect
import io
import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import b2sdk.v1 as b2
import psycopg2
from geoalchemy2 import Geometry
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, event

PAYLOAD_PREFIX = "__E2B_PAYLOAD__:"
RUN_INPUT_PATH = "/home/user/run_input.json"

MYSQL_DEFAULT_SESSION_TABLE_LIMIT_BYTES = 16 * 1024 * 1024
SANDBOX_SESSION_TABLE_LIMIT_BYTES = MYSQL_DEFAULT_SESSION_TABLE_LIMIT_BYTES * 10
tool_semaphore = None


def _ensure_basic_logging():
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
        "tools",
        "agents",
        "tools.sql_security_toolkit",
    ):
        logging.getLogger(namespace).setLevel(logging.INFO)


def _make_step_payload(content: str, typ: str, *, origin: str = "sandbox") -> Dict[str, Any]:
    origin_value = (origin or "sandbox").strip().lower()
    if origin_value == "stderr":
        level = "error"
    elif origin_value == "stdout":
        level = "debug"
    else:
        level = "error" if str(typ).lower() == "error" else "debug"

    return {
        "content": content or "",
        "additional_kwargs": {
            "level": level,
            "is_final": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": {
                "process": "sandbox_log",
                "thinking_stage": None,
                "branch_id": None,
            },
            "is_child": True,
            "artefacts": [],
        },
    }


def _format_error_payload(message: Optional[str] = None, *, origin: str = "sandbox") -> Dict[str, Any]:
    safe_message = (message or "").strip()
    typ = "error" if origin == "stderr" else "progress"
    prefix = f"[{origin.upper()}][Step 0]"
    content = f"{prefix} {safe_message}" if safe_message else prefix
    return _make_step_payload(content=content, typ=typ, origin=origin)


def _emit_payload(payload: Dict[str, Any]) -> None:
    print(f"{PAYLOAD_PREFIX}{json.dumps(payload, default=str)}", flush=True)


class _SandboxStream(io.TextIOBase):
    def __init__(self, original: io.TextIOBase, queue: "asyncio.Queue[Optional[str]]", filter_fn=None):
        self._original = original
        self._queue = queue
        self._buffer: str = ""
        self._filter = filter_fn

    def writable(self) -> bool:  # pragma: no cover
        return True

    def write(self, data: str) -> int:  # pragma: no cover
        if not data:
            return 0
        self._original.write(data)
        normalized = data.replace("\r\n", "\n").replace("\r", "\n")
        self._buffer += normalized
        self._flush_lines()
        return len(data)

    def flush(self) -> None:  # pragma: no cover
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
        if trimmed.startswith(PAYLOAD_PREFIX):
            return
        if self._filter and not self._filter(trimmed):
            return
        try:
            self._queue.put_nowait(trimmed)
        except asyncio.QueueFull:  # pragma: no cover
            pass


def _stdout_error_filter(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    upper = stripped.upper()
    if any(marker in upper for marker in ("[WARNING]", "[ERROR]", "[CRITICAL]", "[EXCEPTION]")):
        return True
    lower = stripped.lower()
    if "traceback" in lower or "exception" in lower:
        return True
    return "error" in lower


class _QueueLoggingHandler(logging.Handler):
    def __init__(self, stdout_queue: "asyncio.Queue[Optional[str]]", stderr_queue: "asyncio.Queue[Optional[str]]"):
        super().__init__(level=logging.INFO)
        self._stdout_queue = stdout_queue
        self._stderr_queue = stderr_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return

        target_queue = self._stderr_queue if record.levelno >= logging.WARNING else self._stdout_queue
        if not msg or target_queue is None:
            return
        if target_queue is self._stdout_queue and not _stdout_error_filter(msg):
            return
        try:
            target_queue.put_nowait(msg)
        except asyncio.QueueFull:  # pragma: no cover
            pass


def _drain_stream_lines(q: "asyncio.Queue[Optional[str]]") -> List[str]:
    lines: List[str] = []
    while True:
        try:
            line = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        if line:
            lines.append(line)
    return lines


def _ensure_sandbox_session_limits():
    default_value = str(SANDBOX_SESSION_TABLE_LIMIT_BYTES)
    if "SANDBOX_SESSION_TMP_TABLE_SIZE" not in os.environ:
        os.environ["SANDBOX_SESSION_TMP_TABLE_SIZE"] = default_value
    if "SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE" not in os.environ:
        os.environ["SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE"] = default_value


def _get_int_env(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


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


def _load_run_input() -> Dict[str, Any]:
    if not os.path.isfile(RUN_INPUT_PATH):
        return {}
    try:
        with open(RUN_INPUT_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _get_project_configs(project_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    if not isinstance(project_data, dict):
        return configs
    try:
        from utils.project_selection import make_project_key
    except Exception:
        make_project_key = None

    for key, cfg in project_data.items():
        if not isinstance(cfg, dict):
            continue
        derived_key = ""
        if make_project_key:
            derived_key = make_project_key(str(cfg.get("db_host", "")), str(cfg.get("db_name", "")))
        resolved_key = derived_key or str(key)
        configs[f"project_data.{resolved_key}"] = cfg
    return configs


def _get_env_json(name: str):
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _decorate_code_yield_output(output: Any) -> Any:
    if not isinstance(output, dict):
        return output
    if "__control__" in output:
        return output

    content = output.get("content", "")
    metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
    meta_type = str(metadata.get("type") or "").lower()

    if meta_type in {"plot", "csv", "final"}:
        level = "debug"
    elif meta_type == "error":
        level = "error"
    elif meta_type == "progress":
        level = "info"
    else:
        level = "debug"

    artefact_entry: Dict[str, Any] | None = None
    if meta_type in {"plot", "csv"}:
        content_dict = content if isinstance(content, dict) else {}
        artefact_entry = {
            "type": meta_type,
            "id": content_dict.get("artefact_id"),
            "description": content_dict.get("description"),
            "tool_name": content_dict.get("tool_name"),
        }

    additional_kwargs = output.get("additional_kwargs")
    if not isinstance(additional_kwargs, dict):
        additional_kwargs = {}

    additional_kwargs.update(
        {
            "level": level,
            "is_final": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_child": True,
            "artefacts": [artefact_entry] if artefact_entry else [],
        }
    )

    origin = additional_kwargs.get("origin")
    if not isinstance(origin, dict):
        origin = {}
    origin["process"] = "code_yield"
    additional_kwargs["origin"] = origin

    output["additional_kwargs"] = additional_kwargs
    return output


async def ainvoke(tool, prompt: str, timeout: float | None = None):
    async with tool_semaphore:  # type: ignore[union-attr]
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


def _collect_stream_payloads(stdout_queue, stderr_queue, stdout_stream, stderr_stream, flush: bool = False) -> List[Dict[str, Any]]:
    if stdout_queue is None or stderr_queue is None:
        return []
    if flush:
        stdout_stream.flush_buffer()
        stderr_stream.flush_buffer()
    payloads: List[Dict[str, Any]] = []
    for origin, queue in (("stderr", stderr_queue), ("stdout", stdout_queue)):
        for line in _drain_stream_lines(queue):
            payloads.append(_format_error_payload(message=line, origin=origin))
    return payloads


async def _run():
    _ensure_basic_logging()
    _ensure_sandbox_session_limits()

    repo_root = os.environ.get("MGS_REPO_ROOT", "/root")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if os.path.isdir(repo_root):
        try:
            os.chdir(repo_root)
        except Exception:
            pass

    run_input = _load_run_input()
    table_info = run_input.get("table_info") or []
    table_relationship_graph = run_input.get("table_relationship_graph") or {}
    stream_sandbox_logs = bool(run_input.get("stream_sandbox_logs", True))

    from parameters import include_tables
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
    from tools.artefact_toolkit import WriteArtefactTool

    project_data = _get_env_json("PROJECT_DATA_JSON") or {}
    project_configs = _get_project_configs(project_data)

    selected_project_key = os.environ.get("SANDBOX_PROJECT_KEY")
    cfg = project_configs.get(selected_project_key)
    if not cfg:
        raise RuntimeError(f"PROJECT_DATA_JSON missing or project config not found for key: {selected_project_key}")

    host = str(cfg.get("db_host", ""))
    user = str(cfg.get("db_user", ""))
    password = str(cfg.get("db_pass", ""))
    database = str(cfg.get("db_name", ""))
    port = str(cfg.get("port", "3306"))
    if not all([host, user, password, database]):
        raise RuntimeError("PROJECT_DATA_JSON missing db_host/db_user/db_pass/db_name")

    uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(
        uri,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_timeout=30,
        connect_args={"connection_timeout": 10},
    )
    metadata = MetaData()
    Table('3d_condours', metadata, Column('id', Integer, primary_key=True), Column('contour_bound', Geometry))
    session_tmp_table_size = _get_int_env("SANDBOX_SESSION_TMP_TABLE_SIZE", SANDBOX_SESSION_TABLE_LIMIT_BYTES)
    session_max_heap_table_size = _get_int_env("SANDBOX_SESSION_MAX_HEAP_TABLE_SIZE", SANDBOX_SESSION_TABLE_LIMIT_BYTES)
    _attach_session_configuration(engine, session_tmp_table_size, session_max_heap_table_size)

    db = SQLDatabase(
        engine=engine,
        metadata=metadata,
        include_tables=include_tables,
        sample_rows_in_table_info=3,
        lazy_table_reflection=True,
    )

    user_id = int(os.environ.get("SANDBOX_USER_ID", "0"))
    gha_env = os.environ.get("SANDBOX_GLOBAL_HIERARCHY_ACCESS", "1")
    global_hierarchy_access = gha_env.lower() not in {"0", "false", "no"}
    thread_id = os.environ.get("SANDBOX_THREAD_ID", "e2b-thread")

    vertex_location = os.environ.get("VERTEX_LOCATION", "global")
    vertex_endpoint = os.environ.get("VERTEX_ENDPOINT")
    llm = ChatVertexAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        location=vertex_location,
        api_endpoint=vertex_endpoint,
    )

    general_sql_query_tool = GeneralSQLQueryTool(
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    metadata_db = psycopg2.connect(
        host=os.environ["ARTEFACT_METADATA_RDS_HOST"],
        user=os.environ["ARTEFACT_METADATA_RDS_USER"],
        password=os.environ["ARTEFACT_METADATA_RDS_PASS"],
        database=os.environ["ARTEFACT_METADATA_RDS_DATABASE"],
        port=os.environ.get("ARTEFACT_METADATA_RDS_PORT", "5432"),
        connect_timeout=10,
        options="-c statement_timeout=30000",
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )
    info = b2.InMemoryAccountInfo()
    b2_api = b2.B2Api(info)
    b2_api.authorize_account(
        "production",
        os.environ["ARTEFACT_BLOB_B2_KEY_ID"],
        os.environ["ARTEFACT_BLOB_B2_KEY"],
    )
    blob_db = b2_api.get_bucket_by_name(os.environ["ARTEFACT_BLOB_B2_BUCKET"])
    write_artefact_tool = WriteArtefactTool(blob_db=blob_db, metadata_db=metadata_db)

    extraction_tool = extraction_sandbox_agent(
        llm=llm,
        db=db,
        table_info=table_info,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    timeseries_plot_tool = timeseries_plot_sandbox_agent(
        llm=llm,
        sql_tool=general_sql_query_tool,
        write_artefact_tool=write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )
    map_plot_tool = map_plot_sandbox_agent(
        llm=llm,
        sql_tool=general_sql_query_tool,
        write_artefact_tool=write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )
    review_by_value_tool = review_by_value_agent(
        llm=llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    review_by_time_tool = review_by_time_agent(
        llm=llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    review_schema_tool = review_schema_agent(
        llm=llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    breach_instr_tool = breach_instr_agent(
        llm=llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    review_changes_across_period_tool = review_changes_across_period_agent(
        llm=llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    csv_saver_tool = CSVSaverTool(
        write_artefact_tool=write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )

    tool_limit = _get_int_env("SANDBOX_MAX_CONCURRENT_TOOL_CALLS", 8) or 8
    tool_limit = max(1, tool_limit)
    global tool_semaphore
    tool_semaphore = asyncio.Semaphore(tool_limit)

    if stream_sandbox_logs:
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
    else:
        stdout_queue = None
        stderr_queue = None
        stdout_stream = None
        stderr_stream = None
        stdout_ctx = contextlib.nullcontext()
        stderr_ctx = contextlib.nullcontext()
        queue_logging_handler = None
        root_logger = logging.getLogger()

    async def _yield_outputs(obj):
        if obj is None:
            return
        if inspect.isasyncgen(obj):
            async for item in obj:
                yield item
            return
        if inspect.iscoroutine(obj):
            awaited = await obj
            async for item in _yield_outputs(awaited):
                yield item
            return
        if isinstance(obj, dict):
            yield obj
            return
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

    try:
        with stdout_ctx, stderr_ctx:
            code_path = "/home/user/llm_code.py"
            if not os.path.isfile(code_path):
                _emit_payload(_make_step_payload("Error: llm_code.py not found", "error"))
                return

            with open(code_path, "r", encoding="utf-8") as handle:
                code = handle.read()

            if not code.strip().startswith("from __future__ import annotations"):
                code = "from __future__ import annotations\n" + code

            local_namespace = {
                "extraction_sandbox_agent": extraction_tool,
                "timeseries_plot_sandbox_agent": timeseries_plot_tool,
                "map_plot_sandbox_agent": map_plot_tool,
                "review_by_value_agent": review_by_value_tool,
                "review_by_time_agent": review_by_time_tool,
                "review_schema_agent": review_schema_tool,
                "breach_instr_agent": breach_instr_tool,
                "review_changes_across_period_agent": review_changes_across_period_tool,
                "general_sql_query_tool": general_sql_query_tool,
                "csv_saver_tool": csv_saver_tool,
                "llm": llm,
                "db": db,
                "pd": __import__("pandas"),
                "np": __import__("numpy"),
                "ainvoke": ainvoke,
                "asyncio": asyncio,
                "PARALLEL_ENABLED": True,
            }

            exec(code, local_namespace)

            if "execute_strategy" not in local_namespace or not callable(local_namespace["execute_strategy"]):
                _emit_payload(_make_step_payload("Error: Code must define an 'execute_strategy' function.", "error"))
                return

            result = local_namespace["execute_strategy"]()
            async for output in _yield_outputs(result):
                if stream_sandbox_logs and stdout_queue is not None and stderr_queue is not None:
                    for payload in _collect_stream_payloads(stdout_queue, stderr_queue, stdout_stream, stderr_stream):
                        _emit_payload(payload)
                _emit_payload(_decorate_code_yield_output(output))

            if stream_sandbox_logs and stdout_queue is not None and stderr_queue is not None:
                for payload in _collect_stream_payloads(stdout_queue, stderr_queue, stdout_stream, stderr_stream, flush=True):
                    _emit_payload(payload)
    finally:
        if queue_logging_handler is not None:
            root_logger.removeHandler(queue_logging_handler)
            queue_logging_handler.close()
        try:
            metadata_db.close()
        except Exception:
            pass


def main():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
        return

    error = {"exc": None}

    def _runner():
        try:
            asyncio.run(_run())
        except Exception as exc:
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error["exc"] is not None:
        raise error["exc"]


if __name__ == "__main__":
    main()
'''


def _build_sandbox_envs(
    *,
    user_id: int,
    global_hierarchy_access: bool,
    thread_id: str,
    selected_project_key: Optional[str],
) -> Dict[str, str]:
    envs: Dict[str, str] = {}
    for key in (
        "ARTEFACT_METADATA_RDS_HOST",
        "ARTEFACT_METADATA_RDS_USER",
        "ARTEFACT_METADATA_RDS_PASS",
        "ARTEFACT_METADATA_RDS_DATABASE",
        "ARTEFACT_METADATA_RDS_PORT",
        "ARTEFACT_BLOB_B2_KEY_ID",
        "ARTEFACT_BLOB_B2_KEY",
        "ARTEFACT_BLOB_B2_BUCKET",
        "PROJECT_DATA_JSON",
        "VERTEX_LOCATION",
        "VERTEX_ENDPOINT",
    ):
        value = os.environ.get(key)
        if value:
            envs[key] = str(value)

    envs["SANDBOX_USER_ID"] = str(user_id)
    envs["SANDBOX_GLOBAL_HIERARCHY_ACCESS"] = "1" if global_hierarchy_access else "0"
    envs["SANDBOX_THREAD_ID"] = str(thread_id)
    if selected_project_key:
        envs["SANDBOX_PROJECT_KEY"] = str(selected_project_key)

    return envs


def _load_google_credentials_text() -> Optional[str]:
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        return None
    path = Path(creds_path)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def execute_remote_sandbox(
    code: str,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    selected_project_key: Optional[str] = None,
    container_slot: Optional[int] = None,
    first_payload_timeout: Optional[int] = None,
    stream_sandbox_logs: bool = True,
) -> Generator[dict, None, None]:
    del container_slot

    template_name = os.environ.get("E2B_TEMPLATE_NAME", "mos-explore-sandbox")
    envs = _build_sandbox_envs(
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
        thread_id=thread_id,
        selected_project_key=selected_project_key,
    )
    envs["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/user/google_credentials.json"

    google_creds_text = _load_google_credentials_text()
    run_input = {
        "table_info": table_info,
        "table_relationship_graph": table_relationship_graph,
        "stream_sandbox_logs": bool(stream_sandbox_logs),
    }

    output_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

    def _queue_payload(payload: Dict[str, Any]) -> None:
        output_queue.put(payload)

    def _queue_done() -> None:
        output_queue.put(None)

    def _queue_status(message: str) -> None:
        payload = _make_step_payload(message, "progress", origin="sandbox")
        additional = payload.get("additional_kwargs")
        if isinstance(additional, dict):
            additional["level"] = "info"
            origin = additional.get("origin")
            if isinstance(origin, dict):
                origin["process"] = "sandbox_status"
        _queue_payload(payload)

    _queue_status(f"Sandbox queued (template={template_name})")

    def _handle_stream_data(data: Any, origin: str) -> None:
        if not stream_sandbox_logs:
            raw = "" if data is None else str(data)
            if _PAYLOAD_PREFIX not in raw:
                return
        for chunk in _iter_log_chunks(data):
            for payload in _process_stream_chunk(chunk, origin, stream_sandbox_logs):
                _queue_payload(payload)

    async def _run_async() -> None:
        controller = get_active_run_controller()
        pooled_sandbox: Optional[AsyncSandbox] = None
        if controller is not None:
            with _sandbox_pool_lock:
                pool = _sandbox_pools.get(controller.run_id)
            if pool is not None:
                pooled_sandbox = pool.acquire()

        sandbox = pooled_sandbox
        if sandbox is None:
            sandbox = await AsyncSandbox.create(
                template=template_name,
                timeout=500,
                envs=envs,
                request_timeout=500,
            )

        cancel_handle: Optional[str] = None
        if controller is not None:
            cancel_handle = controller.register_generic(
                lambda: _kill_sandbox_sync(sandbox),
                label="e2b_sandbox",
            )
        try:
            if google_creds_text:
                await sandbox.files.write("/home/user/google_credentials.json", google_creds_text)
            await sandbox.files.write("/home/user/run_input.json", json.dumps(run_input))
            await sandbox.files.write("/home/user/llm_code.py", code)
            runner_code = _build_runner_code()
            await sandbox.run_code(
                runner_code,
                on_stdout=lambda data: _handle_stream_data(data, "stdout"),
                on_stderr=lambda data: _handle_stream_data(data, "stderr"),
                on_error=lambda error: _queue_payload(_make_step_payload(str(error), "error", origin="stderr")),
            )
        finally:
            try:
                await sandbox.kill()
            except Exception:
                pass
            if controller is not None and cancel_handle:
                controller.unregister(cancel_handle)

    loop = _ensure_async_loop()
    future = asyncio.run_coroutine_threadsafe(_run_async(), loop)

    def _on_done(done_future: "asyncio.Future[None]") -> None:
        try:
            done_future.result()
        except Exception as exc:
            _queue_payload(_make_step_payload(str(exc), "error", origin="stderr"))
        finally:
            _queue_done()

    future.add_done_callback(_on_done)

    first_deadline = None
    if first_payload_timeout is None:
        first_payload_timeout = int(os.environ.get("SANDBOX_FIRST_PAYLOAD_TIMEOUT_SECONDS", "20"))
    if first_payload_timeout:
        first_deadline = time.monotonic() + max(1, first_payload_timeout)

    got_first = False
    while True:
        timeout = 1.0
        if not got_first and first_deadline is not None:
            timeout = max(0.1, first_deadline - time.monotonic())
        try:
            item = output_queue.get(timeout=timeout)
        except queue.Empty:
            if not got_first and first_deadline is not None and time.monotonic() >= first_deadline:
                raise TimeoutError(f"No sandbox output within {first_payload_timeout}s (likely not started)")
            continue
        if item is None:
            break
        got_first = True
        yield item


class _RunSandboxedCodeAdapter:
    def remote(self, **kwargs):
        return execute_remote_sandbox(**kwargs)

    def remote_gen(self, **kwargs):
        return execute_remote_sandbox(**kwargs)


run_sandboxed_code = _RunSandboxedCodeAdapter()
