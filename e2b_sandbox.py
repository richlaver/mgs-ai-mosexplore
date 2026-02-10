import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from e2b_code_interpreter import AsyncSandbox
from utils.run_cancellation import get_active_run_controller

logger = logging.getLogger(__name__)

_PAYLOAD_PREFIX = "__E2B_PAYLOAD__:"

class _AsyncLoopRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._shutdown = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None

    def _thread_main(self) -> None:
        while not self._shutdown.is_set():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._lock:
                self._loop = loop
                self._thread_id = threading.get_ident()
                self._ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.stop()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
                with self._lock:
                    self._loop = None
                    self._thread_id = None
                    self._ready.clear()
            if not self._shutdown.is_set():
                time.sleep(0.05)

    def ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is not None and loop.is_running():
                return loop
            if thread is None or not thread.is_alive():
                self._thread = threading.Thread(target=self._thread_main, daemon=True)
                self._thread.start()

        if not self._ready.wait(timeout=5):
            raise RuntimeError("Failed to initialize E2B async event loop")
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                raise RuntimeError("E2B async event loop not running")
            return self._loop

    def is_loop_thread(self) -> bool:
        with self._lock:
            return self._thread_id == threading.get_ident() and self._loop is not None

    def stop_loop(self) -> None:
        with self._lock:
            loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

    def submit(self, coro: Any) -> concurrent.futures.Future:
        last_exc: Exception | None = None
        for _ in range(2):
            loop = self.ensure_loop()
            try:
                return asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError as exc:
                last_exc = exc
                self.stop_loop()
                time.sleep(0.05)
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to schedule coroutine")


_ASYNC_LOOP_RUNNER = _AsyncLoopRunner()

_SANDBOX_POOL_LOCK = threading.Lock()
_SANDBOX_POOL_CONDITION = threading.Condition(_SANDBOX_POOL_LOCK)
_SANDBOX_POOL: Dict[int, "SandboxSlot"] = {}
_ACTIVE_EXECUTIONS_LOCK = threading.Lock()
_ACTIVE_EXECUTIONS = 0


@dataclass(slots=True)
class SandboxSlot:
    slot_id: int
    sandbox: Optional[AsyncSandbox] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"
    preinit_done: bool = False
    in_use: bool = False
    used: bool = False
    last_error: Optional[str] = None
    cancel_handle: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False


def _assert_template_ready(template_name: str) -> None:
    if not template_name:
        raise RuntimeError("E2B template name is required but missing")


def _set_pool_slot(slot: SandboxSlot) -> None:
    with _SANDBOX_POOL_CONDITION:
        _SANDBOX_POOL[slot.slot_id] = slot
        _SANDBOX_POOL_CONDITION.notify_all()


def _get_pool_slot(slot_id: int) -> Optional[SandboxSlot]:
    with _SANDBOX_POOL_CONDITION:
        return _SANDBOX_POOL.get(slot_id)


def _remove_pool_slot(slot_id: int) -> Optional[SandboxSlot]:
    with _SANDBOX_POOL_CONDITION:
        slot = _SANDBOX_POOL.pop(slot_id, None)
        _SANDBOX_POOL_CONDITION.notify_all()
        return slot


def _claim_pool_slot(slot_id: int) -> Optional[SandboxSlot]:
    with _SANDBOX_POOL_CONDITION:
        slot = _SANDBOX_POOL.get(slot_id)
        if (
            not slot
            or slot.cancelled
            or not slot.preinit_done
            or slot.in_use
            or slot.used
            or slot.sandbox is None
        ):
            return None
        slot.in_use = True
        slot.used = True
        slot.status = "in_use"
        _SANDBOX_POOL_CONDITION.notify_all()
        return slot


def _claim_preinit_failed_slot_locked(prefer_slot_id: Optional[int]) -> Optional[SandboxSlot]:
    candidates = list(_SANDBOX_POOL.values())
    if prefer_slot_id is not None:
        prefer = _SANDBOX_POOL.get(prefer_slot_id)
        if (
            prefer
            and not prefer.cancelled
            and prefer.status == "preinit_failed"
            and not prefer.in_use
            and not prefer.used
            and prefer.sandbox is not None
        ):
            prefer.in_use = True
            prefer.used = True
            prefer.status = "in_use"
            _SANDBOX_POOL_CONDITION.notify_all()
            return prefer

    for slot in candidates:
        if (
            slot.cancelled
            or slot.status != "preinit_failed"
            or slot.in_use
            or slot.used
            or slot.sandbox is None
        ):
            continue
        slot.in_use = True
        slot.used = True
        slot.status = "in_use"
        _SANDBOX_POOL_CONDITION.notify_all()
        return slot
    return None


def _claim_ready_slot_locked(prefer_slot_id: Optional[int]) -> Optional[SandboxSlot]:
    candidates = list(_SANDBOX_POOL.values())
    if prefer_slot_id is not None:
        prefer = _SANDBOX_POOL.get(prefer_slot_id)
        if (
            prefer
            and not prefer.cancelled
            and prefer.preinit_done
            and not prefer.in_use
            and not prefer.used
            and prefer.sandbox is not None
        ):
            prefer.in_use = True
            prefer.used = True
            prefer.status = "in_use"
            _SANDBOX_POOL_CONDITION.notify_all()
            return prefer

    for slot in candidates:
        if (
            slot.cancelled
            or not slot.preinit_done
            or slot.in_use
            or slot.used
            or slot.sandbox is None
        ):
            continue
        slot.in_use = True
        slot.used = True
        slot.status = "in_use"
        _SANDBOX_POOL_CONDITION.notify_all()
        return slot
    return None


def _claim_preinit_failed_slot(prefer_slot_id: Optional[int]) -> Optional[SandboxSlot]:
    with _SANDBOX_POOL_CONDITION:
        return _claim_preinit_failed_slot_locked(prefer_slot_id)


def _claim_ready_slot(prefer_slot_id: Optional[int]) -> Optional[SandboxSlot]:
    with _SANDBOX_POOL_CONDITION:
        return _claim_ready_slot_locked(prefer_slot_id)


def _has_in_progress_slots() -> bool:
    in_progress_status = {"creating", "created", "preinit"}
    with _SANDBOX_POOL_CONDITION:
        return any(
            slot.status in in_progress_status and not slot.cancelled for slot in _SANDBOX_POOL.values()
        )


def _wait_for_pool_slot(prefer_slot_id: Optional[int]) -> Optional[SandboxSlot]:
    with _SANDBOX_POOL_CONDITION:
        while True:
            slot = _claim_ready_slot_locked(prefer_slot_id)
            if slot:
                return slot
            slot = _claim_preinit_failed_slot_locked(prefer_slot_id)
            if slot:
                return slot
            in_progress = any(
                s.status in {"creating", "created", "preinit"} and not s.cancelled
                for s in _SANDBOX_POOL.values()
            )
            if not in_progress:
                return None
            _SANDBOX_POOL_CONDITION.wait(timeout=1.0)


def _get_parallel_executions() -> int:
    raw = os.environ.get("MGS_NUM_PARALLEL_EXECUTIONS")
    try:
        value = int(raw) if raw is not None else 1
    except Exception:
        value = 1
    return max(1, value)


def _increment_active_executions() -> int:
    global _ACTIVE_EXECUTIONS
    with _ACTIVE_EXECUTIONS_LOCK:
        _ACTIVE_EXECUTIONS += 1
        return _ACTIVE_EXECUTIONS


def _decrement_active_executions() -> int:
    global _ACTIVE_EXECUTIONS
    with _ACTIVE_EXECUTIONS_LOCK:
        _ACTIVE_EXECUTIONS = max(0, _ACTIVE_EXECUTIONS - 1)
        return _ACTIVE_EXECUTIONS


def _cleanup_unused_pool(reason: str) -> None:
    to_kill: List[SandboxSlot] = []
    with _SANDBOX_POOL_CONDITION:
        for slot_id, slot in list(_SANDBOX_POOL.items()):
            if slot.in_use or slot.used:
                continue
            if slot.status not in {"creating", "created", "preinit", "preinit_failed"}:
                continue
            slot.cancelled = True
            _SANDBOX_POOL.pop(slot_id, None)
            to_kill.append(slot)
        _SANDBOX_POOL_CONDITION.notify_all()

    if not to_kill:
        return
    logger.info("[E2B Pool] Cleaning up unused sandboxes reason=%s count=%d", reason, len(to_kill))
    for slot in to_kill:
        sandbox = slot.sandbox
        if sandbox is None:
            continue
        _kill_sandbox_sync(sandbox)


def reset_sandbox_pool(reason: str | None = None) -> None:
    slots: List[SandboxSlot] = []
    with _SANDBOX_POOL_CONDITION:
        slots = list(_SANDBOX_POOL.values())
        _SANDBOX_POOL.clear()
        _SANDBOX_POOL_CONDITION.notify_all()
    if not slots:
        return
    logger.info("[E2B Pool] Resetting sandbox pool reason=%s count=%d", reason, len(slots))
    for slot in slots:
        sandbox = slot.sandbox
        if sandbox is None:
            continue
        _kill_sandbox_sync(sandbox)


def _build_preinit_code() -> str:
    return """def execute_strategy():\n    return []\n"""


def _kill_sandbox_sync(sandbox: AsyncSandbox) -> None:
    logger.info("[E2B] Killing sandbox")

    async def _kill() -> None:
        try:
            await sandbox.kill()
        except Exception:
            pass

    if _ASYNC_LOOP_RUNNER.is_loop_thread():
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_kill())
        except Exception:
            pass
        return

    future = _ASYNC_LOOP_RUNNER.submit(_kill())
    try:
        future.result(timeout=10)
    except Exception:
        pass


def _run_async_with_retry(coro_factory: Callable[[], Any], *, timeout: Optional[float], label: str) -> None:
    last_exc: Exception | None = None
    for attempt in range(2):
        future = _ASYNC_LOOP_RUNNER.submit(coro_factory())
        try:
            future.result(timeout=timeout)
            return
        except (RuntimeError, concurrent.futures.CancelledError) as exc:
            last_exc = exc
            message = str(exc).lower()
            if attempt == 0 and ("loop" in message or "event loop" in message):
                logger.warning("[E2B] %s failed due to loop error; restarting loop", label)
                _ASYNC_LOOP_RUNNER.stop_loop()
                time.sleep(0.05)
                continue
            raise
    if last_exc is not None:
        raise last_exc


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
    logger = logging.getLogger("e2b_runner")

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

    preinit = globals().get("MGS_PREINIT")
    preinit_ready = isinstance(preinit, dict) and preinit.get("ready")
    global tool_semaphore

    if preinit_ready:
        logger.info("[runner] using preinitialized cache")
        table_info = preinit.get("table_info", table_info)
        table_relationship_graph = preinit.get("table_relationship_graph", table_relationship_graph)
        db = preinit["db"]
        llm = preinit["llm"]
        general_sql_query_tool = preinit["general_sql_query_tool"]
        metadata_db = preinit["metadata_db"]
        blob_db = preinit["blob_db"]
        write_artefact_tool = preinit["write_artefact_tool"]
        extraction_tool = preinit["extraction_tool"]
        timeseries_plot_tool = preinit["timeseries_plot_tool"]
        map_plot_tool = preinit["map_plot_tool"]
        review_by_value_tool = preinit["review_by_value_tool"]
        review_by_time_tool = preinit["review_by_time_tool"]
        review_schema_tool = preinit["review_schema_tool"]
        breach_instr_tool = preinit["breach_instr_tool"]
        review_changes_across_period_tool = preinit["review_changes_across_period_tool"]
        csv_saver_tool = preinit["csv_saver_tool"]
        tool_semaphore = preinit.get("tool_semaphore")
        if tool_semaphore is None:
            tool_limit = _get_int_env("SANDBOX_MAX_CONCURRENT_TOOL_CALLS", 8) or 8
            tool_limit = max(1, tool_limit)
            tool_semaphore = asyncio.Semaphore(tool_limit)
            preinit["tool_semaphore"] = tool_semaphore
    else:
        logger.info("[runner] preinit cache missing; initializing")
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
        tool_semaphore = asyncio.Semaphore(tool_limit)

        globals()["MGS_PREINIT"] = {
            "ready": True,
            "table_info": table_info,
            "table_relationship_graph": table_relationship_graph,
            "db": db,
            "llm": llm,
            "general_sql_query_tool": general_sql_query_tool,
            "metadata_db": metadata_db,
            "blob_db": blob_db,
            "write_artefact_tool": write_artefact_tool,
            "extraction_tool": extraction_tool,
            "timeseries_plot_tool": timeseries_plot_tool,
            "map_plot_tool": map_plot_tool,
            "review_by_value_tool": review_by_value_tool,
            "review_by_time_tool": review_by_time_tool,
            "review_schema_tool": review_schema_tool,
            "breach_instr_tool": breach_instr_tool,
            "review_changes_across_period_tool": review_changes_across_period_tool,
            "csv_saver_tool": csv_saver_tool,
            "tool_semaphore": tool_semaphore,
        }
        logger.info("[runner] preinit cache stored")

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
            if not locals().get("preinit_ready", False):
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


def prewarm_sandbox(
    *,
    slot_id: int,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    selected_project_key: Optional[str] = None,
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> SandboxSlot:
    def _log(message: str, level: str = "debug") -> None:
        if log_callback:
            log_callback(message, level)

    existing = _get_pool_slot(slot_id)
    if existing and existing.preinit_done and existing.sandbox is not None:
        _log(f"Sandbox slot {slot_id} already prewarmed", "debug")
        return existing
    if existing and existing.status in {"creating", "preinit"}:
        _log(f"Sandbox slot {slot_id} already prewarming", "debug")
        return existing

    slot = SandboxSlot(slot_id=slot_id, status="creating")
    slot.metadata = {
        "slot_id": str(slot_id),
        "branch_id": str(slot_id),
        "purpose": "mgs_preinit",
    }
    _set_pool_slot(slot)

    template_name = os.environ.get("E2B_TEMPLATE_NAME", "mos-explore-sandbox")
    _assert_template_ready(template_name)
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
        "stream_sandbox_logs": False,
    }

    _log(f"Sandbox slot {slot_id} creating (template={template_name})", "debug")

    async def _run_async() -> None:
        controller = get_active_run_controller()
        try:
            sandbox = await AsyncSandbox.create(
                template=template_name,
                timeout=500,
                envs=envs,
                request_timeout=500,
                metadata=slot.metadata,
            )
            slot.sandbox = sandbox
            with _SANDBOX_POOL_CONDITION:
                slot.status = "created"
                _SANDBOX_POOL_CONDITION.notify_all()
            _log(f"Sandbox slot {slot_id} created", "debug")

            if slot.cancelled:
                _log(f"Sandbox slot {slot_id} cancelled after create", "debug")
                try:
                    await sandbox.kill()
                except Exception:
                    pass
                _remove_pool_slot(slot_id)
                return

            if controller is not None:
                slot.cancel_handle = controller.register_e2b_sandbox(
                    sandbox,
                    label=f"e2b_sandbox_prewarm_{slot_id}",
                    kill_callback=lambda: _kill_sandbox_sync(sandbox),
                )

            if google_creds_text:
                _log(f"Sandbox slot {slot_id} writing google credentials", "debug")
                await sandbox.files.write("/home/user/google_credentials.json", google_creds_text)

            _log(f"Sandbox slot {slot_id} writing run_input", "debug")
            await sandbox.files.write("/home/user/run_input.json", json.dumps(run_input))
            _log(f"Sandbox slot {slot_id} writing preinit code", "debug")
            await sandbox.files.write("/home/user/llm_code.py", _build_preinit_code())

            runner_code = _build_runner_code()
            _log(f"Sandbox slot {slot_id} running preinit", "debug")
            with _SANDBOX_POOL_CONDITION:
                slot.status = "preinit"
                _SANDBOX_POOL_CONDITION.notify_all()

            def _handle_stream(data: Any, origin: str) -> None:
                for chunk in _iter_log_chunks(data):
                    for payload in _process_stream_chunk(chunk, origin, True):
                        content = str(payload.get("content", "")).strip()
                        additional = payload.get("additional_kwargs")
                        level = "debug"
                        if isinstance(additional, dict):
                            level = str(additional.get("level") or level)
                        if content:
                            _log(content, "error" if level == "error" else "debug")

            await sandbox.run_code(
                runner_code,
                on_stdout=lambda data: _handle_stream(data, "stdout"),
                on_stderr=lambda data: _handle_stream(data, "stderr"),
                on_error=lambda error: _log(str(error), "error"),
            )

            with _SANDBOX_POOL_CONDITION:
                slot.preinit_done = True
                slot.status = "ready"
                _SANDBOX_POOL_CONDITION.notify_all()
            _log(f"Sandbox slot {slot_id} preinit complete", "debug")
        except Exception as exc:
            slot.last_error = str(exc)
            if slot.sandbox is not None:
                with _SANDBOX_POOL_CONDITION:
                    slot.status = "preinit_failed"
                    slot.preinit_done = False
                    _SANDBOX_POOL_CONDITION.notify_all()
            else:
                with _SANDBOX_POOL_CONDITION:
                    slot.status = "error"
                    _SANDBOX_POOL_CONDITION.notify_all()
            _log(f"Sandbox slot {slot_id} preinit failed: {exc}", "error")
            if slot.sandbox is None:
                _remove_pool_slot(slot_id)
            if slot.cancelled and slot.sandbox is not None:
                try:
                    await slot.sandbox.kill()
                except Exception:
                    pass
                _remove_pool_slot(slot_id)

    _run_async_with_retry(_run_async, timeout=600, label="sandbox_prewarm")
    return slot


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
    logger.info("[E2B Sandbox] execute_remote_sandbox start thread_id=%s user_id=%s", thread_id, user_id)

    template_name = os.environ.get("E2B_TEMPLATE_NAME", "mos-explore-sandbox")
    _assert_template_ready(template_name)
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

    pooled_slot = None
    if isinstance(container_slot, int):
        pooled_slot = _claim_pool_slot(container_slot)
    if pooled_slot is None:
        pooled_slot = _claim_ready_slot(container_slot if isinstance(container_slot, int) else None)
    if pooled_slot is None:
        pooled_slot = _claim_preinit_failed_slot(container_slot if isinstance(container_slot, int) else None)
    if pooled_slot is None:
        pooled_slot = _wait_for_pool_slot(container_slot if isinstance(container_slot, int) else None)
    if pooled_slot is not None:
        _queue_status(f"Using pooled sandbox slot {pooled_slot.slot_id}")

    _queue_status(f"Sandbox queued (template={template_name})")
    logger.info("[E2B Sandbox] status queued template=%s", template_name)

    def _handle_stream_data(data: Any, origin: str) -> None:
        if not stream_sandbox_logs:
            raw = "" if data is None else str(data)
            if _PAYLOAD_PREFIX not in raw:
                return
        if not getattr(_handle_stream_data, "_first_seen", False):
            _handle_stream_data._first_seen = True  # type: ignore[attr-defined]
            logger.info("[E2B Sandbox] first sandbox log received origin=%s", origin)
        for chunk in _iter_log_chunks(data):
            for payload in _process_stream_chunk(chunk, origin, stream_sandbox_logs):
                _queue_payload(payload)

    async def _run_async() -> None:
        controller = get_active_run_controller()
        active_count = _increment_active_executions()
        parallel_count = _get_parallel_executions()
        if active_count >= parallel_count:
            _cleanup_unused_pool("all_branches_active")
        sandbox = None
        cancel_handle: Optional[str] = None
        if pooled_slot and pooled_slot.sandbox is not None:
            sandbox = pooled_slot.sandbox
            logger.info("[E2B Sandbox] using prewarmed sandbox slot=%s", pooled_slot.slot_id)
            if controller is not None and pooled_slot.cancel_handle is None:
                pooled_slot.cancel_handle = controller.register_e2b_sandbox(
                    sandbox,
                    label=f"e2b_sandbox_pooled_{pooled_slot.slot_id}",
                    kill_callback=lambda: _kill_sandbox_sync(sandbox),
                )
                cancel_handle = pooled_slot.cancel_handle
            else:
                cancel_handle = pooled_slot.cancel_handle
        else:
            if sandbox is None:
                logger.info("[E2B Sandbox] creating sandbox")
            metadata = {
                "slot_id": str(container_slot) if container_slot is not None else "",
                "branch_id": str(container_slot) if container_slot is not None else "",
                "purpose": "mgs_on_demand",
            }
            if sandbox is None:
                sandbox = await AsyncSandbox.create(
                    template=template_name,
                    timeout=500,
                    envs=envs,
                    request_timeout=500,
                    metadata=metadata,
                )
                if controller is not None:
                    cancel_handle = controller.register_e2b_sandbox(
                        sandbox,
                        label="e2b_sandbox",
                        kill_callback=lambda: _kill_sandbox_sync(sandbox),
                    )
        try:
            logger.info("[E2B Sandbox] preparing sandbox files")
            if google_creds_text:
                await sandbox.files.write("/home/user/google_credentials.json", google_creds_text)
            await sandbox.files.write("/home/user/run_input.json", json.dumps(run_input))
            await sandbox.files.write("/home/user/llm_code.py", code)
            logger.info("[E2B Sandbox] files written, launching runner")
            runner_code = _build_runner_code()
            logger.info("[E2B Sandbox] runner_code built, starting run_code")
            await sandbox.run_code(
                runner_code,
                on_stdout=lambda data: _handle_stream_data(data, "stdout"),
                on_stderr=lambda data: _handle_stream_data(data, "stderr"),
                on_error=lambda error: _queue_payload(_make_step_payload(str(error), "error", origin="stderr")),
            )
            logger.info("[E2B Sandbox] run_code completed")
        finally:
            try:
                if sandbox is not None:
                    await sandbox.kill()
            except Exception:
                pass
            if pooled_slot is not None:
                _remove_pool_slot(pooled_slot.slot_id)
            if controller is not None and cancel_handle:
                controller.unregister(cancel_handle)
            remaining = _decrement_active_executions()
            if remaining >= _get_parallel_executions():
                _cleanup_unused_pool("all_branches_active")

    future = _ASYNC_LOOP_RUNNER.submit(_run_async())

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
