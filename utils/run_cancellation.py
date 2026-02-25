"""Centralized run cancellation controller and helpers.

This module provides a single place to coordinate cancellation across
LLM calls, SQL queries, E2B sandbox executions, and artefact writes.
The controller can be activated via a contextvar so any downstream code
can introspect the currently active run and short-circuit work when a
user presses the stop button.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, Optional

logger = logging.getLogger(__name__)


class RunCancelledError(RuntimeError):
    """Raised when work is attempted after a run has been cancelled."""

    def __init__(self, message: str = "Run cancelled by user") -> None:
        super().__init__(message)


CancelCallback = Callable[[], None]


@dataclass(slots=True)
class _RegisteredResource:
    label: str
    cancel: CancelCallback


class RunCancellationController:
    """Tracks resources tied to a single user query and cancels them on demand."""

    def __init__(self, *, run_id: str, user_id: int) -> None:
        self.run_id = run_id
        self.user_id = user_id
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._resources: Dict[str, _RegisteredResource] = {}

    # ---------------------------------------------------------------------
    # Resource registration helpers
    # ------------------------------------------------------------------
    def register_e2b_sandbox(
        self,
        sandbox: Any,
        *,
        label: str = "e2b_sandbox",
        kill_callback: Optional[CancelCallback] = None,
    ) -> str:
        """Register an E2B sandbox-like object for termination on cancel."""

        def _cancel() -> None:
            try:
                if kill_callback is not None:
                    logger.info("[RunCancel %s] Invoking E2B kill callback label=%s", self.run_id, label)
                    kill_callback()
                    return
                kill = getattr(sandbox, "kill", None)
                if kill is None:
                    logger.warning("[RunCancel %s] E2B sandbox missing kill() label=%s", self.run_id, label)
                    return
                result = kill()
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        logger.info("[RunCancel %s] Running E2B async kill label=%s", self.run_id, label)
                        asyncio.run(result)
                    else:
                        loop.create_task(result)
                        logger.info("[RunCancel %s] Scheduled E2B async kill label=%s", self.run_id, label)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[RunCancel %s] Failed to cancel E2B sandbox %s: %s", self.run_id, label, exc)

        return self._register_callback(_cancel, label=f"{label}:e2b")

    def register_sql_connection(self, connection: Any, *, label: str = "sql") -> str:
        """Register a SQLAlchemy connection for forced closure on cancel."""

        def _cancel() -> None:
            try:
                raw = getattr(connection, "connection", None)
                if raw is not None:
                    try:
                        raw.close()
                    except Exception:  # pragma: no cover - best effort
                        pass
                connection.invalidate()
            except Exception:  # pragma: no cover - best effort
                with contextlib.suppress(Exception):
                    connection.close()

        return self._register_callback(_cancel, label=f"{label}:sql")

    def register_generic(self, callback: CancelCallback, *, label: str) -> str:
        """Register an arbitrary callback executed when cancellation happens."""

        return self._register_callback(callback, label=label)

    def register_asyncio_task(
        self,
        task: asyncio.Task[Any],
        *,
        label: str = "asyncio_task",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> str:
        """Register an asyncio task for cancellation on controller cancel.

        This supports thread-safe task cancellation when cancel() is called from a
        different thread than the task's event loop.
        """

        task_loop = loop
        if task_loop is None:
            with contextlib.suppress(Exception):
                task_loop = task.get_loop()

        logger.info(
            "[RunCancel %s] Registering asyncio task label=%s task_id=%s loop_id=%s done=%s",
            self.run_id,
            label,
            id(task),
            id(task_loop) if task_loop else None,
            task.done(),
        )

        def _cancel() -> None:
            if task.done():
                logger.info(
                    "[RunCancel %s] Asyncio task already done label=%s task_id=%s",
                    self.run_id,
                    label,
                    id(task),
                )
                return

            if task_loop and task_loop.is_running():
                try:
                    logger.info(
                        "[RunCancel %s] Scheduling asyncio task cancel label=%s task_id=%s loop_id=%s",
                        self.run_id,
                        label,
                        id(task),
                        id(task_loop),
                    )
                    task_loop.call_soon_threadsafe(task.cancel)
                    return
                except Exception:
                    logger.warning(
                        "[RunCancel %s] Failed thread-safe task cancel scheduling label=%s task_id=%s",
                        self.run_id,
                        label,
                        id(task),
                        exc_info=True,
                    )
                    pass

            logger.info(
                "[RunCancel %s] Direct asyncio task cancel fallback label=%s task_id=%s",
                self.run_id,
                label,
                id(task),
            )
            with contextlib.suppress(Exception):
                task.cancel()

        return self._register_callback(_cancel, label=f"{label}:asyncio")

    def _register_callback(self, callback: CancelCallback, *, label: str) -> str:
        handle = uuid.uuid4().hex
        with self._lock:
            self._resources[handle] = _RegisteredResource(label=label, cancel=callback)
            trigger_now = self._cancelled.is_set()
            resource_count = len(self._resources)
        logger.info(
            "[RunCancel %s] Registered resource handle=%s label=%s trigger_now=%s total=%d",
            self.run_id,
            handle,
            label,
            trigger_now,
            resource_count,
        )
        if trigger_now:
            self._invoke_callback(handle, callback, label)
        return handle

    def unregister(self, handle: str) -> None:
        with self._lock:
            removed = self._resources.pop(handle, None)
            remaining = len(self._resources)
        logger.info(
            "[RunCancel %s] Unregistered resource handle=%s removed=%s remaining=%d",
            self.run_id,
            handle,
            bool(removed),
            remaining,
        )

    # ------------------------------------------------------------------
    # Cancellation lifecycle
    # ------------------------------------------------------------------
    def cancel(self, reason: str | None = None) -> None:
        """Trigger cancellation for all registered resources (idempotent)."""

        if reason:
            logger.info("[RunCancel %s] Cancelling run: %s", self.run_id, reason)
        else:
            logger.info("[RunCancel %s] Cancelling run", self.run_id)
        handles, resources = self._snapshot_callbacks()
        logger.info(
            "[RunCancel %s] Snapshot for cancel handles=%d",
            self.run_id,
            len(handles),
        )
        self._cancelled.set()
        for handle, resource in zip(handles, resources, strict=False):
            logger.info(
                "[RunCancel %s] Cancelling resource handle=%s label=%s",
                self.run_id,
                handle,
                resource.label,
            )
            self._invoke_callback(handle, resource.cancel, resource.label)

    def cancel_active_resources(self, reason: str | None = None) -> None:
        """Best-effort cancellation of currently registered resources without marking the run cancelled.

        This is useful when we want to stop in-flight work (e.g., LLM calls, SQL queries,
        sandbox executions) while allowing the overall graph to continue to downstream nodes.
        """

        handles, resources = self._snapshot_callbacks()
        if reason:
            logger.info("[RunCancel %s] Cancelling active resources only: %s (handles=%d)", self.run_id, reason, len(handles))
        if not handles:
            logger.info("[RunCancel %s] No active resources to cancel", self.run_id)
        for handle, resource in zip(handles, resources, strict=False):
            logger.info("[RunCancel %s] Cancelling resource handle=%s label=%s", self.run_id, handle, resource.label)
            self._invoke_callback(handle, resource.cancel, resource.label)

    def _snapshot_callbacks(self) -> tuple[list[str], list[_RegisteredResource]]:
        with self._lock:
            handles = list(self._resources.keys())
            callbacks = list(self._resources.values())
            logger.debug(
                "[RunCancel %s] Snapshot handles=%d labels=%s",
                self.run_id,
                len(handles),
                [resource.label for resource in callbacks],
            )
        return handles, callbacks

    def _invoke_callback(self, handle: str, callback: CancelCallback, label: str) -> None:
        logger.info(
            "[RunCancel %s] Invoking cancel callback handle=%s label=%s",
            self.run_id,
            handle,
            label,
        )
        try:
            callback()
            logger.info(
                "[RunCancel %s] Cancel callback completed handle=%s label=%s",
                self.run_id,
                handle,
                label,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[RunCancel %s] Resource callback '%s' failed during cancel: %s",
                self.run_id,
                label,
                exc,
            )
        finally:
            with self._lock:
                removed = self._resources.pop(handle, None)
                remaining = len(self._resources)
            logger.info(
                "[RunCancel %s] Cancel cleanup handle=%s removed=%s remaining=%d",
                self.run_id,
                handle,
                bool(removed),
                remaining,
            )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def is_cancelled(self) -> bool:
        cancelled = self._cancelled.is_set()
        #logger.debug("[RunCancel %s] is_cancelled=%s", self.run_id, cancelled)
        return cancelled

    def raise_if_cancelled(self, where: str | None = None) -> None:
        if self.is_cancelled():
            suffix = f" ({where})" if where else ""
            logger.info("[RunCancel %s] raise_if_cancelled triggered%s", self.run_id, suffix)
            raise RunCancelledError(f"Run cancelled{suffix}")


class ScopedRunCancellationController:
    """Scoped cancellation controller for a subset of resources.

    Uses a parent controller for run-level cancellation checks while
    tracking and cancelling only the resources registered within this scope.
    """

    def __init__(self, *, parent: Optional[RunCancellationController], label: str) -> None:
        self._parent = parent
        self._label = label
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._resources: Dict[str, _RegisteredResource] = {}
        self._parent_handle: Optional[str] = None
        if parent:
            self._parent_handle = parent.register_generic(
                lambda: self.cancel_active_resources(reason=f"parent_cancel:{label}"),
                label=f"scope:{label}",
            )

    def register_generic(self, callback: CancelCallback, *, label: str) -> str:
        handle = uuid.uuid4().hex
        with self._lock:
            self._resources[handle] = _RegisteredResource(label=label, cancel=callback)
            trigger_now = self._cancelled.is_set()
        logger.info(
            "[RunCancelScope %s] Registered resource handle=%s label=%s trigger_now=%s",
            self._label,
            handle,
            label,
            trigger_now,
        )
        if trigger_now:
            self._invoke_callback(handle, callback, label)
        return handle

    def register_asyncio_task(
        self,
        task: asyncio.Task[Any],
        *,
        label: str = "asyncio_task",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> str:
        """Register an asyncio task for cancellation within this scope."""

        task_loop = loop
        if task_loop is None:
            with contextlib.suppress(Exception):
                task_loop = task.get_loop()

        logger.info(
            "[RunCancelScope %s] Registering asyncio task label=%s task_id=%s loop_id=%s done=%s",
            self._label,
            label,
            id(task),
            id(task_loop) if task_loop else None,
            task.done(),
        )

        def _cancel() -> None:
            if task.done():
                logger.info(
                    "[RunCancelScope %s] Asyncio task already done label=%s task_id=%s",
                    self._label,
                    label,
                    id(task),
                )
                return

            if task_loop and task_loop.is_running():
                try:
                    logger.info(
                        "[RunCancelScope %s] Scheduling asyncio task cancel label=%s task_id=%s loop_id=%s",
                        self._label,
                        label,
                        id(task),
                        id(task_loop),
                    )
                    task_loop.call_soon_threadsafe(task.cancel)
                    return
                except Exception:
                    logger.warning(
                        "[RunCancelScope %s] Failed thread-safe task cancel scheduling label=%s task_id=%s",
                        self._label,
                        label,
                        id(task),
                        exc_info=True,
                    )

            logger.info(
                "[RunCancelScope %s] Direct asyncio task cancel fallback label=%s task_id=%s",
                self._label,
                label,
                id(task),
            )
            with contextlib.suppress(Exception):
                task.cancel()

        return self.register_generic(_cancel, label=f"{label}:asyncio")

    def unregister(self, handle: str) -> None:
        with self._lock:
            removed = self._resources.pop(handle, None)
            remaining = len(self._resources)
        logger.info(
            "[RunCancelScope %s] Unregistered resource handle=%s removed=%s remaining=%d",
            self._label,
            handle,
            bool(removed),
            remaining,
        )

    def cancel_active_resources(self, reason: str | None = None) -> None:
        handles, resources = self._snapshot_callbacks()
        if reason:
            logger.info(
                "[RunCancelScope %s] Cancelling active resources: %s (handles=%d)",
                self._label,
                reason,
                len(handles),
            )
        if not handles:
            logger.info("[RunCancelScope %s] No active resources to cancel", self._label)
        self._cancelled.set()
        for handle, resource in zip(handles, resources, strict=False):
            self._invoke_callback(handle, resource.cancel, resource.label)

    def cancel(self, reason: str | None = None) -> None:
        if reason:
            logger.info("[RunCancelScope %s] Cancelling scope: %s", self._label, reason)
        else:
            logger.info("[RunCancelScope %s] Cancelling scope", self._label)
        self.cancel_active_resources(reason=reason)

    def is_cancelled(self) -> bool:
        if self._cancelled.is_set():
            return True
        if self._parent:
            return self._parent.is_cancelled()
        return False

    def raise_if_cancelled(self, where: str | None = None) -> None:
        if self._cancelled.is_set():
            suffix = f" ({where})" if where else ""
            raise RunCancelledError(f"Scoped cancellation{suffix}")
        if self._parent:
            self._parent.raise_if_cancelled(where)

    def _snapshot_callbacks(self) -> tuple[list[str], list[_RegisteredResource]]:
        with self._lock:
            handles = list(self._resources.keys())
            callbacks = list(self._resources.values())
        return handles, callbacks

    def _invoke_callback(self, handle: str, callback: CancelCallback, label: str) -> None:
        logger.info(
            "[RunCancelScope %s] Invoking cancel callback handle=%s label=%s",
            self._label,
            handle,
            label,
        )
        try:
            callback()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[RunCancelScope %s] Resource callback '%s' failed during cancel: %s",
                self._label,
                label,
                exc,
            )
        finally:
            with self._lock:
                self._resources.pop(handle, None)


_current_controller: contextvars.ContextVar[Optional[RunCancellationController]] = contextvars.ContextVar(
    "current_run_cancellation_controller",
    default=None,
)


def activate_controller(controller: RunCancellationController) -> contextvars.Token[Optional[RunCancellationController]]:
    """Activate a controller for downstream code running in the same context."""

    return _current_controller.set(controller)


def reset_controller(token: contextvars.Token[Optional[RunCancellationController]]) -> None:
    """Reset the controller context to a previous token."""

    _current_controller.reset(token)


def get_active_run_controller() -> Optional[RunCancellationController]:
    """Return the controller active in the current context, if any."""

    return _current_controller.get()


__all__ = [
    "RunCancelledError",
    "RunCancellationController",
    "ScopedRunCancellationController",
    "activate_controller",
    "reset_controller",
    "get_active_run_controller",
]
