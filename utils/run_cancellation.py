"""Centralized run cancellation controller and helpers.

This module provides a single place to coordinate cancellation across
LLM calls, SQL queries, Modal sandbox executions, and artefact writes.
The controller can be activated via a contextvar so any downstream code
can introspect the currently active run and short-circuit work when a
user presses the stop button.
"""

from __future__ import annotations

import contextlib
import contextvars
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
    def register_modal_call(self, call_obj: Any, *, label: str = "modal") -> str:
        """Register a Modal FunctionCall-like object that exposes cancel()."""

        def _cancel() -> None:
            try:
                call_obj.cancel()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to cancel Modal call %s: %s", label, exc)

        return self._register_callback(_cancel, label=f"{label}:modal")

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

    def _register_callback(self, callback: CancelCallback, *, label: str) -> str:
        handle = uuid.uuid4().hex
        with self._lock:
            self._resources[handle] = _RegisteredResource(label=label, cancel=callback)
            trigger_now = self._cancelled.is_set()
        if trigger_now:
            self._invoke_callback(handle, callback, label)
        return handle

    def unregister(self, handle: str) -> None:
        with self._lock:
            self._resources.pop(handle, None)

    # ------------------------------------------------------------------
    # Cancellation lifecycle
    # ------------------------------------------------------------------
    def cancel(self, reason: str | None = None) -> None:
        """Trigger cancellation for all registered resources (idempotent)."""

        if reason:
            logger.info("[RunCancel %s] Cancelling run: %s", self.run_id, reason)
        handles, resources = self._snapshot_callbacks()
        self._cancelled.set()
        for handle, resource in zip(handles, resources, strict=False):
            self._invoke_callback(handle, resource.cancel, resource.label)

    def _snapshot_callbacks(self) -> tuple[list[str], list[_RegisteredResource]]:
        with self._lock:
            handles = list(self._resources.keys())
            callbacks = list(self._resources.values())
        return handles, callbacks

    def _invoke_callback(self, handle: str, callback: CancelCallback, label: str) -> None:
        try:
            callback()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[RunCancel %s] Resource callback '%s' failed during cancel: %s",
                self.run_id,
                label,
                exc,
            )
        finally:
            with self._lock:
                self._resources.pop(handle, None)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self, where: str | None = None) -> None:
        if self.is_cancelled():
            suffix = f" ({where})" if where else ""
            raise RunCancelledError(f"Run cancelled{suffix}")


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
    "activate_controller",
    "reset_controller",
    "get_active_run_controller",
]
