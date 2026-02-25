from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import Future
import threading
from typing import Awaitable, Optional, TypeVar

T = TypeVar("T")

_loop_lock = threading.Lock()
_ready = threading.Event()
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_background_thread: Optional[threading.Thread] = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _background_loop, _background_thread

    with _loop_lock:
        if _background_loop is not None and _background_thread is not None and _background_thread.is_alive():
            return _background_loop

        _ready.clear()

        def _runner() -> None:
            global _background_loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _background_loop = loop
            _ready.set()
            loop.run_forever()

        _background_thread = threading.Thread(target=_runner, name="mgs-async-utils-loop", daemon=True)
        _background_thread.start()

    _ready.wait(timeout=5)
    if _background_loop is None:
        raise RuntimeError("Failed to start background asyncio loop")
    return _background_loop


def run_async_syncsafe(coro: Awaitable[T]) -> T:
    """Run a coroutine from synchronous code safely.

    Runs on a shared background event loop to avoid cross-loop issues with
    async clients that keep loop-bound primitives (e.g. asyncio.Lock).
    Captures caller contextvars and recreates the task under that context.
    """

    loop = _ensure_background_loop()
    caller_ctx = contextvars.copy_context()
    result_future: Future[T] = Future()

    def _submit() -> None:
        try:
            task = caller_ctx.run(asyncio.create_task, coro)
        except BaseException as exc:  # pragma: no cover - defensive
            if not result_future.done():
                result_future.set_exception(exc)
            return

        def _done_callback(done_task: asyncio.Task[T]) -> None:
            try:
                result = done_task.result()
            except BaseException as exc:  # pragma: no cover - defensive
                if not result_future.done():
                    result_future.set_exception(exc)
            else:
                if not result_future.done():
                    result_future.set_result(result)

        task.add_done_callback(_done_callback)

    loop.call_soon_threadsafe(_submit)
    return result_future.result()
