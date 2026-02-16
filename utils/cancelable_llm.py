from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from utils.run_cancellation import RunCancelledError, RunCancellationController, get_active_run_controller

logger = logging.getLogger(__name__)


class InterruptibleChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI subclass that checks the active cancellation controller."""

    def __init__(
        self,
        *,
        controller_getter: Optional[Callable[[], Optional[RunCancellationController]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._controller_getter = controller_getter or get_active_run_controller

    def _start_cancel_watch(self, stage: str):
        controller = self._get_controller()
        cancel_event = threading.Event()
        handle: Optional[str] = None

        if controller:
            label = f"llm:{getattr(self, 'model', 'unknown')}"

            def _cancel() -> None:
                logger.warning(
                    "[LLMCancel] Cancel requested for %s at stage=%s (run_id=%s)",
                    label,
                    stage,
                    getattr(controller, "run_id", "?"),
                )
                cancel_event.set()

            handle = controller.register_generic(_cancel, label=label)
            logger.info(
                "[LLMCancel] Registered cancellable LLM call | handle=%s stage=%s model=%s",
                handle,
                stage,
                getattr(self, "model", "unknown"),
            )

        return controller, cancel_event, handle

    @staticmethod
    def _stop_cancel_watch(controller: Optional[RunCancellationController], handle: Optional[str]) -> None:
        if controller and handle:
            controller.unregister(handle)

    @staticmethod
    def _raise_if_cancelled(controller: Optional[RunCancellationController], cancel_event: threading.Event, stage: str) -> None:
        if cancel_event.is_set():
            raise RunCancelledError(f"LLM call cancelled ({stage})")
        if controller:
            controller.raise_if_cancelled(stage)

    def _get_controller(self) -> Optional[RunCancellationController]:
        return self._controller_getter() if self._controller_getter else None

    def _guard(self, stage: str) -> None:
        controller = self._get_controller()
        if controller:
            controller.raise_if_cancelled(stage)

    def _generate(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        controller, cancel_event, handle = self._start_cancel_watch("llm:generate")
        try:
            self._raise_if_cancelled(controller, cancel_event, "llm:generate:pre")
            result = super()._generate(*args, **kwargs)
            self._raise_if_cancelled(controller, cancel_event, "llm:generate:post")
            return result
        finally:
            self._stop_cancel_watch(controller, handle)

    def _stream(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        controller, cancel_event, handle = self._start_cancel_watch("llm:stream")
        try:
            for chunk in super()._stream(*args, **kwargs):
                self._raise_if_cancelled(controller, cancel_event, "llm:stream")
                yield chunk
        finally:
            self._stop_cancel_watch(controller, handle)

    async def _agenerate(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        controller, cancel_event, handle = self._start_cancel_watch("llm:agenerate")
        try:
            self._raise_if_cancelled(controller, cancel_event, "llm:agenerate:pre")
            result = await super()._agenerate(*args, **kwargs)
            self._raise_if_cancelled(controller, cancel_event, "llm:agenerate:post")
            return result
        finally:
            self._stop_cancel_watch(controller, handle)

    async def _astream(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        controller, cancel_event, handle = self._start_cancel_watch("llm:astream")
        try:
            async for chunk in super()._astream(*args, **kwargs):
                self._raise_if_cancelled(controller, cancel_event, "llm:astream")
                yield chunk
        finally:
            self._stop_cancel_watch(controller, handle)


def clone_llm(llm: ChatGoogleGenerativeAI, **overrides: Any) -> ChatGoogleGenerativeAI:
    """Clone a ChatGoogleGenerativeAI preserving interruptible behavior."""

    base_params = {
        "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
        "temperature": getattr(llm, "temperature", None),
        "max_tokens": getattr(llm, "max_tokens", None),
        "max_output_tokens": getattr(llm, "max_output_tokens", None),
        "top_p": getattr(llm, "top_p", None),
        "top_k": getattr(llm, "top_k", None),
        "safety_settings": getattr(llm, "safety_settings", None),
        "location": getattr(llm, "location", None),
        "project": getattr(llm, "project", None),
        "vertexai": getattr(llm, "vertexai", None),
        "api_key": getattr(llm, "google_api_key", None) or getattr(llm, "api_key", None),
        "thinking_budget": getattr(llm, "thinking_budget", None),
    }
    base_params.update({k: v for k, v in overrides.items() if v is not None})
    if "max_output_tokens" not in base_params and base_params.get("max_tokens") is not None:
        base_params["max_output_tokens"] = base_params["max_tokens"]
    base_params.pop("max_tokens", None)
    filtered = {k: v for k, v in base_params.items() if v is not None}
    if "model" not in filtered:
        raise ValueError("Cannot clone ChatGoogleGenerativeAI without a model name")
    target_cls = llm.__class__ if isinstance(llm, ChatGoogleGenerativeAI) else ChatGoogleGenerativeAI
    return target_cls(**filtered)
