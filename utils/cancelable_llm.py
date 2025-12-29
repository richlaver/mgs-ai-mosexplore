from __future__ import annotations

from typing import Any, Callable, Optional

from langchain_google_vertexai import ChatVertexAI

from utils.run_cancellation import RunCancelledError, RunCancellationController, get_active_run_controller


class InterruptibleChatVertexAI(ChatVertexAI):
    """ChatVertexAI subclass that checks the active cancellation controller."""

    def __init__(
        self,
        *,
        controller_getter: Optional[Callable[[], Optional[RunCancellationController]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._controller_getter = controller_getter or get_active_run_controller

    def _get_controller(self) -> Optional[RunCancellationController]:
        return self._controller_getter() if self._controller_getter else None

    def _guard(self, stage: str) -> None:
        controller = self._get_controller()
        if controller:
            controller.raise_if_cancelled(stage)

    def _generate(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self._guard("llm:generate")
        return super()._generate(*args, **kwargs)

    def _stream(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        controller = self._get_controller()
        for chunk in super()._stream(*args, **kwargs):
            if controller and controller.is_cancelled():
                raise RunCancelledError("LLM stream cancelled")
            yield chunk

    async def _agenerate(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self._guard("llm:agenerate")
        return await super()._agenerate(*args, **kwargs)

    async def _astream(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        controller = self._get_controller()
        async for chunk in super()._astream(*args, **kwargs):
            if controller and controller.is_cancelled():
                raise RunCancelledError("LLM stream cancelled")
            yield chunk


def clone_llm(llm: ChatVertexAI, **overrides: Any) -> ChatVertexAI:
    """Clone a ChatVertexAI preserving interruptible behavior."""

    base_params = {
        "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
        "temperature": getattr(llm, "temperature", None),
        "max_output_tokens": getattr(llm, "max_output_tokens", None),
        "top_p": getattr(llm, "top_p", None),
        "top_k": getattr(llm, "top_k", None),
        "safety_settings": getattr(llm, "safety_settings", None),
        "location": getattr(llm, "location", None),
        "project": getattr(llm, "project", None),
    }
    base_params.update({k: v for k, v in overrides.items() if v is not None})
    filtered = {k: v for k, v in base_params.items() if v is not None}
    if "model" not in filtered:
        raise ValueError("Cannot clone ChatVertexAI without a model name")
    target_cls = llm.__class__ if isinstance(llm, ChatVertexAI) else ChatVertexAI
    return target_cls(**filtered)
