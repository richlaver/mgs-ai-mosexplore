import logging
import os
import threading
from typing import Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

from agents.instrument_id_validator import instrument_id_validator
from agents.instrument_type_validator import instrument_type_validator
from agents.database_expert import database_expert
from agents.period_expert import period_expert
from agents.project_insider import project_insider
from agents.platform_expert import platform_expert
from classes import ContextState
from utils.run_cancellation import (
    ScopedRunCancellationController,
    activate_controller,
    get_active_run_controller,
    reset_controller,
)

logger = logging.getLogger(__name__)

def _get_parallel_setting(env_key: str, default: int) -> int:
    raw = os.environ.get(env_key)
    try:
        value = int(raw) if raw is not None else default
    except Exception:
        value = default
    return max(1, value)


DB_EXPERT_PARALLEL_EXECUTIONS = _get_parallel_setting("MGS_DB_EXPERT_PARALLEL_EXECUTIONS", 1)
PERIOD_EXPERT_PARALLEL_EXECUTIONS = _get_parallel_setting("MGS_PERIOD_EXPERT_PARALLEL_EXECUTIONS", 1)
INSTRUMENT_TYPE_VALIDATOR_PARALLEL_EXECUTIONS = _get_parallel_setting("MGS_INSTRUMENT_TYPE_VALIDATOR_PARALLEL_EXECUTIONS", 1)


def _has_clarification_requests(state: ContextState) -> bool:
    try:
        if isinstance(state.context, dict):
            clar = state.context.get("clarification_requests") or []
            return bool(clar)
        clar = getattr(state.context, "clarification_requests", []) or []
        return bool(clar)
    except Exception:
        return False

def get_context_graph(llms: Dict[str, BaseLanguageModel], db: any, selected_project_key: str | None) -> any:
    instrument_type_scope_lock = threading.Lock()
    instrument_type_scope: Optional[ScopedRunCancellationController] = None
    instrument_type_winner_lock = threading.Lock()
    instrument_type_winner_set = False

    db_scope_lock = threading.Lock()
    db_scope: Optional[ScopedRunCancellationController] = None
    db_winner_lock = threading.Lock()
    db_winner_set = False

    period_scope_lock = threading.Lock()
    period_scope: Optional[ScopedRunCancellationController] = None
    period_winner_lock = threading.Lock()
    period_winner_set = False

    def _get_instrument_type_scope() -> ScopedRunCancellationController:
        nonlocal instrument_type_scope
        with instrument_type_scope_lock:
            if instrument_type_scope is None:
                instrument_type_scope = ScopedRunCancellationController(
                    parent=get_active_run_controller(),
                    label="context_instrument_type_validator",
                )
        return instrument_type_scope

    def _get_db_scope() -> ScopedRunCancellationController:
        nonlocal db_scope
        with db_scope_lock:
            if db_scope is None:
                db_scope = ScopedRunCancellationController(
                    parent=get_active_run_controller(),
                    label="context_db_expert",
                )
        return db_scope

    def _get_period_scope() -> ScopedRunCancellationController:
        nonlocal period_scope
        with period_scope_lock:
            if period_scope is None:
                period_scope = ScopedRunCancellationController(
                    parent=get_active_run_controller(),
                    label="context_period_expert",
                )
        return period_scope

    def _run_with_scope(scope: ScopedRunCancellationController, fn):
        token = activate_controller(scope)
        try:
            return fn()
        finally:
            reset_controller(token)

    def instrument_id_validator_node(state: ContextState):
        logger.info("Context node enter: instrument_id_validator")
        if _has_clarification_requests(state):
            logger.info("instrument_id_validator skipped due to clarification requests")
            return {}
        result = instrument_id_validator(state, llms["FAST"], db)
        logger.info("Context node exit: instrument_id_validator")
        return result

    def instrument_type_validator_node(state: ContextState, branch_id: int):
        nonlocal instrument_type_winner_set
        logger.info("Context node enter: instrument_type_validator branch=%d", branch_id)
        if _has_clarification_requests(state):
            logger.info("instrument_type_validator skipped due to clarification requests")
            return {}

        with instrument_type_winner_lock:
            if instrument_type_winner_set:
                logger.info("instrument_type_validator branch skipped (winner already selected)")
                return {}

        scope = _get_instrument_type_scope()
        result = _run_with_scope(
            scope,
            lambda: instrument_type_validator(state, selected_project_key),
        )

        winner = False
        with instrument_type_winner_lock:
            if not instrument_type_winner_set:
                instrument_type_winner_set = True
                winner = True

        if winner:
            scope.cancel_active_resources(reason="instrument_type_validator_branch_complete")
            logger.info("instrument_type_validator branch selected as winner")
            logger.info("Context node exit: instrument_type_validator branch=%d", branch_id)
            return result

        logger.info("instrument_type_validator branch completed after winner selected")
        logger.info("Context node exit: instrument_type_validator branch=%d", branch_id)
        return {}

    def database_expert_node(state: ContextState):
        nonlocal db_winner_set
        logger.info("Context node enter: database_expert")
        if _has_clarification_requests(state):
            logger.info("database_expert skipped due to clarification requests")
            return {}

        with db_winner_lock:
            if db_winner_set:
                logger.info("database_expert branch skipped (winner already selected)")
                return {}

        scope = _get_db_scope()
        result = _run_with_scope(
            scope,
            lambda: database_expert(state, llms["LONG"], db, selected_project_key),
        )

        winner = False
        with db_winner_lock:
            if not db_winner_set:
                db_winner_set = True
                winner = True

        if winner:
            scope.cancel_active_resources(reason="database_expert_branch_complete")
            logger.info("database_expert branch selected as winner")
            logger.info("Context node exit: database_expert")
            return result

        logger.info("database_expert branch completed after winner selected")
        logger.info("Context node exit: database_expert")
        return {}

    def period_expert_node(state: ContextState, branch_id: int):
        nonlocal period_winner_set
        logger.info("Context node enter: period_expert branch=%d", branch_id)
        if _has_clarification_requests(state):
            logger.info("period_expert skipped due to clarification requests")
            return {}

        with period_winner_lock:
            if period_winner_set:
                logger.info("period_expert branch skipped (winner already selected)")
                return {}

        scope = _get_period_scope()
        result = _run_with_scope(
            scope,
            lambda: period_expert(state, llms["FAST"], db),
        )

        winner = False
        with period_winner_lock:
            if not period_winner_set:
                period_winner_set = True
                winner = True

        if winner:
            scope.cancel_active_resources(reason="period_expert_branch_complete")
            logger.info("period_expert branch selected as winner")
            logger.info("Context node exit: period_expert branch=%d", branch_id)
            return result

        logger.info("period_expert branch completed after winner selected")
        logger.info("Context node exit: period_expert branch=%d", branch_id)
        return {}

    def platform_expert_node(state: ContextState):
        logger.info("Context node enter: platform_expert")
        if _has_clarification_requests(state):
            logger.info("platform_expert skipped due to clarification requests")
            return {}
        result = platform_expert(state)
        logger.info("Context node exit: platform_expert")
        return result

    def project_insider_node(state: ContextState):
        logger.info("Context node enter: project_insider")
        if _has_clarification_requests(state):
            logger.info("project_insider skipped due to clarification requests")
            return {}
        result = project_insider(state, selected_project_key)
        logger.info("Context node exit: project_insider")
        return result

    def instrument_db_fan_out(state: ContextState):
        logger.info("Context subgraph fan-out: instrument validators")
        if _has_clarification_requests(state):
            logger.info("instrument_db_fan_out aborted due to clarification requests")
            return Command(goto=END)
        return Command(goto=[
            Send("instrument_id_validator", state),
            Send("instrument_type_validator_fan_out", state),
        ])

    def instrument_db_fan_in(_: ContextState):
        logger.info("Context subgraph fan-in: instrument validators")
        return {}

    def instrument_type_validator_fan_out(state: ContextState):
        logger.info(
            "Context subgraph fan-out: instrument_type_validator (%d branches)",
            INSTRUMENT_TYPE_VALIDATOR_PARALLEL_EXECUTIONS,
        )
        if _has_clarification_requests(state):
            logger.info("instrument_type_validator fan-out aborted due to clarification requests")
            return Command(goto="instrument_type_validator_fan_in")
        return Command(goto=[
            Send(f"instrument_type_validator_{i}", state)
            for i in range(INSTRUMENT_TYPE_VALIDATOR_PARALLEL_EXECUTIONS)
        ])

    def instrument_type_validator_fan_in(_: ContextState):
        logger.info("Context subgraph fan-in: instrument_type_validator")
        return {}

    def database_expert_fan_out(state: ContextState):
        logger.info("Context subgraph fan-out: database_expert (%d branches)", DB_EXPERT_PARALLEL_EXECUTIONS)
        if _has_clarification_requests(state):
            logger.info("database_expert fan-out aborted due to clarification requests")
            return Command(goto="database_expert_fan_in")
        return Command(goto=[
            Send(f"database_expert_{i}", state)
            for i in range(DB_EXPERT_PARALLEL_EXECUTIONS)
        ])

    def database_expert_fan_in(_: ContextState):
        logger.info("Context subgraph fan-in: database_expert")
        return {}

    instrument_db_graph = StateGraph(ContextState)
    instrument_db_graph.add_node("instrument_db_fan_out", instrument_db_fan_out)
    instrument_db_graph.add_node("instrument_id_validator", instrument_id_validator_node)
    instrument_db_graph.add_node("instrument_type_validator_fan_out", instrument_type_validator_fan_out)
    instrument_db_graph.add_node("instrument_type_validator_fan_in", instrument_type_validator_fan_in)
    for i in range(INSTRUMENT_TYPE_VALIDATOR_PARALLEL_EXECUTIONS):
        instrument_db_graph.add_node(
            f"instrument_type_validator_{i}",
            lambda state, branch_id=i: instrument_type_validator_node(state, branch_id),
        )
        instrument_db_graph.add_edge(
            f"instrument_type_validator_{i}",
            "instrument_type_validator_fan_in",
        )
    instrument_db_graph.add_node("instrument_db_fan_in", instrument_db_fan_in)
    instrument_db_graph.add_node("database_expert_fan_out", database_expert_fan_out)
    instrument_db_graph.add_node("database_expert_fan_in", database_expert_fan_in)
    for i in range(DB_EXPERT_PARALLEL_EXECUTIONS):
        instrument_db_graph.add_node(f"database_expert_{i}", database_expert_node)
        instrument_db_graph.add_edge(f"database_expert_{i}", "database_expert_fan_in")

    instrument_db_graph.add_edge(START, "instrument_db_fan_out")
    instrument_db_graph.add_edge("instrument_id_validator", "instrument_db_fan_in")
    instrument_db_graph.add_edge("instrument_type_validator_fan_out", "instrument_type_validator_fan_in")
    instrument_db_graph.add_edge("instrument_type_validator_fan_in", "instrument_db_fan_in")
    instrument_db_graph.add_edge("instrument_db_fan_in", "database_expert_fan_out")
    instrument_db_graph.add_edge("database_expert_fan_out", "database_expert_fan_in")
    instrument_db_graph.add_edge("database_expert_fan_in", END)

    instrument_db_subgraph = instrument_db_graph.compile()

    def parent_fan_out(state: ContextState):
        logger.info("Context parent fan-out: launching parallel context branches")
        if _has_clarification_requests(state):
            logger.info("parent_fan_out aborted due to clarification requests")
            return Command(goto=END)
        targets = [
            Send("instrument_db_subgraph", state),
            Send("platform_expert", state),
            Send("project_insider", state),
        ]
        targets.extend(
            Send(f"period_expert_{i}", state)
            for i in range(PERIOD_EXPERT_PARALLEL_EXECUTIONS)
        )
        return Command(goto=targets)

    def parent_fan_in(_: ContextState):
        logger.info("Context parent fan-in: merging context updates")
        return {}

    graph = StateGraph(ContextState)
    graph.add_node("parent_fan_out", parent_fan_out)
    graph.add_node("parent_fan_in", parent_fan_in)
    graph.add_node("instrument_db_subgraph", instrument_db_subgraph)
    graph.add_node("platform_expert", platform_expert_node)
    graph.add_node("project_insider", project_insider_node)
    for i in range(PERIOD_EXPERT_PARALLEL_EXECUTIONS):
        graph.add_node(
            f"period_expert_{i}",
            lambda state, branch_id=i: period_expert_node(state, branch_id),
        )

    graph.add_edge(START, "parent_fan_out")
    graph.add_edge("instrument_db_subgraph", "parent_fan_in")
    graph.add_edge("platform_expert", "parent_fan_in")
    graph.add_edge("project_insider", "parent_fan_in")
    for i in range(PERIOD_EXPERT_PARALLEL_EXECUTIONS):
        graph.add_edge(f"period_expert_{i}", "parent_fan_in")
    graph.add_edge("parent_fan_in", END)

    return graph.compile()