from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List

import b2sdk.v1 as b2
import psycopg2
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, ValidationError

from agents.codeact_coder_agent import codeact_coder_agent
from agents.context_orchestrator import get_context_graph
from agents.history_summariser_agent import history_summariser
from agents.map_plot_sandbox_agent import map_plot_sandbox_agent
from agents.query_clarifier import query_clarifier_agent
from agents.reporter_agent import reporter_agent
from agents.response_selector import response_selector
from agents.review_level_agents import (
    breach_instr_agent,
    review_by_time_agent,
    review_by_value_agent,
    review_changes_across_period_agent,
    review_schema_agent,
)
from agents.timeseries_plot_sandbox_agent import timeseries_plot_sandbox_agent
from agents.extraction_sandbox_agent import extraction_sandbox_agent
from utils.json_utils import strip_to_json_payload
from classes import AgentState, Context, Execution
from e2b_sandbox import execute_remote_sandbox, reset_sandbox_pool
from setup import clone_llm_with_overrides
from thinking_messages import child_messages, parent_messages
from tools.artefact_toolkit import WriteArtefactTool
from tools.create_output_toolkit import CSVSaverTool
from tools.sql_security_toolkit import GeneralSQLQueryTool
from utils.chat_history import filter_messages_only_final
from utils.run_cancellation import get_active_run_controller
from utils.sandbox_prewarm import start_sandbox_prewarm_threads

logger = logging.getLogger(__name__)

_stream_message_queue: List[Dict[str, Any]] = []
_stream_message_lock = threading.Lock()
_stream_message_condition = threading.Condition(_stream_message_lock)


def enqueue_stream_message(message: AIMessage) -> None:
    payload = {
        "timestamp": time.monotonic(),
        "content": message.content,
        "additional_kwargs": message.additional_kwargs or {},
    }
    with _stream_message_condition:
        _stream_message_queue.append(payload)
        _stream_message_condition.notify_all()


def wait_for_stream_message(last_seen_ts: float | None, timeout: float) -> Dict[str, Any] | None:
    deadline = time.monotonic() + max(timeout, 0.0)
    with _stream_message_condition:
        while True:
            fresh = [m for m in _stream_message_queue if last_seen_ts is None or m.get("timestamp", 0) > last_seen_ts]
            if fresh:
                return fresh[-1]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            _stream_message_condition.wait(timeout=remaining)


def drain_stream_messages(max_items: int | None = None) -> List[Dict[str, Any]]:
    with _stream_message_condition:
        if not _stream_message_queue:
            return []
        if max_items is None or max_items <= 0:
            items = list(_stream_message_queue)
            _stream_message_queue.clear()
            return items
        items = _stream_message_queue[:max_items]
        del _stream_message_queue[:max_items]
        return items


def clear_stream_message_queue() -> None:
    with _stream_message_lock:
        _stream_message_queue.clear()


class FactItem(BaseModel):
    number: int
    text: str


class FactsResponse(BaseModel):
    facts: List[FactItem]


def _ensure_run_not_cancelled(stage: str) -> None:
    controller = get_active_run_controller()
    if controller:
        controller.raise_if_cancelled(stage)


def _get_parallel_executions_from_env() -> int:
    raw = os.environ.get("MGS_NUM_PARALLEL_EXECUTIONS")
    try:
        value = int(raw) if raw is not None else 1
    except Exception:
        value = 1
    return max(1, value)


def build_graph(
    llms: Dict[str, BaseLanguageModel],
    db: SQLDatabase,
    blob_db: b2.Bucket,
    metadata_db: psycopg2.extensions.connection,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    min_successful_responses: int = 3,
    min_explained_variance: float = 0.7,
    selected_project_key: str | None = None,
    stream_sandbox_logs: bool = True,
) -> StateGraph:
    st.toast("Building graph...", icon=":material/account_tree:")

    num_parallel_executions = _get_parallel_executions_from_env()

    timezone_context_default = st.session_state.get("timezone_context")

    tools_llm = clone_llm_with_overrides(llms["BALANCED"], temperature=0.1)
    sufficiency_llm = clone_llm_with_overrides(llms["BALANCED"], temperature=0.0, max_output_tokens=256)

    extraction_tool = extraction_sandbox_agent(
        llm=tools_llm,
        db=db,
        table_info=table_info,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    _general_sql_query_tool = GeneralSQLQueryTool(
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    _write_artefact_tool = WriteArtefactTool(blob_db=blob_db, metadata_db=metadata_db)
    timeseries_plot_tool = timeseries_plot_sandbox_agent(
        llm=tools_llm,
        sql_tool=_general_sql_query_tool,
        write_artefact_tool=_write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )
    map_plot_tool = map_plot_sandbox_agent(
        llm=tools_llm,
        sql_tool=_general_sql_query_tool,
        write_artefact_tool=_write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )
    review_by_value_tool = review_by_value_agent(
        llm=tools_llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    review_by_time_tool = review_by_time_agent(
        llm=tools_llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    review_schema_tool = review_schema_agent(
        llm=tools_llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    breach_instr_tool = breach_instr_agent(
        llm=tools_llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    review_changes_across_period_tool = review_changes_across_period_agent(
        llm=tools_llm,
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )
    csv_saver_tool = CSVSaverTool(
        write_artefact_tool=_write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )

    try:
        st.session_state.codeact_tools_snapshot = [
            extraction_tool,
            timeseries_plot_tool,
            map_plot_tool,
            review_by_value_tool,
            review_by_time_tool,
            review_schema_tool,
            breach_instr_tool,
            review_changes_across_period_tool,
            csv_saver_tool,
        ]
    except Exception:
        pass


    termination_lock = threading.Lock()
    thinking_stage_lock = threading.Lock()
    termination_triggered = False
    thinking_stage = 0
    parent_messages_pool: List[Dict[str, Any]] = []
    child_messages_pool: List[Dict[str, Any]] = []
    child_emitter_lock = threading.Lock()
    child_emitter_token: object | None = None
    child_message_interval_seconds = 3.0
    successful_executions: List[Dict[str, Any]] = []
    response_facts: List[Dict[str, Any]] = []
    counted_branches: dict[int, int] = {}
    success_order_counter = 0

    def _stop_child_message_emitter() -> None:
        nonlocal child_emitter_token
        with child_emitter_lock:
            child_emitter_token = None

    def _reset_run_state() -> None:
        """Clear termination and fact-tracking state for a fresh run."""
        nonlocal termination_triggered, thinking_stage, successful_executions, response_facts, counted_branches, success_order_counter
        with termination_lock:
            termination_triggered = False
            successful_executions = []
            response_facts = []
            counted_branches = {}
            success_order_counter = 0
        with thinking_stage_lock:
            thinking_stage = 0
        _stop_child_message_emitter()
        try:
            reset_sandbox_pool("new_run")
        except Exception:
            pass

    def _init_child_messages() -> None:
        nonlocal child_messages_pool
        child_messages_pool = []
        for entry in child_messages:
            messages = list(entry["messages"])
            random.shuffle(messages)
            child_messages_pool.append({"thinking_stage": entry["thinking_stage"], "messages": messages})

    def _sample_parent_message(stage: int) -> str:
        source_entry = next(
            (entry for entry in parent_messages if entry.get("thinking_stage") == stage),
            None,
        )
        if not source_entry:
            return ""
        pool_messages = list(source_entry.get("messages") or [])
        return random.choice(pool_messages) if pool_messages else ""

    def _sample_child_message(stage: int) -> str:
        pool_entry = next(
            (entry for entry in child_messages_pool if entry.get("thinking_stage") == stage),
            None,
        )
        if not pool_entry:
            return ""

        pool_messages = pool_entry.get("messages")
        if not pool_messages:
            source_entry = next(
                (entry for entry in child_messages if entry.get("thinking_stage") == stage),
                None,
            )
            if not source_entry:
                return ""
            pool_messages = list(source_entry["messages"])
            random.shuffle(pool_messages)
            pool_entry["messages"] = pool_messages

        return pool_messages.pop() if pool_messages else ""

    _init_child_messages()

    def _start_child_message_emitter(stage: int, process_name: str) -> None:
        nonlocal child_emitter_token
        token = object()
        with child_emitter_lock:
            child_emitter_token = token

        def _emit_loop() -> None:
            time.sleep(child_message_interval_seconds)
            while True:
                with child_emitter_lock:
                    if child_emitter_token is not token:
                        break
                with thinking_stage_lock:
                    current_stage = thinking_stage
                if current_stage != stage:
                    break
                try:
                    _ensure_run_not_cancelled("child_message_emitter")
                except Exception:
                    break

                child_text = _sample_child_message(stage)
                if child_text:
                    msg = AIMessage(
                        content=child_text,
                        additional_kwargs={
                            "level": "info",
                            "is_final": False,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "origin": {
                                "process": process_name,
                                "thinking_stage": stage,
                                "branch_id": None,
                            },
                            "is_child": True,
                            "artefacts": [],
                        },
                    )
                    enqueue_stream_message(msg)
                time.sleep(child_message_interval_seconds)

        threading.Thread(
            target=_emit_loop,
            name=f"child-message-emitter-{stage}",
            daemon=True,
        ).start()

    def _set_thinking_stage(next_stage: int, process_name: str) -> None:
        nonlocal thinking_stage
        message_text = ""
        stage_advanced = False
        with thinking_stage_lock:
            if thinking_stage < next_stage:
                thinking_stage = next_stage
                message_text = _sample_parent_message(next_stage)
                stage_advanced = True
        if message_text:
            msg = AIMessage(
                content=message_text,
                additional_kwargs={
                    "level": "info",
                    "is_final": False,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "origin": {
                        "process": process_name,
                        "thinking_stage": next_stage,
                        "branch_id": None,
                    },
                    "is_child": False,
                    "artefacts": [],
                },
            )
            enqueue_stream_message(msg)
        if stage_advanced:
            if next_stage in (1, 2):
                _start_child_message_emitter(next_stage, process_name)
            else:
                _stop_child_message_emitter()

    def _evaluate_consistency_and_maybe_terminate(branch_id: int) -> None:
        nonlocal termination_triggered
        with termination_lock:
            successes_snapshot = list(successful_executions)
            current_entry = next((e for e in successes_snapshot if e.get("branch_id") == branch_id), None)

        total_responses = len(successes_snapshot)
        threshold_met = total_responses >= min_successful_responses
        if not current_entry:
            logger.info("[Consistency] Branch %d has no recorded successful execution; skipping.", branch_id)
            return

        facts_snapshot = [f for f in response_facts if isinstance(f, dict)]
        fact_numbers_sorted = sorted({f.get("number") for f in facts_snapshot if isinstance(f.get("number"), int)})
        if not fact_numbers_sorted:
            logger.info("[Consistency] No facts available to build covariance matrix; skipping.")
            return

        index_map = {num: idx for idx, num in enumerate(fact_numbers_sorted)}
        n = len(fact_numbers_sorted)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for entry in successes_snapshot:
            indices = entry.get("response_fact_indices") or []
            valid = sorted(set(num for num in indices if num in index_map))
            if not valid:
                continue
            for num in valid:
                idx = index_map[num]
                matrix[idx][idx] += 1
            for a, b in combinations(valid, 2):
                ia, ib = index_map[a], index_map[b]
                matrix[ia][ib] += 1
                matrix[ib][ia] += 1

        logger.info(
            "[Consistency] Covariance matrix fact order (rows = columns): %s",
            fact_numbers_sorted,
        )
        if fact_numbers_sorted:
            col_header = " ".join(str(num) for num in fact_numbers_sorted)
            logger.info("[Consistency] Columns -> %s", col_header)
        for idx_row, row in enumerate(matrix):
            row_label = fact_numbers_sorted[idx_row]
            row_values = " ".join(str(row[index]) for index in range(len(row)))
            logger.info("[Consistency] Row %s -> %s", row_label, row_values)

        off_diag_nonzero = any(matrix[i][j] for i in range(n) for j in range(n) if i != j)
        trace = sum(matrix[i][i] for i in range(n))
        logger.info(
            "[Consistency] Covariance matrix built: size=%d trace=%d off_diag=%s responses=%d (threshold=%d)",
            n,
            trace,
            off_diag_nonzero,
            total_responses,
            min_successful_responses,
        )

        if trace == 0:
            logger.info("[Consistency] Trace is zero; skipping eigen analysis.")
            return

        def _connected_blocks() -> List[List[int]]:
            visited: set[int] = set()
            blocks: List[List[int]] = []
            for i in range(n):
                if i in visited:
                    continue
                stack = [i]
                comp: List[int] = []
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.append(node)
                    for j in range(n):
                        if node == j:
                            continue
                        if matrix[node][j]:
                            stack.append(j)
                blocks.append(sorted(comp))
            return blocks

        def _pairable(block: List[int]) -> bool:
            for i in block:
                degree = sum(1 for j in block if j != i and matrix[i][j])
                if degree > 1:
                    return False
            return True

        def _block_eigen(block: List[int]) -> tuple[float, List[float]]:
            size = len(block)
            if size == 1:
                idx = block[0]
                val = float(matrix[idx][idx])
                vec = [0.0] * n
                vec[idx] = 1.0
                return val, vec
            if size == 2:
                i, j = block
                a = float(matrix[i][i])
                b = float(matrix[i][j])
                c = float(matrix[j][j])
                trace_local = a + c
                det_local = a * c - b * b
                disc = max(trace_local * trace_local - (4 * det_local), 0.0)
                root = math.sqrt(disc)
                lam1 = 0.5 * (trace_local + root)
                v_raw = [b, lam1 - a] if abs(b) > abs(lam1 - a) else [lam1 - c, b]
                norm = math.sqrt(sum(v * v for v in v_raw)) or 1.0
                vec = [0.0] * n
                vec[i] = v_raw[0] / norm
                vec[j] = v_raw[1] / norm
                return lam1, vec
            if _pairable(block):
                visited_block: set[int] = set()
                subblocks: List[List[int]] = []
                for idx in block:
                    if idx in visited_block:
                        continue
                    neighbours = [nbr for nbr in block if nbr != idx and matrix[idx][nbr]]
                    if neighbours:
                        partner = neighbours[0]
                        visited_block.add(idx)
                        visited_block.add(partner)
                        subblocks.append([idx, partner])
                    else:
                        visited_block.add(idx)
                        subblocks.append([idx])

                best_val = -math.inf
                best_vec = [0.0] * n
                for sub in subblocks:
                    val, vec = _block_eigen(sub)
                    if val > best_val:
                        best_val = val
                        best_vec = vec
                return best_val, best_vec

            sub_indices = block
            size = len(sub_indices)
            vec_local = [1.0 / size] * size
            max_iters = min(60, 10 + size * 5)
            sub_matrix = [[float(matrix[i][j]) for j in sub_indices] for i in sub_indices]
            for _ in range(max_iters):
                new_vec = [sum(sub_matrix[i][j] * vec_local[j] for j in range(size)) for i in range(size)]
                norm = math.sqrt(sum(v * v for v in new_vec))
                if norm == 0:
                    break
                new_vec = [v / norm for v in new_vec]
                if max(abs(new_vec[i] - vec_local[i]) for i in range(size)) < 1e-6:
                    vec_local = new_vec
                    break
                vec_local = new_vec

            rayleigh_num = sum(
                vec_local[i] * sum(sub_matrix[i][j] * vec_local[j] for j in range(size))
                for i in range(size)
            )
            rayleigh_den = sum(v * v for v in vec_local) or 1e-9
            lam_val = rayleigh_num / rayleigh_den
            vec = [0.0] * n
            for local_idx, idx in enumerate(sub_indices):
                vec[idx] = vec_local[local_idx]
            return lam_val, vec

        if not off_diag_nonzero:
            diag_values = [float(matrix[i][i]) for i in range(n)]
            largest_val = max(diag_values)
            max_idx = diag_values.index(largest_val)
            principal_component = [0.0] * n
            principal_component[max_idx] = 1.0
            logger.info(
                "[Consistency] Matrix is diagonal; largest eigenvalue=%s at fact=%s.",
                largest_val,
                fact_numbers_sorted[max_idx],
            )
        else:
            blocks = _connected_blocks()
            largest_val = -math.inf
            principal_component = [0.0] * n
            for block in blocks:
                block_val, block_vec = _block_eigen(block)
                if block_val > largest_val:
                    largest_val = block_val
                    principal_component = block_vec
            logger.info(
                "[Consistency] Block-diagonal analysis complete; largest eigenvalue=%s across %d block(s).",
                largest_val,
                len(blocks),
            )

        principal_eigenvalue = largest_val if largest_val != -math.inf else 0.0
        denom = trace + (n / total_responses if total_responses else 0)
        explained_variance = principal_eigenvalue / (denom or 1e-9)

        with termination_lock:
            for idx, entry in enumerate(successful_executions):
                if entry.get("branch_id") == branch_id:
                    updated_entry = dict(entry)
                    updated_entry.update(
                        {
                            "principal_eigenvalue": principal_eigenvalue,
                            "explained_variance": explained_variance,
                            "principal_component_1": principal_component,
                            "principal_component_facts": fact_numbers_sorted,
                        }
                    )
                    successful_executions[idx] = updated_entry
                    current_entry = updated_entry
                    break

        logger.info(
            "[Consistency] Eigenvalue=%0.4f, REV=%0.4f (threshold=%0.2f) for branch %d (responses=%d, threshold_met=%s).",
            principal_eigenvalue,
            explained_variance,
            min_explained_variance,
            branch_id,
            total_responses,
            threshold_met,
        )

        def _ranking(entry: Dict[str, Any]) -> List[int] | None:
            vector = entry.get("principal_component_1")
            facts_order = entry.get("principal_component_facts")
            if not isinstance(vector, list) or not isinstance(facts_order, list):
                return None
            if len(vector) != len(facts_order):
                return None
            try:
                pairs = [
                    (int(fact_num), abs(float(weight)), int(fact_num))
                    for fact_num, weight in zip(facts_order, vector)
                ]
            except Exception:
                return None
            pairs.sort(key=lambda x: (-x[1], x[0]))
            return [p[0] for p in pairs]

        current_order = current_entry.get("order") if isinstance(current_entry, dict) else None
        if not isinstance(current_order, int):
            logger.info("[Consistency] Missing arrival order; skipping termination check.")
            return

        current_ranking = _ranking(current_entry)
        if not current_ranking:
            logger.info("[Consistency] Missing ranking data for current execution; skipping termination.")
            return

        with termination_lock:
            for idx, entry in enumerate(successful_executions):
                if entry.get("branch_id") == branch_id and entry.get("order") == current_order:
                    updated_entry = dict(entry)
                    updated_entry["principal_component_ranking"] = current_ranking
                    successful_executions[idx] = updated_entry

        if not threshold_met:
            logger.info(
                "[Consistency] Below minimum responses (%d/%d); metrics stored, termination not evaluated.",
                total_responses,
                min_successful_responses,
            )
            return

        if explained_variance < min_explained_variance:
            logger.info("[Consistency] REV below threshold; not terminating.")
            return

        if min_successful_responses == 1:
            with termination_lock:
                if termination_triggered:
                    logger.info("[Consistency] Termination already triggered; skipping duplicate cancel.")
                    return
                termination_triggered = True

            _set_thinking_stage(4, "_evaluate_consistency_and_maybe_terminate")

            controller = get_active_run_controller()
            if controller:
                controller.cancel_active_resources("consistency threshold met (single response)")
            try:
                reset_sandbox_pool("terminated_early")
            except Exception:
                pass
            logger.info(
                "[Consistency] Termination triggered with single sufficient response (REV=%0.4f).",
                explained_variance,
            )
            return

        with termination_lock:
            latest_snapshot = list(successful_executions)

        prior_candidates = [
            e for e in latest_snapshot if isinstance(e, dict) and isinstance(e.get("order"), int) and e.get("order") < current_order
        ]
        if not prior_candidates:
            logger.info("[Consistency] No prior execution to compare; skipping termination.")
            return
        prior_entry = max(prior_candidates, key=lambda e: e.get("order", -1))

        prior_ranking = _ranking(prior_entry)
        if not prior_ranking:
            prior_ranking = current_ranking
            logger.info("[Consistency] Prior execution lacked ranking; using current ranking as fallback.")

        with termination_lock:
            for idx, entry in enumerate(successful_executions):
                if entry.get("branch_id") == prior_entry.get("branch_id") and entry.get("order") == prior_entry.get("order"):
                    updated_prior = dict(entry)
                    updated_prior["principal_component_ranking"] = prior_ranking
                    successful_executions[idx] = updated_prior
        if current_ranking != prior_ranking:
            logger.info(
                "[Consistency] Ranking differs (prev=%s, current=%s); skipping termination.",
                prior_ranking,
                current_ranking,
            )
            return

        with termination_lock:
            if termination_triggered:
                logger.info("[Consistency] Termination already triggered; skipping duplicate cancel.")
                return
            termination_triggered = True

        _set_thinking_stage(4, "_evaluate_consistency_and_maybe_terminate")

        controller = get_active_run_controller()
        if controller:
            controller.cancel_active_resources("consistency threshold met")
        try:
            reset_sandbox_pool("terminated_early")
        except Exception:
            pass
        logger.info(
            "[Consistency] Termination triggered after REV=%0.4f with stable ranking %s.",
            explained_variance,
            current_ranking,
        )

    def history_summariser_node(state: AgentState) -> dict:
        _reset_run_state()
        start_sandbox_prewarm_threads(
            num_slots=num_parallel_executions,
            table_info=table_info,
            table_relationship_graph=table_relationship_graph,
            thread_id=thread_id,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access,
            selected_project_key=selected_project_key,
            log_callback=_emit_sandbox_prewarm_log,
        )
        _ensure_run_not_cancelled("history_summariser")
        retrospective_query = history_summariser(messages=state.messages, llm=llms["BALANCED"])
        base_context = state.context if isinstance(state.context, Context) else None
        timezone_context = None
        if base_context and base_context.timezone_context:
            timezone_context = base_context.timezone_context
        elif timezone_context_default:
            timezone_context = timezone_context_default

        update_payload = {"retrospective_query": retrospective_query}
        if timezone_context:
            update_payload["timezone_context"] = timezone_context

        new_context = (base_context or Context(retrospective_query=retrospective_query)).model_copy(
            update=update_payload
        )
        return {"context": new_context}

    def context_orchestrator_node(state: AgentState):
        _ensure_run_not_cancelled("context_orchestrator")
        base_context_dict: Dict[str, Any] = {}
        if isinstance(state.context, Context):
            base_context_dict = state.context.model_dump(exclude_none=True)
        elif isinstance(state.context, dict):
            base_context_dict = state.context
        sub_input = {"messages": state.messages, "context": base_context_dict, "clarification_requests": []}
        sub_graph = get_context_graph(llms, db, selected_project_key)
        _set_thinking_stage(1, "context_orchestrator_node")
        accumulated_context = dict(base_context_dict)
        for sub_chunk in sub_graph.stream(sub_input, stream_mode="updates"):
            _ensure_run_not_cancelled("context_orchestrator_stream")
            for _, context_update in sub_chunk.items():
                if not context_update:
                    continue
                update_model = context_update.get("context") if isinstance(context_update, dict) else None
                if update_model is None:
                    continue
                try:
                    update_dict = (
                        update_model.model_dump(exclude_none=True)
                        if isinstance(update_model, Context)
                        else update_model
                        if isinstance(update_model, dict)
                        else {}
                    )
                    update_dict = {k: v for k, v in update_dict.items() if v is not None}
                    accumulated_context = {**accumulated_context, **update_dict}
                except Exception as exc:
                    logger.error("Failed to merge context update: %s", exc)
        try:
            merged_context_model = Context.model_validate(accumulated_context)
        except Exception as exc:
            logger.error("Failed to validate merged context dict: %s", exc)
            merged_context_model = state.context if isinstance(state.context, Context) else None
        return {"context": merged_context_model}

    def router_after_context(state: AgentState):
        if state.context and state.context.clarification_requests:
            return "request_clarification"
        return "continue_execution"

    def execution_initializer_node(state: AgentState) -> dict:
        _ensure_run_not_cancelled("execution_initializer")
        new_executions = [
            Execution(
                parallel_agent_id=i,
                retry_number=0,
                codeact_code="",
                final_response=None,
                artefacts=[],
                error_summary="",
                is_sufficient=False,
            )
            for i in range(num_parallel_executions)
        ]
        return {"executions": new_executions}

    def query_clarifier_node(state: AgentState) -> dict:
        _ensure_run_not_cancelled("query_clarifier")
        clarification_requests = state.context.clarification_requests if state.context else []
        chat_history = filter_messages_only_final(state.messages)
        message = query_clarifier_agent(
            llm=llms["BALANCED"],
            clarification_requests=clarification_requests,
            chat_history=chat_history,
        )
        return {"messages": [message]}

    def enter_parallel_execution_node(state: AgentState):
        _ensure_run_not_cancelled("enter_parallel_execution")
        _set_thinking_stage(2, "enter_parallel_execution_node")
        targets = [Send(f"run_branch_{i}", state) for i in range(num_parallel_executions)]
        return Command(goto=targets)

    def exit_parallel_execution_node(state: AgentState):
        _ensure_run_not_cancelled("exit_parallel_execution")
        _set_thinking_stage(4, "exit_parallel_execution_node")
        try:
            reset_sandbox_pool("execution_complete")
        except Exception:
            pass
        return {}

    def response_selector_node(state: AgentState) -> dict:
        _ensure_run_not_cancelled("response_selector")
        updated_executions = response_selector(
            llm=llms["BALANCED"],
            executions=state.executions,
            context=state.context,
            successful_executions=list(successful_executions),
            response_facts=list(response_facts),
        )
        return {"executions": updated_executions}

    def reporter_node(state: AgentState) -> Dict[str, List]:
        _ensure_run_not_cancelled("reporter")
        base_len = len(state.messages)
        updated_messages_full = reporter_agent(
            llm=llms["FAST"],
            context=state.context,
            messages=state.messages,
            executions=state.executions,
        )
        new_messages = updated_messages_full[base_len:]
        with thinking_stage_lock:
            current_stage = thinking_stage
        for msg in new_messages:
            if not isinstance(msg, AIMessage):
                continue
            additional_kwargs = dict(msg.additional_kwargs or {})
            origin = additional_kwargs.get("origin")
            if not isinstance(origin, dict):
                origin = {}
            origin["thinking_stage"] = current_stage
            origin["branch_id"] = None
            additional_kwargs["origin"] = origin
            msg.additional_kwargs = additional_kwargs
        return {"messages": new_messages}

    graph = StateGraph(AgentState)

    def _emit_sandbox_prewarm_log(message: str, level: str, branch_id: int) -> None:
        msg = AIMessage(
            content=message,
            additional_kwargs={
                "level": level,
                "is_final": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "origin": {
                    "process": "sandbox_log",
                    "thinking_stage": 1,
                    "branch_id": branch_id,
                },
                "is_child": True,
                "artefacts": [],
            },
        )
        enqueue_stream_message(msg)

    def make_run_branch(branch_id: int):
        def _progress_msg(
            content: str,
            level: str = "debug",
            metadata: Dict[str, Any] | None = None,
        ) -> AIMessage:
            with thinking_stage_lock:
                current_stage = thinking_stage
            default_origin = {
                "process": "execution_branch",
                "thinking_stage": current_stage,
                "branch_id": branch_id,
            }
            default_kwargs: Dict[str, Any] = {
                "level": level,
                "is_final": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "origin": default_origin,
                "is_child": True,
                "artefacts": [],
            }

            if metadata:
                additional_kwargs = dict(default_kwargs)
                for key, value in metadata.items():
                    if key == "origin" and isinstance(value, dict):
                        merged_origin = dict(default_origin)
                        for origin_key, origin_value in value.items():
                            if origin_value is not None:
                                merged_origin[origin_key] = origin_value
                        additional_kwargs["origin"] = merged_origin
                    elif value is not None:
                        additional_kwargs[key] = value
            else:
                additional_kwargs = default_kwargs

            msg = AIMessage(
                content=str(content),
                additional_kwargs=additional_kwargs,
            )
            enqueue_stream_message(msg)
            return msg

        def _code_display_message(code_text: str) -> AIMessage:
            msg = AIMessage(
                content=f"```python\n{code_text}\n```",
                additional_kwargs={
                    "level": "debug",
                    "is_final": False,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "origin": {
                        "process": "code_correcter",
                        "thinking_stage": 2,
                        "branch_id": branch_id,
                    },
                    "is_child": True,
                    "artefacts": [],
                },
            )
            enqueue_stream_message(msg)
            return msg

        def _flatten_message_content(message: BaseMessage | None) -> str:
            if message is None:
                return ""
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                try:
                    return "\n".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
                except Exception:
                    return str(content)
            return str(content)

        def _judge_sufficiency(
            final_msg: BaseMessage | None,
            user_query: str,
            branch_context: List[str] | None = None,
            artefacts: List[Dict[str, Any]] | None = None,
        ) -> bool:
            if not final_msg:
                return False
            final_response_text = _flatten_message_content(final_msg)
            context_notes = "\n".join([c for c in (branch_context or []) if isinstance(c, str) and c.strip()])
            artefact_notes = []
            for item in artefacts or []:
                if not isinstance(item, dict):
                    continue
                artefact_type = item.get("type")
                artefact_desc = item.get("description")
                if artefact_desc or artefact_type:
                    artefact_notes.append(
                        f"- type={artefact_type or 'unknown'} desc={artefact_desc or '(no description)'}"
                    )
            artefact_summary = "\n".join(artefact_notes) if artefact_notes else "(none)"
            prompt = ("""
                You are a fast, strict judge.
                Decide if the assistant's final reply fully answers the user's latest request.
                Use a concise checklist:
                    (1) answers the question
                    (2) no unresolved TODOs, errors, or apologies;
                    (3) actionable, specific, and contextually relevant;
                    (4) flags missing data if needed.
                You are also given the list of artefacts actually produced by the response.
                If the response references an artefact, verify it appears in the produced list.
                Return ONLY compact JSON: {"sufficient": true|false, "notes": "<=30 words why"}.
            """)
            llm_input = [
                ("system", prompt),
                (
                    "human",
                    (
                        "User query:\n"
                        f"{user_query}\n\n"
                        # f"Assistant plan/notes:\n{context_notes}\n\n"
                        f"Produced artefacts:\n{artefact_summary}\n\n"
                        f"Assistant final reply:\n{final_response_text}\n\nIs it sufficient?"
                    ),
                ),
            ]
            logger.info("[Sufficiency] Evaluating branch %d response sufficiency with prompt %s", branch_id, llm_input)
            try:
                llm_result = sufficiency_llm.invoke(llm_input)
                raw_text = _flatten_message_content(llm_result)
                parsed: dict[str, Any] | None = None
                try:
                    raw_text = strip_to_json_payload(
                        raw_text,
                        [
                            '"sufficient"',
                        ],
                    )
                    parsed = json.loads(raw_text)
                    logger.info("[Sufficiency] Parsed JSON sufficiency result: %s", parsed)
                except Exception:
                    lowered = raw_text.strip().lower()
                    if "true" in lowered and "false" not in lowered:
                        parsed = {"sufficient": True, "notes": raw_text[:200]}
                    elif "false" in lowered:
                        parsed = {"sufficient": False, "notes": raw_text[:200]}
                return bool(parsed and str(parsed.get("sufficient")).lower() == "true")
            except Exception:
                return False

        def _decompose_and_update_facts(final_msg: BaseMessage | None, state: AgentState) -> List[int]:
            nonlocal response_facts
            if not final_msg:
                return []
            current_response = _flatten_message_content(final_msg).strip()
            if not current_response:
                return []

            retrospective_query = ""
            if state and getattr(state, "context", None):
                try:
                    retrospective_query = getattr(state.context, "retrospective_query", "") or ""
                except Exception:
                    retrospective_query = ""

            system_prompt = (
                """
                # Role
                You are an expert fact identifier working on a response to a user query.

                # Instructions
                Extract only facts that materially answer the query or cite results.
                Extract only atomic facts; decompose compound facts until no smaller fact remains.
                Reference candidate_facts to see if any facts you identified match.
                Matching facts:
                - need not share exact wording
                - must match semantically
                - must not contradict
                Number identified facts according to candidate_facts numbering: - if a fact matches a candidate_fact in meaning, reuse its number;
                - if a fact is new, assign it the next integer after the current maximum number in candidate_facts.
                Output all facts identified in the current response along with their numbers.

                # Chain-of-Thought
                Always think step-by-step:
                1) restate the user query context
                2) restate the current response
                3) list atomic facts in the current response
                4) compare candidates to candidate_facts for semantic matches that do not contradict
                5) assign numbers
                6) output JSON only

                # Output Format
                You MUST respond using the structured output tool for FactsResponse.
                Do NOT print raw JSON, code, or any extra text. The tool arguments must conform to:
                - facts (list of {{number:int, text:str}}) where text is the concise wording of the fact as it appears in the response.
                Facts must stay sorted by number; text must be drawn from or faithfully paraphrase the response.
                Do the reasoning internally but DO NOT include chain-of-thought in the final tool output.

                # Example
                ## Input
                candidate_facts = ["1. The sky is blue.", "2. Water boils at 100C."]
                user_query = "What color is the sky and at what temperature does water boil?"
                response = "The sky is green and water boils at 100 degrees Celsius."
                ## Output
                {{"facts": [{{"number": 2, "text": "Water boils at 100C."}}, {{"number": 3, "text": "The sky is green."}}]}}
                """
            )

            numbered_facts: List[str] = []
            try:
                sorted_facts = sorted(
                    [f for f in response_facts if isinstance(f, dict)],
                    key=lambda x: x.get("number", 0),
                )
                for f in sorted_facts:
                    num = f.get("number")
                    txt = f.get("text")
                    if isinstance(num, int) and isinstance(txt, str):
                        numbered_facts.append(f"{num}. {txt}")
            except Exception:
                numbered_facts = []

            payload = {
                "candidate_facts": numbered_facts,
                "response": current_response,
                "user_query": retrospective_query,
            }

            fact_llm = clone_llm_with_overrides(llms["BALANCED"], temperature=0.0, max_output_tokens=768)
            structured_llm = fact_llm.with_structured_output(FactsResponse)

            max_format_retries = 1
            llm_result: FactsResponse | None = None
            for attempt in range(max_format_retries + 1):
                try:
                    candidate = structured_llm.invoke([
                        ("system", system_prompt),
                        ("human", json.dumps(payload)),
                    ])

                    if not candidate or not getattr(candidate, "facts", None):
                        raise ValueError("Fact decomposition returned no facts.")

                    llm_result = candidate
                    break
                except Exception as exc:
                    msg = str(exc).lower()
                    is_format_issue = isinstance(exc, (ValidationError, json.JSONDecodeError)) or any(
                        hint in msg for hint in ["parse", "json", "schema", "pydantic", "format", "no facts"]
                    )

                    if is_format_issue and attempt < max_format_retries:
                        logger.warning(
                            "Fact decomposition format issue (attempt %d/%d); retrying: %s",
                            attempt + 1,
                            max_format_retries + 1,
                            exc,
                        )
                        continue

                    logger.error("Failed to invoke fact decomposition LLM: %s", exc)
                    return []

            if not llm_result:
                logger.error("Fact decomposition failed: no response after retries.")
                return []

            facts = [fact.model_dump() for fact in llm_result.facts]

            current_indices: List[int] = []
            existing_numbers = {f.get("number") for f in response_facts if isinstance(f, dict)}
            new_facts: List[Dict[str, Any]] = []

            for f in facts:
                if not isinstance(f, dict):
                    continue
                num = f.get("number")
                if isinstance(num, int):
                    current_indices.append(num)
                    if num not in existing_numbers:
                        new_facts.append(f)
            try:
                current_indices = sorted(set(current_indices))
            except Exception:
                pass

            if new_facts:
                merged = list(response_facts) + new_facts
                try:
                    merged.sort(key=lambda x: x.get("number") if isinstance(x, dict) else 0)
                except Exception:
                    pass
                response_facts = merged

            return current_indices

        def run_branch(state: AgentState):
            _ensure_run_not_cancelled(f"run_branch_{branch_id}")
            ex = next((e for e in state.executions if e.parallel_agent_id == branch_id), None)
            if not ex or ex.final_response is not None:
                return {}

            if termination_triggered:
                return {}

            payload: Dict[str, Any] = {}

            _ensure_run_not_cancelled(f"run_branch_{branch_id}:codeact")
            coder_result = codeact_coder_agent(
                generating_llm=llms["CODING"],
                tools=[
                    extraction_tool,
                    timeseries_plot_tool,
                    map_plot_tool,
                    review_by_value_tool,
                    review_by_time_tool,
                    review_schema_tool,
                    breach_instr_tool,
                    review_changes_across_period_tool,
                    csv_saver_tool,
                ],
                context=state.context,
                previous_attempts=[p for p in state.executions if p.parallel_agent_id == branch_id and p.retry_number < ex.retry_number],
            )
            new_execution = coder_result["executions"][0]
            code = (new_execution.codeact_code or "").strip()
            updated = ex.model_copy(
                update={
                    "codeact_code": code,
                    "objective": getattr(new_execution, "objective", "") or "",
                    "plan": list(getattr(new_execution, "plan", []) or []),
                }
            )

            if termination_triggered:
                payload["executions"] = [updated]
                return payload

            coder_messages: List[AIMessage] = list(coder_result.get("messages", []) or [])
            for msg in coder_messages:
                if not isinstance(msg, AIMessage):
                    continue
                additional_kwargs = dict(msg.additional_kwargs or {})
                origin = additional_kwargs.get("origin")
                if not isinstance(origin, dict):
                    origin = {}
                origin["thinking_stage"] = 2
                origin["branch_id"] = branch_id
                additional_kwargs["origin"] = origin
                msg.additional_kwargs = additional_kwargs
                enqueue_stream_message(msg)

            def _strip_code_fences(text: str) -> str:
                if "```" not in text:
                    return text.strip()
                parts = text.split("```")
                if len(parts) >= 3:
                    candidate = parts[1]
                    if candidate.lstrip().startswith("python"):
                        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
                    return candidate.strip()
                return text.strip()

            def _fix_syntax_error(code_to_fix: str, error_summary: str) -> str:
                prompt = (
                    "You are a fast Python syntax fixer. Only correct the syntax error described. "
                    "Make no other changes. Do not refactor, rename, or adjust logic. "
                    "Return ONLY the entire code with corrections and no markdown."
                )
                llm_input = [
                    ("system", prompt),
                    (
                        "human",
                        f"Syntax error summary:\n{error_summary}\n\nCode to fix:\n{code_to_fix}",
                    ),
                ]
                llm_result = llms["FAST"].invoke(llm_input)
                raw_text = _flatten_message_content(llm_result)
                cleaned = _strip_code_fences(raw_text)
                return cleaned if cleaned else code_to_fix

            def _is_timeout_or_cancel(error_summary: str) -> bool:
                if not error_summary:
                    return False
                lowered = error_summary.lower()
                timeout_signals = ["timeout", "timed out", "time out", "deadline exceeded"]
                cancel_signals = ["cancelled", "canceled", "cancel", "aborted", "user abort"]
                return any(sig in lowered for sig in timeout_signals + cancel_signals)

            def _execute_code(code_to_run: str) -> tuple[List[AIMessage], AIMessage | None, List[Dict[str, Any]], List[str], List[str], float]:
                messages: List[AIMessage] = []
                artefacts: List[Dict[str, Any]] = []
                final_msg: AIMessage | None = None
                error_logs_all: List[str] = []
                error_logs_actionable: List[str] = []
                terminated_early = False
                start_ts = time.monotonic()
                if code_to_run:
                    try:
                        _set_thinking_stage(3, "_execute_code")
                        gen = execute_remote_sandbox(
                            code=code_to_run,
                            table_info=table_info,
                            table_relationship_graph=table_relationship_graph,
                            thread_id=thread_id,
                            user_id=user_id,
                            global_hierarchy_access=global_hierarchy_access,
                            selected_project_key=selected_project_key,
                            container_slot=branch_id,
                            stream_sandbox_logs=stream_sandbox_logs,
                        )
                        if termination_triggered:
                            terminated_early = True
                        else:
                            for out in gen:
                                _ensure_run_not_cancelled(f"run_branch_{branch_id}:sandbox_stream")
                                if termination_triggered:
                                    terminated_early = True
                                    break
                                if not isinstance(out, dict):
                                    err = f"Invalid output: {out}"
                                    error_logs_all.append(err)
                                    messages.append(_progress_msg(
                                        err,
                                        level="error",
                                        metadata={
                                            "origin": {
                                                "process": "sandbox_log",
                                                "thinking_stage": 3,
                                                "branch_id": branch_id
                                            }
                                        }
                                    ))
                                    continue
                                if "__control__" in out:
                                    continue

                                content = out.get("content", "")
                                additional_kwargs: Dict[str, Any] = {}
                                meta_type = ""

                                if isinstance(out.get("additional_kwargs"), dict):
                                    additional_kwargs = dict(out.get("additional_kwargs") or {})

                                if isinstance(out.get("metadata"), dict):
                                    metadata = out.get("metadata") or {}
                                    meta_type = str(metadata.get("type") or "").lower()

                                artefact_payloads = additional_kwargs.get("artefacts")
                                if isinstance(artefact_payloads, list):
                                    for entry in artefact_payloads:
                                        if isinstance(entry, dict):
                                            artefacts.append(entry)

                                origin = additional_kwargs.get("origin")
                                if not isinstance(origin, dict):
                                    origin = {}
                                origin["thinking_stage"] = 3
                                origin["branch_id"] = branch_id
                                additional_kwargs["origin"] = origin

                                msg = AIMessage(
                                    content=str(content),
                                    additional_kwargs=additional_kwargs,
                                )
                                enqueue_stream_message(msg)
                                messages.append(msg)

                                if meta_type == "final" and additional_kwargs.get("artefacts") == []:
                                    final_msg = msg

                                level = additional_kwargs.get("level")
                                origin_process = origin.get("process")
                                if level == "error":
                                    error_logs_all.append(str(content))
                                    if origin_process == "code_yield":
                                        error_logs_actionable.append(str(content))
                    except Exception as e:
                        err = f"Sandbox error: {e}"
                        error_logs_all.append(err)
                        error_logs_actionable.append(err)
                        messages.append(_progress_msg(
                            err,
                            level="error",
                            metadata={
                                "origin": {"process": "sandbox_log"},
                                "thinking_stage": 3,
                                "branch_id": branch_id,
                            }
                        ))
                    finally:
                        if terminated_early:
                            try:
                                gen.close()
                            except Exception:
                                pass
                exec_duration = time.monotonic() - start_ts
                return messages, final_msg, artefacts, error_logs_all, error_logs_actionable, exec_duration

            execution_updates: List[Execution] = []
            fact_numbers: List[int] = []
            current_execution = updated
            current_code = code
            final_msg: AIMessage | None = None
            max_retry_number = 2
            max_codegen_retries = 2
            codegen_retry_number = 0

            while True:
                attempt_messages, final_msg, artefacts, error_logs_all, error_logs_actionable, exec_duration = _execute_code(current_code)

                updated_attempt = current_execution.model_copy(
                    update={
                        "final_response": final_msg,
                        "artefacts": artefacts,
                        "error_summary": "\n".join(error_logs_all) if error_logs_all else "",
                        "is_sufficient": False,
                    }
                )

                error_text = updated_attempt.error_summary or ""
                has_syntax_error = error_text in [
                    "SyntaxError",
                    "AttributeError",
                    "IndentationError",
                ]
                logger.info("Syntax error detected in branch %d: %s", branch_id, has_syntax_error)
                deterministic_error = bool(error_logs_actionable)
                message_errors = any(
                    isinstance(m, AIMessage)
                    and isinstance(m.additional_kwargs, dict)
                    and m.additional_kwargs.get("level") == "error"
                    and isinstance(m.additional_kwargs.get("origin"), dict)
                    and m.additional_kwargs["origin"].get("process") != "sandbox_log"
                    for m in attempt_messages
                )
                has_any_error = bool(error_text) or deterministic_error or message_errors
                is_timeout_or_cancel = _is_timeout_or_cancel(error_text)

                if not has_syntax_error and not (deterministic_error or message_errors):
                    user_query = state.context.retrospective_query if state.context else ""
                    branch_context: List[str] = []
                    if getattr(updated_attempt, "objective", ""):
                        branch_context.append(f"Objective:\n{updated_attempt.objective}")
                    if getattr(updated_attempt, "plan", None):
                        try:
                            plan_lines = [
                                f"{idx + 1}. {step}"
                                for idx, step in enumerate(updated_attempt.plan)
                                if isinstance(step, str) and step.strip()
                            ]
                        except Exception:
                            plan_lines = []
                        if plan_lines:
                            branch_context.append("Plan:\n" + "\n".join(plan_lines))
                    is_sufficient = _judge_sufficiency(final_msg, user_query, branch_context, artefacts)
                else:
                    is_sufficient = False

                if is_sufficient:
                    fact_numbers = _decompose_and_update_facts(final_msg, state)
                    try:
                        logger.info("[Facts] Current response fact indices: %s", fact_numbers)
                        logger.info("[Facts] Stored response_facts (count=%d): %s", len(response_facts), response_facts)
                    except Exception as exc:
                        logger.error("[Facts] Logging failed: %s", exc)

                updated_attempt = updated_attempt.model_copy(update={"is_sufficient": is_sufficient})
                execution_updates.append(updated_attempt)

                if has_syntax_error and current_execution.retry_number < max_retry_number:
                    _progress_msg(
                        f"SyntaxError detected. Attempting correction (retry {current_execution.retry_number + 1}/{max_retry_number}).",
                        level="error",
                        metadata={"origin": {
                            "process": "code_correcter",
                            "thinking_stage": 2,
                            "branch_id": branch_id
                        }},
                    )
                    corrected_code = _fix_syntax_error(current_code, updated_attempt.error_summary)
                    _code_display_message(corrected_code)
                    current_execution = Execution(
                        parallel_agent_id=updated_attempt.parallel_agent_id,
                        retry_number=updated_attempt.retry_number + 1,
                        objective=updated_attempt.objective,
                        plan=list(updated_attempt.plan),
                        codeact_code=corrected_code,
                        final_response=None,
                        artefacts=[],
                        error_summary="",
                    )
                    current_code = corrected_code
                    continue

                if (
                    has_any_error
                    and not has_syntax_error
                    and not is_timeout_or_cancel
                    and exec_duration <= 10.0
                    and codegen_retry_number < max_codegen_retries
                ):
                    codegen_retry_number += 1
                    _progress_msg(
                        f"Sandbox error detected. Re-running code generation (retry {codegen_retry_number}/{max_codegen_retries}).",
                        level="error",
                        metadata={"origin": {
                            "process": "codeact_coder",
                            "thinking_stage": 2,
                            "branch_id": branch_id,
                        }},
                    )

                    previous_attempts_for_codegen = [
                        p
                        for p in (list(state.executions) + execution_updates)
                        if isinstance(p, Execution) and p.parallel_agent_id == branch_id
                    ]
                    coder_retry_result = codeact_coder_agent(
                        generating_llm=llms["CODING"],
                        tools=[
                            extraction_tool,
                            timeseries_plot_tool,
                            map_plot_tool,
                            review_by_value_tool,
                            review_by_time_tool,
                            review_schema_tool,
                            breach_instr_tool,
                            review_changes_across_period_tool,
                            csv_saver_tool,
                        ],
                        context=state.context,
                        previous_attempts=previous_attempts_for_codegen,
                    )
                    retry_execution = coder_retry_result["executions"][0]
                    retry_code = (retry_execution.codeact_code or "").strip()

                    coder_retry_messages: List[AIMessage] = list(coder_retry_result.get("messages", []) or [])
                    for msg in coder_retry_messages:
                        if not isinstance(msg, AIMessage):
                            continue
                        additional_kwargs = dict(msg.additional_kwargs or {})
                        origin = additional_kwargs.get("origin")
                        if not isinstance(origin, dict):
                            origin = {}
                        origin["thinking_stage"] = 2
                        origin["branch_id"] = branch_id
                        additional_kwargs["origin"] = origin
                        msg.additional_kwargs = additional_kwargs
                        enqueue_stream_message(msg)

                    current_execution = Execution(
                        parallel_agent_id=updated_attempt.parallel_agent_id,
                        retry_number=updated_attempt.retry_number + 1,
                        objective=getattr(retry_execution, "objective", "") or "",
                        plan=list(getattr(retry_execution, "plan", []) or []),
                        codeact_code=retry_code,
                        final_response=None,
                        artefacts=[],
                        error_summary="",
                    )
                    current_code = retry_code
                    continue

                break

            payload["executions"] = execution_updates
            latest_execution = execution_updates[-1] if execution_updates else updated

            nonlocal successful_executions, success_order_counter
            with termination_lock:
                prev_retry = counted_branches.get(branch_id, -1)
                if latest_execution.is_sufficient and final_msg and latest_execution.retry_number >= prev_retry:
                    counted_branches[branch_id] = latest_execution.retry_number
                    successful_executions = [entry for entry in successful_executions if entry.get("branch_id") != branch_id]
                    success_order_counter += 1
                    successful_executions.append(
                        {
                            "branch_id": branch_id,
                            "retry_number": latest_execution.retry_number,
                            "final_message": final_msg,
                            "response_fact_indices": fact_numbers,
                            "order": success_order_counter,
                        }
                    )
                else:
                    counted_branches.pop(branch_id, None)
                    successful_executions = [entry for entry in successful_executions if entry.get("branch_id") != branch_id]
                successes = len(successful_executions)

            logger.info(
                "[Sufficiency] Branch %d tally after update: %d/%d (is_sufficient=%s, actionable_errors=%d, total_errors=%d, counted=%s)",
                branch_id,
                successes,
                min_successful_responses,
                is_sufficient,
                len(error_logs_actionable),
                len(error_logs_all),
                list(counted_branches.items()),
            )
            if is_sufficient:
                _evaluate_consistency_and_maybe_terminate(branch_id)

            return payload or {}

        return run_branch

    graph.add_node("history_summariser_node", history_summariser_node)
    graph.add_node("context_orchestrator_node", context_orchestrator_node)
    graph.add_node("query_clarifier_node", query_clarifier_node)
    graph.add_node("execution_initializer_node", execution_initializer_node)
    graph.add_node("enter_parallel_execution_node", enter_parallel_execution_node)
    graph.add_node("exit_parallel_execution_node", exit_parallel_execution_node)
    graph.add_node("response_selector_node", response_selector_node)
    graph.add_node("reporter_node", reporter_node)

    graph.add_edge(START, "history_summariser_node")
    graph.add_edge("history_summariser_node", "context_orchestrator_node")
    graph.add_conditional_edges(
        "context_orchestrator_node",
        router_after_context,
        {"continue_execution": "execution_initializer_node", "request_clarification": "query_clarifier_node"},
    )
    graph.add_edge("query_clarifier_node", END)
    graph.add_edge("execution_initializer_node", "enter_parallel_execution_node")

    for i in range(num_parallel_executions):
        graph.add_node(f"run_branch_{i}", make_run_branch(i))
        graph.add_edge(f"run_branch_{i}", "exit_parallel_execution_node")

    graph.add_edge("exit_parallel_execution_node", "response_selector_node")
    graph.add_edge("response_selector_node", "reporter_node")
    graph.add_edge("reporter_node", END)

    return graph.compile()
