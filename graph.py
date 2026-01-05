from __future__ import annotations

import json
import logging
import math
import threading
from itertools import combinations
from typing import Any, Dict, List

import b2sdk.v1 as b2
import psycopg2
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel

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
from classes import AgentState, Context, Execution
from modal_sandbox_remote import run_sandboxed_code
from parameters import progress_messages
from setup import clone_llm_with_overrides
from tools.artefact_toolkit import WriteArtefactTool
from tools.create_output_toolkit import CSVSaverTool
from tools.sql_security_toolkit import GeneralSQLQueryTool
from utils.chat_history import filter_messages_only_final
from utils.run_cancellation import get_active_run_controller

logger = logging.getLogger(__name__)


class FactItem(BaseModel):
    number: int
    text: str


class FactsResponse(BaseModel):
    facts: List[FactItem]


def _ensure_run_not_cancelled(stage: str) -> None:
    controller = get_active_run_controller()
    if controller:
        controller.raise_if_cancelled(stage)


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
    num_parallel_executions: int = 2,
    num_completions_before_response: int = 2,
    response_mode: str = "Intelligent",
    min_successful_responses: int = 3,
    min_explained_variance: float = 0.7,
    selected_project_key: str | None = None,
) -> StateGraph:
    st.toast("Building graph...", icon=":material/account_tree:")

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

    termination_lock = threading.Lock()
    termination_triggered = False
    successful_executions: List[Dict[str, Any]] = []
    response_facts: List[Dict[str, Any]] = []
    counted_branches: dict[int, int] = {}
    success_order_counter = 0
    response_mode_normalized = (response_mode or "").strip().lower()

    def _terminate_running_branches_if_threshold_met(successes: int) -> None:
        nonlocal termination_triggered
        if successes < num_completions_before_response:
            return
        with termination_lock:
            if termination_triggered:
                return
            termination_triggered = True
        controller = get_active_run_controller()
        if controller:
            controller.cancel_active_resources("sufficiency threshold met")
        logger.info("[MainGraph] Termination triggered at success count %d", successes)

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

        controller = get_active_run_controller()
        if controller:
            controller.cancel_active_resources("consistency threshold met")
        logger.info(
            "[Consistency] Termination triggered after REV=%0.4f with stable ranking %s.",
            explained_variance,
            current_ranking,
        )

    def progress_messenger_node(state: AgentState, node: str) -> dict:
        _ensure_run_not_cancelled(f"progress:{node}")
        template = progress_messages.get(node)
        if template:
            msg = AIMessage(
                content=template.content,
                additional_kwargs=dict(template.additional_kwargs or {}),
                name=template.name,
            )
            return {"messages": [msg]}
        return {"messages": []}

    def history_summariser_node(state: AgentState) -> dict:
        _ensure_run_not_cancelled("history_summariser")
        retrospective_query = history_summariser(messages=state.messages, llm=llms["BALANCED"])
        base_context = state.context if isinstance(state.context, Context) else None
        new_context = (base_context or Context(retrospective_query=retrospective_query)).model_copy(
            update={"retrospective_query": retrospective_query}
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
        targets = [Send(f"run_branch_{i}", state) for i in range(num_parallel_executions)]
        return Command(goto=targets)

    def exit_parallel_execution_node(state: AgentState):
        _ensure_run_not_cancelled("exit_parallel_execution")
        return {}

    def response_selector_node(state: AgentState) -> dict:
        _ensure_run_not_cancelled("response_selector")
        updated_executions = response_selector(
            llm=llms["THINKING"],
            executions=state.executions,
            context=state.context,
            response_mode=response_mode,
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
        return {"messages": new_messages}

    graph = StateGraph(AgentState)

    def make_run_branch(branch_id: int):
        def _progress_msg(content: str, process: str = "progress", origin: str | None = None) -> AIMessage:
            kwargs = {"stage": "execution_output", "process": process}
            if origin:
                kwargs["origin"] = origin
            return AIMessage(name=f"Executor_{branch_id}", content=str(content), additional_kwargs=kwargs)

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

        def _judge_sufficiency(final_msg: BaseMessage | None, user_query: str) -> bool:
            if not final_msg:
                return False
            final_response_text = _flatten_message_content(final_msg)
            prompt = (
                "You are a fast, strict judge. Decide if the assistant's final reply fully answers the user's latest request.\n"
                "Use a concise checklist: (1) directly answers the question; (2) uses provided artefacts/results when present;"
                " (3) no unresolved TODOs, errors, or apologies; (4) actionable, specific, and contextually relevant;"
                " (5) flags missing data if needed."
                " Return ONLY compact JSON: {\\\"sufficient\\\": true|false, \\\"notes\\\": \\\"<=30 words why\\\"}."
            )
            llm_input = [
                ("system", prompt),
                (
                    "human",
                    f"User query:\n{user_query}\n\nAssistant final reply:\n{final_response_text}\n\nIs it sufficient?",
                ),
            ]
            try:
                llm_result = sufficiency_llm.invoke(llm_input)
                raw_text = _flatten_message_content(llm_result)
                parsed: dict[str, Any] | None = None
                try:
                    parsed = json.loads(raw_text)
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
                Return ONLY JSON with key:
                - facts (list of {{number:int, text:str}}) where text is the concise wording of the fact as it appears in the response.
                Facts must stay sorted by number; text must be drawn from or faithfully paraphrase the response.
                Do the reasoning internally but DO NOT include chain-of-thought in the final JSON output.

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

            try:
                fact_llm = clone_llm_with_overrides(llms["BALANCED"], temperature=0.0, max_output_tokens=768)
                structured_llm = fact_llm.with_structured_output(FactsResponse)
                llm_result: FactsResponse = structured_llm.invoke([
                    ("system", system_prompt),
                    ("human", json.dumps(payload)),
                ])
            except Exception as exc:
                logger.error("Failed to invoke fact decomposition LLM: %s", exc)
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
                generating_llm=llms["THINKING"],
                checking_llm=llms["BALANCED"],
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
            updated = ex.model_copy(update={"codeact_code": code})

            if termination_triggered:
                payload["executions"] = [updated]
                return payload

            messages: List[AIMessage] = list(coder_result.get("messages", []) or [])
            artefacts: List[AIMessage] = []
            final_msg: AIMessage | None = None
            error_logs_all: List[str] = []
            error_logs_actionable: List[str] = []
            terminated_early = False
            if code:
                try:
                    gen = run_sandboxed_code.remote(
                        code=code,
                        table_info=table_info,
                        table_relationship_graph=table_relationship_graph,
                        thread_id=thread_id,
                        user_id=user_id,
                        global_hierarchy_access=global_hierarchy_access,
                        selected_project_key=selected_project_key,
                        container_slot=branch_id,
                    )
                    for out in gen:
                        _ensure_run_not_cancelled(f"run_branch_{branch_id}:sandbox_stream")
                        if termination_triggered:
                            terminated_early = True
                            break
                        if not isinstance(out, dict) or "metadata" not in out:
                            err = f"Invalid output: {out}"
                            error_logs_all.append(err)
                            error_logs_actionable.append(err)
                            continue
                        typ = out["metadata"].get("type")
                        origin = out["metadata"].get("origin")
                        content = out.get("content", "")
                        if typ in ("progress", "error"):
                            messages.append(_progress_msg(str(content), process=typ, origin=origin))
                            if typ == "error":
                                error_logs_all.append(str(content))
                                if origin not in {"stdout", "stderr"}:
                                    error_logs_actionable.append(str(content))
                        elif typ == "final":
                            final_msg = _progress_msg(str(content), process="final", origin=origin)
                            messages.append(final_msg)
                        elif typ in ("plot", "csv"):
                            try:
                                import json as _json

                                jd = _json.loads(content) if isinstance(content, str) else content
                            except Exception:
                                jd = None
                            if jd:
                                msg = AIMessage(
                                    name=f"Executor_{branch_id}",
                                    content=jd.get("description", "(no description)"),
                                    additional_kwargs={"stage": "execution_output", "process": typ, "artefact_id": jd.get("artefact_id")},
                                )
                                messages.append(msg)
                                artefacts.append(msg)
                            else:
                                err = f"Failed to parse {typ}: {str(content)[:200]}"
                                error_logs_all.append(err)
                                error_logs_actionable.append(err)
                                messages.append(_progress_msg(err, process="error"))
                        else:
                            messages.append(_progress_msg(str(content), process="progress", origin=origin))
                except Exception as e:
                    err = f"Sandbox error: {e}"
                    error_logs_all.append(err)
                    error_logs_actionable.append(err)
                    messages.append(_progress_msg(err, process="error"))
                finally:
                    if terminated_early:
                        try:
                            gen.close()
                        except Exception:
                            pass

            updated = updated.model_copy(
                update={
                    "final_response": final_msg,
                    "artefacts": artefacts,
                    "error_summary": "\n".join(error_logs_all) if error_logs_all else "",
                    "is_sufficient": False,
                }
            )

            deterministic_error = bool(error_logs_actionable)
            message_errors = any(
                isinstance(m, AIMessage)
                and m.name == f"Executor_{branch_id}"
                and m.additional_kwargs.get("stage") == "execution_output"
                and m.additional_kwargs.get("process") == "error"
                and m.additional_kwargs.get("origin") not in {"stdout", "stderr"}
                for m in messages
            )
            if deterministic_error or message_errors:
                is_sufficient = False
            else:
                user_query = state.context.retrospective_query if state.context else ""
                is_sufficient = _judge_sufficiency(final_msg, user_query)

            fact_numbers: List[int] = []
            if is_sufficient:
                fact_numbers = _decompose_and_update_facts(final_msg, state)
                try:
                    logger.info("[Facts] Current response fact indices: %s", fact_numbers)
                    logger.info("[Facts] Stored response_facts (count=%d): %s", len(response_facts), response_facts)
                except Exception as exc:
                    logger.error("[Facts] Logging failed: %s", exc)

            updated = updated.model_copy(update={"is_sufficient": is_sufficient})
            payload["executions"] = [updated]
            if messages:
                payload["messages"] = messages

            nonlocal successful_executions, success_order_counter
            with termination_lock:
                prev_retry = counted_branches.get(branch_id, -1)
                if is_sufficient and final_msg and updated.retry_number >= prev_retry:
                    counted_branches[branch_id] = updated.retry_number
                    successful_executions = [entry for entry in successful_executions if entry.get("branch_id") != branch_id]
                    success_order_counter += 1
                    successful_executions.append(
                        {
                            "branch_id": branch_id,
                            "retry_number": updated.retry_number,
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
                num_completions_before_response,
                is_sufficient,
                len(error_logs_actionable),
                len(error_logs_all),
                list(counted_branches.items()),
            )
            if response_mode_normalized == "intelligent":
                if is_sufficient:
                    _evaluate_consistency_and_maybe_terminate(branch_id)
            else:
                _terminate_running_branches_if_threshold_met(successes)

            return payload or {}

        return run_branch

    graph.add_node("history_summariser_messenger_node", lambda state: progress_messenger_node(state, "history_summariser_node"))
    graph.add_node("history_summariser_node", history_summariser_node)
    graph.add_node("context_orchestrator_messenger_node", lambda state: progress_messenger_node(state, "context_orchestrator_node"))
    graph.add_node("context_orchestrator_node", context_orchestrator_node)
    graph.add_node("query_clarifier_node", query_clarifier_node)
    graph.add_node("execution_initializer_messenger_node", lambda state: progress_messenger_node(state, "execution_initializer_node"))
    graph.add_node("execution_initializer_node", execution_initializer_node)
    graph.add_node("enter_parallel_execution_node", enter_parallel_execution_node)
    graph.add_node("exit_parallel_execution_node", exit_parallel_execution_node)
    graph.add_node("reporter_messenger_node", lambda state: progress_messenger_node(state, "reporter_node"))
    graph.add_node("response_selector_node", response_selector_node)
    graph.add_node("reporter_node", reporter_node)

    graph.set_entry_point("history_summariser_messenger_node")
    graph.add_edge("history_summariser_messenger_node", "history_summariser_node")
    graph.add_edge("history_summariser_node", "context_orchestrator_messenger_node")
    graph.add_edge("context_orchestrator_messenger_node", "context_orchestrator_node")
    graph.add_conditional_edges(
        "context_orchestrator_node",
        router_after_context,
        {"continue_execution": "execution_initializer_messenger_node", "request_clarification": "query_clarifier_node"},
    )
    graph.add_edge("query_clarifier_node", END)
    graph.add_edge("execution_initializer_messenger_node", "execution_initializer_node")
    graph.add_edge("execution_initializer_node", "enter_parallel_execution_node")

    for i in range(num_parallel_executions):
        graph.add_node(f"run_branch_{i}", make_run_branch(i))
        graph.add_edge(f"run_branch_{i}", "exit_parallel_execution_node")

    graph.add_edge("exit_parallel_execution_node", "reporter_messenger_node")
    graph.add_edge("reporter_messenger_node", "response_selector_node")
    graph.add_edge("response_selector_node", "reporter_node")
    graph.add_edge("reporter_node", END)

    return graph.compile()
