from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List

import b2sdk.v1 as b2
import psycopg2
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send

from agents.codeact_coder_agent import codeact_coder_agent
from agents.context_orchestrator import get_context_graph
from agents.history_summariser_agent import history_summariser
from agents.map_plot_sandbox_agent import map_plot_sandbox_agent
from agents.query_clarifier import query_clarifier_agent
from agents.query_classifier import query_classifier_agent
from agents.react_agent import react_agent
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
from agents.tool_calling_agent import tool_calling_agent
from agents.extraction_sandbox_agent import extraction_sandbox_agent
from classes import AgentState, Context, Execution
from modal_sandbox_local import execute_local_sandbox
from modal_sandbox_remote import run_sandboxed_code
from parameters import progress_messages
from setup import clone_llm_with_overrides
from tools.artefact_toolkit import WriteArtefactTool
from tools.create_output_toolkit import CSVSaverTool
from tools.sql_security_toolkit import GeneralSQLQueryTool
from utils.chat_history import filter_messages_only_final
from utils.run_cancellation import get_active_run_controller

logger = logging.getLogger(__name__)


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
    remote_sandbox: bool,
    num_parallel_executions: int = 2,
    num_completions_before_response: int = 2,
    agent_type: str = "Auto",
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
    successes_counter = 0
    counted_branches: dict[int, int] = {}

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

    def query_classifier_node(state: AgentState) -> dict:
        _ensure_run_not_cancelled("query_classifier")
        classified_agent_type = (
            query_classifier_agent(llm=llms["BALANCED"], messages=state.messages, context=state.context)
            if agent_type == "Auto"
            else agent_type
        )
        new_executions = []
        for i in range(num_parallel_executions):
            execution = Execution(
                agent_type=classified_agent_type,
                parallel_agent_id=i,
                retry_number=0,
                codeact_code="",
                final_response=None,
                artefacts=[],
                error_summary="",
                is_sufficient=False,
            )
            new_executions.append(execution)
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
        updated_executions = response_selector(llm=llms["THINKING"], executions=state.executions, context=state.context)
        return {"executions": updated_executions}

    def reporter_node(state: AgentState) -> Dict[str, List]:
        _ensure_run_not_cancelled("reporter")
        base_len = len(state.messages)
        updated_messages_full = reporter_agent(
            llm=llms["THINKING"],
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

        def run_branch(state: AgentState):
            _ensure_run_not_cancelled(f"run_branch_{branch_id}")
            ex = next((e for e in state.executions if e.parallel_agent_id == branch_id), None)
            if not ex or ex.final_response is not None:
                return {}

            if termination_triggered:
                return {}

            payload: Dict[str, Any] = {}

            if ex.agent_type == "CodeAct":
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
                        if remote_sandbox:
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
                        else:
                            gen = execute_local_sandbox(
                                code=code,
                                table_info=table_info,
                                table_relationship_graph=table_relationship_graph,
                                thread_id=thread_id,
                                user_id=user_id,
                                global_hierarchy_access=global_hierarchy_access,
                                selected_project_key=selected_project_key,
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
                elif not final_msg:
                    is_sufficient = False
                else:
                    user_query = state.context.retrospective_query if state.context else ""
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
                        ("human", f"User query:\n{user_query}\n\nAssistant final reply:\n{final_response_text}\n\nIs it sufficient?"),
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
                        is_sufficient = bool(parsed and str(parsed.get("sufficient")).lower() == "true")
                    except Exception:
                        is_sufficient = False

                updated = updated.model_copy(update={"is_sufficient": is_sufficient})
                updated_executions = [e if e.parallel_agent_id != branch_id else updated for e in state.executions]
                payload["executions"] = [updated]
                if messages:
                    payload["messages"] = messages

                nonlocal successes_counter
                with termination_lock:
                    prev_retry = counted_branches.get(branch_id, -1)
                    if is_sufficient and final_msg and updated.retry_number >= prev_retry:
                        counted_branches[branch_id] = updated.retry_number
                    else:
                        counted_branches.pop(branch_id, None)
                    successes_counter = len(counted_branches)
                    successes = successes_counter

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
                _terminate_running_branches_if_threshold_met(successes)

            elif ex.agent_type == "ReAct":
                if termination_triggered:
                    return {}
                _ensure_run_not_cancelled(f"run_branch_{branch_id}:react")
                tools = [
                    extraction_tool,
                    timeseries_plot_tool,
                    map_plot_tool,
                    review_by_value_tool,
                    review_by_time_tool,
                    review_schema_tool,
                    breach_instr_tool,
                    review_changes_across_period_tool,
                ]
                result = react_agent(
                    llm=llms["THINKING"],
                    tools=tools,
                    context=state.context,
                    previous_attempts=[p for p in state.executions if p.parallel_agent_id == branch_id and p.retry_number < ex.retry_number],
                )
                updated = ex.model_copy(
                    update={
                        "final_response": result.get("final_response"),
                        "artefacts": result.get("artefacts", []),
                        "error_summary": result.get("error_summary", ""),
                        "is_sufficient": False,
                    }
                )
                msgs = result.get("messages", []) or []
                payload["executions"] = [updated]
                if msgs:
                    payload["messages"] = msgs
                updated_executions = [e if e.parallel_agent_id != branch_id else updated for e in state.executions]
                with termination_lock:
                    prev_retry = counted_branches.get(branch_id, -1)
                    if updated.is_sufficient and updated.final_response and updated.retry_number >= prev_retry:
                        counted_branches[branch_id] = updated.retry_number
                    else:
                        counted_branches.pop(branch_id, None)
                    successes = len(counted_branches)
                _terminate_running_branches_if_threshold_met(successes)

            elif ex.agent_type == "Tool-Calling":
                if termination_triggered:
                    return {}
                _ensure_run_not_cancelled(f"run_branch_{branch_id}:tool_calling")
                tools = [
                    extraction_tool,
                    timeseries_plot_tool,
                    map_plot_tool,
                    review_by_value_tool,
                    review_by_time_tool,
                    review_schema_tool,
                    breach_instr_tool,
                    review_changes_across_period_tool,
                ]
                result = tool_calling_agent(
                    llm=llms["THINKING"],
                    tools=tools,
                    context=state.context,
                    previous_attempts=[p for p in state.executions if p.parallel_agent_id == branch_id and p.retry_number < ex.retry_number],
                )
                updated = ex.model_copy(
                    update={
                        "final_response": result.get("final_response"),
                        "artefacts": result.get("artefacts", []),
                        "error_summary": result.get("error_summary", ""),
                        "is_sufficient": False,
                    }
                )
                msgs = result.get("messages", []) or []
                payload["executions"] = [updated]
                if msgs:
                    payload["messages"] = msgs
                updated_executions = [e if e.parallel_agent_id != branch_id else updated for e in state.executions]
                with termination_lock:
                    prev_retry = counted_branches.get(branch_id, -1)
                    if updated.is_sufficient and updated.final_response and updated.retry_number >= prev_retry:
                        counted_branches[branch_id] = updated.retry_number
                    else:
                        counted_branches.pop(branch_id, None)
                    successes = len(counted_branches)
                _terminate_running_branches_if_threshold_met(successes)

            return payload or {}

        return run_branch

    graph.add_node("history_summariser_messenger_node", lambda state: progress_messenger_node(state, "history_summariser_node"))
    graph.add_node("history_summariser_node", history_summariser_node)
    graph.add_node("context_orchestrator_messenger_node", lambda state: progress_messenger_node(state, "context_orchestrator_node"))
    graph.add_node("context_orchestrator_node", context_orchestrator_node)
    graph.add_node("query_clarifier_node", query_clarifier_node)
    graph.add_node("query_classifier_messenger_node", lambda state: progress_messenger_node(state, "query_classifier_node"))
    graph.add_node("query_classifier_node", query_classifier_node)
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
        {"continue_execution": "query_classifier_messenger_node", "request_clarification": "query_clarifier_node"},
    )
    graph.add_edge("query_clarifier_node", END)
    graph.add_edge("query_classifier_messenger_node", "query_classifier_node")
    graph.add_edge("query_classifier_node", "enter_parallel_execution_node")

    for i in range(num_parallel_executions):
        graph.add_node(f"run_branch_{i}", make_run_branch(i))
        graph.add_edge(f"run_branch_{i}", "exit_parallel_execution_node")

    graph.add_edge("exit_parallel_execution_node", "reporter_messenger_node")
    graph.add_edge("reporter_messenger_node", "response_selector_node")
    graph.add_edge("response_selector_node", "reporter_node")
    graph.add_edge("reporter_node", END)

    return graph.compile()
