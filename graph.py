from agents.history_summariser_agent import history_summariser
from agents.codeact_coder_agent import codeact_coder_agent
from agents.reporter_agent import reporter_agent
from agents.response_selector import response_selector
from classes import AgentState, Context, Execution
from agents.extraction_sandbox_agent import extraction_sandbox_agent
from agents.timeseries_plot_sandbox_agent import timeseries_plot_sandbox_agent
from agents.map_plot_sandbox_agent import map_plot_sandbox_agent
from tools.sql_security_toolkit import GeneralSQLQueryTool
from tools.artefact_toolkit import WriteArtefactTool
from agents.context_orchestrator import get_context_graph
from agents.query_clarifier import query_clarifier_agent
from agents.query_classifier import query_classifier_agent
from agents.react_agent import react_agent
from agents.tool_calling_agent import tool_calling_agent
from utils.chat_history import filter_messages_only_final

from langgraph.graph import StateGraph, END
from langgraph.types import Command, Send

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseLanguageModel

import streamlit as st
import b2sdk.v1 as b2
import psycopg2

from typing import List, Dict, Any
import logging

from parameters import progress_messages

import json
import traceback
from modal_sandbox_remote import execute_remote_sandbox
from modal_sandbox_local import execute_local_sandbox

logger = logging.getLogger(__name__)

def build_graph(
    llm: BaseLanguageModel,
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

    extraction_tool = extraction_sandbox_agent(
        llm=llm,
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
        llm=llm,
        sql_tool=_general_sql_query_tool,
        write_artefact_tool=_write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )
    map_plot_tool = map_plot_sandbox_agent(
        llm=llm,
        sql_tool=_general_sql_query_tool,
        write_artefact_tool=_write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )

    def progress_messenger_node(state: AgentState, node: str) -> dict:
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
        retrospective_query = history_summariser(
            messages=state.messages,
            llm=llm
        )
        new_context = state.context.model_copy(update={"retrospective_query": retrospective_query})
        return {"context": new_context}

    def context_orchestrator_node(state: AgentState):
        base_context_dict = {}
        if isinstance(state.context, Context):
            base_context_dict = state.context.model_dump(exclude_none=True)
        elif isinstance(state.context, dict):
            base_context_dict = state.context
        sub_input = {
            "messages": state.messages,
            "context": base_context_dict,
            "clarification_requests": []
        }
        logger.info(f'Starting context in context_orchestrator_node with context: {state.context}')

        sub_graph = get_context_graph(llm, db)
        accumulated_context = base_context_dict
        for sub_chunk in sub_graph.stream(sub_input, stream_mode="updates"):
            logger.info(f'Context Orchestrator sub-chunk: {sub_chunk}')
            for node, context_update in sub_chunk.items():
                if context_update:
                    update_model = context_update.get("context")
                    if update_model is not None:
                        try:
                            if not isinstance(accumulated_context, dict):
                                accumulated_context = (
                                    accumulated_context.model_dump(exclude_none=True)
                                    if isinstance(accumulated_context, Context)
                                    else {}
                                )
                            update_dict = (
                                update_model.model_dump(exclude_none=True)
                                if isinstance(update_model, Context)
                                else update_model if isinstance(update_model, dict)
                                else {}
                            )
                            update_dict = {k: v for k, v in update_dict.items() if v is not None}
                            accumulated_context = {**accumulated_context, **update_dict}
                        except Exception as e:
                            logger.error(f"Failed to merge context update from node {node}: {e}")
                logger.info(f'Context Orchestrator update from node {node}: {context_update}')
                logger.info(f'Accumulated context so far: {accumulated_context}')

        try:
            merged_context_model = Context.model_validate(accumulated_context)
        except Exception as e:
            logger.error(f"Failed to validate merged context dict into Context model: {e}")
            merged_context_model = state.context if isinstance(state.context, Context) else None
        yield {"context": merged_context_model}

    def router_after_context(state: AgentState):
        if state.context and state.context.clarification_requests:
            return "request_clarification"
        return "continue_execution"
    
    def query_classifier_node(state: AgentState) -> dict:
        classified_agent_type = query_classifier_agent(llm=llm, messages=state.messages, context=state.context) if agent_type == "Auto" else agent_type
        logger.info(f"Classified agent type: {classified_agent_type}")

        new_executions = []
        for i in range(num_parallel_executions):
            execution = Execution(
                agent_type=classified_agent_type,
                parallel_agent_id=i,
                retry_number=0,
                codeact_code="",
                final_response=None,
                artefacts=[],
                error_summary=""
            )
            new_executions.append(execution)
        logger.info(f"Created new executions: {new_executions}")

        return {"executions": new_executions}

    def query_clarifier_node(state: AgentState) -> dict:
        clarification_requests = state.context.clarification_requests if state.context else []
        chat_history = filter_messages_only_final(state.messages)
        message = query_clarifier_agent(llm=llm, clarification_requests=clarification_requests, chat_history=chat_history)
        return {"messages": [message]}
    
    def enter_parallel_execution_node(state: AgentState):
        logger.info("[MainGraph] Entering parallel execution (inline branches): %d", num_parallel_executions)
        targets = [Send(f"run_branch_{i}", state) for i in range(num_parallel_executions)]
        return Command(goto=targets)
    
    def exit_parallel_execution_node(state: AgentState):
        logger.info("[MainGraph] Exiting parallel execution (inline branches)")
        return {}
    
    def response_selector_node(state: AgentState) -> dict:
        updated_executions = response_selector(llm=llm, executions=state.executions, context=state.context)
        return {"executions": updated_executions}
        
    def reporter_node(state: AgentState) -> Dict[str, List]:
        base_len = len(state.messages)
        updated_messages_full = reporter_agent(
            llm=llm,
            context=state.context,
            messages=state.messages,
            executions=state.executions
        )
        new_messages = updated_messages_full[base_len:]
        return {"messages": new_messages}
    
    graph = StateGraph(AgentState)

    def make_run_branch(branch_id: int):
        def _progress_msg(content: str, process: str = "progress") -> AIMessage:
            return AIMessage(name=f"Executor_{branch_id}", content=str(content), additional_kwargs={"stage": "execution_output", "process": process})

        def run_branch(state: AgentState):
            ex = next((e for e in state.executions if e.parallel_agent_id == branch_id), None)
            if not ex or ex.final_response is not None:
                return {}

            payload: Dict[str, Any] = {}

            if ex.agent_type == "CodeAct":
                coder_result = codeact_coder_agent(
                    llm=llm,
                    tools=[extraction_tool, timeseries_plot_tool, map_plot_tool],
                    context=state.context,
                    previous_attempts=[p for p in state.executions if p.parallel_agent_id == branch_id and p.retry_number < ex.retry_number],
                )
                new_execution = coder_result["executions"][0]
                code = (new_execution.codeact_code or "").strip()
                updated = ex.model_copy(update={"codeact_code": code})

                messages: List[AIMessage] = []
                artefacts: List[AIMessage] = []
                final_msg: AIMessage | None = None
                logs: List[str] = []
                if code:
                    try:
                        gen = (execute_remote_sandbox if remote_sandbox else execute_local_sandbox)(
                            code=code,
                            table_info=table_info,
                            table_relationship_graph=table_relationship_graph,
                            thread_id=thread_id,
                            user_id=user_id,
                            global_hierarchy_access=global_hierarchy_access,
                            selected_project_key=selected_project_key,
                        )
                        for out in gen:
                            if not isinstance(out, dict) or "metadata" not in out:
                                logs.append(f"Invalid output: {out}")
                                continue
                            typ = out["metadata"].get("type")
                            content = out.get("content", "")
                            if typ in ("progress", "error"):
                                messages.append(_progress_msg(str(content), process=typ))
                                logs.append(str(content))
                            elif typ == "final":
                                final_msg = _progress_msg(str(content), process="final")
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
                                    logs.append(err)
                                    messages.append(_progress_msg(err, process="error"))
                            else:
                                messages.append(_progress_msg(str(content), process="progress"))
                    except Exception as e:
                        err = f"Sandbox error: {e}"
                        logs.append(err)
                        messages.append(_progress_msg(err, process="error"))

                updated = updated.model_copy(update={
                    "final_response": final_msg,
                    "artefacts": artefacts,
                    "error_summary": "\n".join(logs) if logs else "",
                    "is_sufficient": bool(final_msg),
                })
                payload["executions"] = [updated]
                if messages:
                    payload["messages"] = messages

            elif ex.agent_type == "ReAct":
                tools = [extraction_tool, timeseries_plot_tool, map_plot_tool]
                result = react_agent(
                    llm=llm,
                    tools=tools,
                    context=state.context,
                    previous_attempts=[p for p in state.executions if p.parallel_agent_id == branch_id and p.retry_number < ex.retry_number],
                )
                updated = ex.model_copy(update={
                    "final_response": result.get("final_response"),
                    "artefacts": result.get("artefacts", []),
                    "error_summary": result.get("error_summary", ""),
                    "is_sufficient": bool(result.get("final_response")),
                })
                msgs = result.get("messages", []) or []
                payload["executions"] = [updated]
                if msgs:
                    payload["messages"] = msgs

            elif ex.agent_type == "Tool-Calling":
                tools = [extraction_tool, timeseries_plot_tool, map_plot_tool]
                result = tool_calling_agent(
                    llm=llm,
                    tools=tools,
                    context=state.context,
                    previous_attempts=[p for p in state.executions if p.parallel_agent_id == branch_id and p.retry_number < ex.retry_number],
                )
                updated = ex.model_copy(update={
                    "final_response": result.get("final_response"),
                    "artefacts": result.get("artefacts", []),
                    "error_summary": result.get("error_summary", ""),
                    "is_sufficient": bool(result.get("final_response")),
                })
                msgs = result.get("messages", []) or []
                payload["executions"] = [updated]
                if msgs:
                    payload["messages"] = msgs

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