from agents.history_summariser_agent import history_summariser
from agents.planner_coder_agent import planner_coder_agent
from agents.reporter_agent import reporter_agent
from classes import AgentState, Context
from agents.context_agent import context_agent
from agents.extraction_sandbox_agent import extraction_sandbox_agent
from agents.context_orchestrator import get_context_graph

from langgraph.graph import StateGraph, END

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseLanguageModel

import streamlit as st

from typing import List, Dict
import logging

from modal_sandbox_remote import execute_remote_sandbox
from modal_sandbox_local import execute_local_sandbox
from parameters import progress_messages

logger = logging.getLogger(__name__)

def build_codeact_graph(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    table_info: List[Dict],
    table_relationship_graph: Dict[str, List[tuple]],
    thread_id: str,
    user_id: int,
    global_hierarchy_access: bool,
    remote_sandbox: bool,
) -> StateGraph:
    REMOTE_SANDBOX = remote_sandbox
    st.toast("Building CodeAct graph...", icon=":material/account_tree:")

    def progress_messenger_node(state: AgentState, node: str) -> dict:
        progress_msg = progress_messages.get(node)
        if progress_msg:
            return {"messages": [progress_msg]}
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
            "clarification_request": ""
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
                            # dict union (last-writer-wins per key) but drop Nones
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

    def query_enricher_node(state: AgentState) -> dict:
        return context_agent(llm=llm, chat_history=state.messages, db=db)

    
    def planner_coder_node(state: AgentState) -> dict:
        extraction_tool = extraction_sandbox_agent(
            llm=llm,
            db=db,
            table_info=table_info,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        
        tools = [extraction_tool]
        coder_result = planner_coder_agent(
            llm=llm,
            tools=tools,
            context=state.context,
            previous_attempts=state.coding_attempts
        )
        return {**coder_result}

    def code_executor_node(state: AgentState):
        current_attempt = state.get_current_coding_attempt()
        if not current_attempt:
            error_msg = AIMessage(
                name="CodeExecutor",
                content="Hit an error during execution.",
                additional_kwargs={
                    "stage": "node",
                    "process": "code_executor"
                }
            )
            yield {"messages": [error_msg]}
            return

        code = current_attempt.code
        if not code:
            current_attempt.errors.append("No code to execute.")
            error_msg = AIMessage(
                name="CodeExecutor",
                content="Hit an error during execution.",
                additional_kwargs={
                    "stage": "node",
                    "process": "code_executor"
                }
            )
            yield {"messages": [error_msg]}
            return

        local_outputs = []
        local_errors = []
        try:
            if REMOTE_SANDBOX:
                gen = execute_remote_sandbox(
                    code=code,
                    table_info=table_info,
                    table_relationship_graph=table_relationship_graph,
                    thread_id=thread_id,
                    user_id=user_id,
                    global_hierarchy_access=global_hierarchy_access,
                )
            else:
                gen = execute_local_sandbox(
                    code=code,
                    table_info=table_info,
                    table_relationship_graph=table_relationship_graph,
                    thread_id=thread_id,
                    user_id=user_id,
                    global_hierarchy_access=global_hierarchy_access,
                )

            for output in gen:
                local_outputs.append(output)
                if output.get("type") == "error":
                    local_errors.append(output.get("content", ""))
                if output.get("type") in ["progress", "error"]:
                    progress_msg = AIMessage(
                        name="CodeExecutor",
                        content=output.get("content", ""),
                        additional_kwargs={
                            "stage": "execution_output",
                            "process": "code_executor"
                        }
                    )
                    yield {"messages": [progress_msg]}
        except Exception as e:
            import traceback
            error_message = f"An unexpected error occurred while invoking the sandbox: {e}\n{traceback.format_exc()}"
            local_errors.append(error_message)
            error_msg = AIMessage(
                name="CodeExecutor",
                content=error_message,
                additional_kwargs={
                    "stage": "execution_output",
                    "process": "code_executor"
                }
            )
            yield {"messages": [error_msg]}

        # Update the current attempt with collected outputs/errors
        updated_attempt = current_attempt.model_copy()
        updated_attempt.execution_output = local_outputs
        updated_attempt.errors = local_errors
        updated_coding_attempts = state.coding_attempts[:-1] + [updated_attempt]

        end_msg = AIMessage(
            name="CodeExecutor",
            content="Completed the steps required to answer your query.",
            additional_kwargs={
                "stage": "node",
                "process": "code_executor"
            }
        )
        yield {"messages": [end_msg], "coding_attempts": updated_coding_attempts}
    
    
    def reporter_node(state: AgentState) -> Dict[str, List]:
        base_len = len(state.messages)
        updated_messages_full = reporter_agent(
            llm=llm,
            context=state.context,
            coding_attempts=state.coding_attempts,
            messages=state.messages
        )
        new_messages = updated_messages_full[base_len:]
        return {"messages": new_messages}
    

    def router_after_context(state: AgentState):
        last_ai_message = None
        for message in reversed(state.messages):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if last_ai_message and getattr(last_ai_message, "name", None) == "QueryClarification":
            kwargs = getattr(last_ai_message, "additional_kwargs", None)
            if isinstance(kwargs, dict) and kwargs.get("stage") == "final":
                return "request_clarification"

        return "continue_execution"

    logger.debug("Building CodeAct graph with tables=%d relationship_nodes=%d user_id=%s", len(table_info), len(table_relationship_graph or {}), user_id)
    graph = StateGraph(AgentState)
    graph.add_node("history_summariser_messenger_node", lambda state: progress_messenger_node(state, "history_summariser_node"))
    graph.add_node("history_summariser_node", history_summariser_node)
    graph.add_node("context_orchestrator_messenger_node", lambda state: progress_messenger_node(state, "context_orchestrator_node"))
    graph.add_node("context_orchestrator_node", context_orchestrator_node)
    graph.add_node("query_enricher_node", query_enricher_node)
    graph.add_node("planner_coder_messenger_node", lambda state: progress_messenger_node(state, "planner_coder_node"))
    graph.add_node("planner_coder_node", planner_coder_node)
    graph.add_node("code_executor_messenger_node", lambda state: progress_messenger_node(state, "code_executor_node"))
    graph.add_node("code_executor_node", code_executor_node)
    graph.add_node("reporter_messenger_node", lambda state: progress_messenger_node(state, "reporter_node"))
    graph.add_node("reporter_node", reporter_node)

    graph.set_entry_point("history_summariser_messenger_node")
    graph.add_edge("history_summariser_messenger_node", "history_summariser_node")
    graph.add_edge("history_summariser_node", "context_orchestrator_messenger_node")
    graph.add_edge("context_orchestrator_messenger_node", "context_orchestrator_node")
    graph.add_conditional_edges(
        "context_orchestrator_node",
        router_after_context,
        {
            "continue_execution": "planner_coder_messenger_node",
            "request_clarification": END,
        }
    )
    graph.add_edge("planner_coder_messenger_node", "planner_coder_node")
    graph.add_edge("planner_coder_node", "code_executor_messenger_node")
    graph.add_edge("code_executor_messenger_node", "code_executor_node")
    graph.add_edge("code_executor_node", "reporter_messenger_node")
    graph.add_edge("reporter_messenger_node", "reporter_node")
    graph.add_edge("reporter_node", END)
    
    return graph.compile()