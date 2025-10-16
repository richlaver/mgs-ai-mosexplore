from agents.planner_coder_agent import planner_coder_agent
from agents.reporter_agent import reporter_agent
from classes import AgentState
from agents.context_agent import context_agent
from agents.extraction_sandbox_agent import extraction_sandbox_agent

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
    graph.add_node("query_enricher_node", query_enricher_node)
    graph.add_node("planner_coder_messenger_node", lambda state: progress_messenger_node(state, "planner_coder_node"))
    graph.add_node("planner_coder_node", planner_coder_node)
    graph.add_node("code_executor_messenger_node", lambda state: progress_messenger_node(state, "code_executor_node"))
    graph.add_node("code_executor_node", code_executor_node)
    graph.add_node("reporter_messenger_node", lambda state: progress_messenger_node(state, "reporter_node"))
    graph.add_node("reporter_node", reporter_node)

    graph.set_entry_point("query_enricher_node")    
    graph.add_conditional_edges(
        "query_enricher_node",
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