"""Configures the LangGraph workflow for query processing in the MissionHelp Demo.

This module defines the retrieval-augmented generation pipeline, integrating the
language model and vector store.
"""

from typing import Generator

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_tool_calling_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool
)
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor
# from classes import CustomInfoSQLDatabaseTool, InstrTypeTool00, UserPermissionsTool
from classes import CustomInfoSQLDatabaseTool, InstrTypeTool00
import time
import json
import logging

from classes import State
from prompts import prompts
from langchain import hub

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def build_graph(llm, db) -> StateGraph:
    """Builds the LangGraph workflow for query processing.

    Args:
        llm: Language model instance.
        db: SQL database instance.

    Returns:
        Compiled LangGraph instance.
    """
    st.toast("Building LangGraph workflow...", icon=":material/build:")


    def orchestrate_flow(state: State, config: dict) -> Generator[State, None, None]:
        DescribeSQLDatabaseTool = CustomInfoSQLDatabaseTool()
        InferInstrTypeTool = InstrTypeTool00(db=db, llm=llm)
        # GetUserPermissionsTool = UserPermissionsTool(db=db)
        tools=[
            # GetUserPermissionsTool,
            InferInstrTypeTool,
            DescribeSQLDatabaseTool,
            ListSQLDatabaseTool(db=db),
            QuerySQLDatabaseTool(db=db),
            QuerySQLCheckerTool(db=db, llm=llm)
        ]

        system_message = prompts['prompt-001']['content']
        question = state["messages"][-1].content

        agent_executor = create_react_agent(llm, tools, prompt=system_message)

        new_state = state.copy()
        for chunk in agent_executor.stream(
             {'messages': state["messages"]},
             config=config,
             stream_mode='messages'):
            logging.debug('Streamed chunk in generate_response: ')
            logging.debug(chunk)
            if isinstance(chunk, tuple):
                message, metadata = chunk
                content = ''
                
                if isinstance(message, AIMessageChunk):
                    # Handle tool calls
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            args_str = json.dumps(tool_call["args"], ensure_ascii=False)
                            content = f'<strong>Calling Tool</strong>: <code>{tool_call["name"]}</code> with input <code>{args_str}</code>'
                            new_state['messages'] = new_state['messages'] + [
                                AIMessage(content=content, additional_kwargs={"type": "action"})
                            ]
                            logging.debug(f"Tool call: {tool_call['name']} with args: {tool_call['args']}")
                            yield new_state
                    # Handle token-by-token response
                    elif message.content:
                        content = message.content
                        new_state['messages'] = new_state['messages'] + [
                            AIMessage(content=content, additional_kwargs={"type": "output"})
                        ]
                        logging.debug(f"AIMessageChunk content: {content}")
                        yield new_state
                elif isinstance(message, ToolMessage):
                    content = f'<strong>Tool Result</strong>: <code>{message.content}</code>'
                    new_state['messages'] = new_state['messages'] + [
                        AIMessage(content=content, additional_kwargs={"type": "step"})
                    ]
                    yield new_state
                else:
                    logging.warning(f"Unexpected message type in chunk: {type(message)}")
                    content = str(message)
                    new_state['messages'] = new_state['messages'] + [
                        AIMessage(content=content, additional_kwargs={"type": "step"})
                    ]
                    yield new_state
            else:
                logging.warning(f"Unexpected chunk type: {type(chunk)}")
                content = str(chunk)
                new_state['messages'] = new_state['messages'] + [
                    AIMessage(content=content, additional_kwargs={"type": "step"})
                ]
                yield new_state

    graph_builder = StateGraph(State)
    graph_builder.add_node('flow_orchestration', orchestrate_flow)
    graph_builder.set_entry_point('flow_orchestration')
    graph_builder.add_edge('flow_orchestration', END)
    return graph_builder.compile(checkpointer=MemorySaver())
