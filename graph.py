"""Configures the LangGraph workflow for query processing in the MissionHelp Demo.

This module defines the retrieval-augmented generation pipeline, integrating the
language model and vector store.
"""

from typing import Generator

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    QuerySQLCheckerTool
)
from langgraph.prebuilt import create_react_agent
from tools.get_user_permissions import UserPermissionsTool, UserPermissionsToolOutput
from tools.get_database_schema import CustomInfoSQLDatabaseTool
from tools.write_sql_with_permissions import CustomQuerySQLDatabaseTool
import json
import logging

from classes import State
from prompts import prompts

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def build_graph(
        llm,
        db,
        table_relationship_graph,
        user_id: int,
        global_hierarchy_access: bool = False
    ) -> StateGraph:
    """Builds the LangGraph workflow for query processing.

    Args:
        llm: Language model instance.
        db: SQL database instance.

    Returns:
        Compiled LangGraph instance.
    """
    st.toast("Building LangGraph workflow...", icon=":material/build:")


    def filter_messages(messages):
        """Filter messages to include only HumanMessages and AIMessages with final=True."""
        filtered = []
        for msg in messages:
            # Keep HumanMessages (queries) and AIMessages with final=True
            if msg.type == "human" or (msg.type == "ai" and msg.additional_kwargs.get("final") is True):
                filtered.append(msg)
            else:
                logger.debug("Filtered out message: %s", json.dumps(msg.__dict__, default=str))
        return filtered

    def orchestrate_flow(state: State, config: dict) -> Generator[State, None, None]:
        tools = [
            UserPermissionsTool(db=db),
            CustomInfoSQLDatabaseTool(),
            ListSQLDatabaseTool(db=db),
            CustomQuerySQLDatabaseTool(
                db=db,
                table_relationship_graph=dict(table_relationship_graph),
                user_id=user_id,
                global_hierarchy_access=global_hierarchy_access
            ),
            QuerySQLCheckerTool(db=db, llm=llm)
        ]

        system_message = prompts['prompt-001']['content']
        question = state["messages"][-1].content

        # Log input messages for every query
        logger.debug("Processing query: %s", question)
        logger.debug("Input messages to agent_executor: %s",
                     json.dumps([msg.__dict__ for msg in state["messages"]], default=str, indent=2))

        # Filter messages to include only queries and final responses
        filtered_messages = filter_messages(state["messages"])
        logger.debug("Filtered messages: %s",
                     json.dumps([msg.__dict__ for msg in filtered_messages], default=str, indent=2))

        agent_executor = create_react_agent(llm, tools, prompt=system_message)

        new_state = state.copy()
        try:
            for chunk in agent_executor.stream(
                 {'messages': [
                     msg for msg in state["messages"] if msg.type == "human" or (msg.type == "ai" and msg.additional_kwargs.get("final") is True)
                    ]},
                 config=config,
                 stream_mode='messages'):
                logger.debug("Streamed chunk: %s", json.dumps(chunk, default=str, indent=2))
                if isinstance(chunk, tuple):
                    message, metadata = chunk
                    logger.debug("Message type: %s", type(message).__name__)
                    logger.debug("Message content: %s", message.content)
                    logger.debug("Message content type: %s", type(message.content).__name__)
                    logger.debug("Message additional_kwargs: %s", message.additional_kwargs)
                    if hasattr(message, 'tool_calls'):
                        logger.debug("Message tool_calls: %s", message.tool_calls)
                    content = ''

                    if isinstance(message, AIMessageChunk):
                        if message.tool_calls:
                            for tool_call in message.tool_calls:
                                args_str = json.dumps(tool_call["args"], ensure_ascii=False)
                                content = f'<strong>Calling Tool</strong>: <code>{tool_call["name"]}</code> with input <code>{args_str}</code>'
                                new_state['messages'] = new_state['messages'] + [
                                    AIMessage(content=content, additional_kwargs={"type": "action"})
                                ]
                                logger.debug("Tool call: %s with args: %s", tool_call['name'], tool_call['args'])
                                yield new_state
                        elif message.content:
                            content = message.content
                            new_state['messages'] = new_state['messages'] + [
                                AIMessage(content=content, additional_kwargs={"type": "output"})
                            ]
                            logger.debug("AIMessageChunk content: %s", content)
                            yield new_state
                    elif isinstance(message, ToolMessage):
                        content = f'<strong>Tool Result</strong>: <code>{message.content}</code>'
                        new_state['messages'] = new_state['messages'] + [
                            AIMessage(content=content, additional_kwargs={"type": "step"})
                        ]
                        logger.debug("ToolMessage content: %s", content)
                        yield new_state
                    else:
                        logger.warning("Unexpected message type in chunk: %s", type(message).__name__)
                        content = str(message.content)
                        new_state['messages'] = new_state['messages'] + [
                            AIMessage(content=content, additional_kwargs={"type": "step"})
                        ]
                        yield new_state
                else:
                    logger.warning("Unexpected chunk type: %s", type(chunk).__name__)
                    content = str(chunk)
                    new_state['messages'] = new_state['messages'] + [
                        AIMessage(content=content, additional_kwargs={"type": "step"})
                    ]
                    yield new_state
        except KeyError as e:
            logger.error("KeyError occurred during chunk generation: %s", str(e))
            logger.error("Last processed state: %s", json.dumps(new_state, default=str, indent=2))
            logger.error("Input messages at time of error: %s",
                         json.dumps([msg.__dict__ for msg in filtered_messages], default=str, indent=2))
            new_state['messages'].append(
                AIMessage(content="Error processing query. Please try again.", additional_kwargs={"type": "error"})
            )
            yield new_state

        # agent_executor = create_react_agent(llm, tools, prompt=system_message)

        # new_state = state.copy()
        # for chunk in agent_executor.stream(
        #      {'messages': state["messages"]},
        #      config=config,
        #      stream_mode='messages'):
        #     logging.debug('Streamed chunk in generate_response: ')
        #     logging.debug(chunk)
        #     if isinstance(chunk, tuple):
        #         message, metadata = chunk
        #         # Log detailed information about the message
        #         logger.debug("Message type: %s", type(message).__name__)
        #         logger.debug("Message content: %s", message.content)
        #         logger.debug("Message content type: %s", type(message.content).__name__)
        #         logger.debug("Message additional_kwargs: %s", message.additional_kwargs)
        #         if hasattr(message, 'tool_calls'):
        #             logger.debug("Message tool_calls: %s", message.tool_calls)
        #         content = ''
                
        #         if isinstance(message, AIMessageChunk):
        #             if message.tool_calls:
        #                 for tool_call in message.tool_calls:
        #                     args_str = json.dumps(tool_call["args"], ensure_ascii=False)
        #                     content = f'<strong>Calling Tool</strong>: <code>{tool_call["name"]}</code> with input <code>{args_str}</code>'
        #                     new_state['messages'] = new_state['messages'] + [
        #                         AIMessage(content=content, additional_kwargs={"type": "action"})
        #                     ]
        #                     logging.debug(f"Tool call: {tool_call['name']} with args: {tool_call['args']}")
        #                     yield new_state
        #             elif message.content:
        #                 content = message.content
        #                 new_state['messages'] = new_state['messages'] + [
        #                     AIMessage(content=content, additional_kwargs={"type": "output"})
        #                 ]
        #                 logging.debug(f"AIMessageChunk content: {content}")
        #                 yield new_state
        #         elif isinstance(message, ToolMessage):
        #             content = f'<strong>Tool Result</strong>: <code>{message.content}</code>'
        #             new_state['messages'] = new_state['messages'] + [
        #                 AIMessage(content=content, additional_kwargs={"type": "step"})
        #             ]
        #             logger.debug("ToolMessage content: %s", content)
        #             yield new_state
        #         else:
        #             logging.warning(f"Unexpected message type in chunk: {type(message)}")
        #             content = str(message)
        #             new_state['messages'] = new_state['messages'] + [
        #                 AIMessage(content=content, additional_kwargs={"type": "step"})
        #             ]
        #             yield new_state
        #     else:
        #         logging.warning(f"Unexpected chunk type: {type(chunk)}")
        #         content = str(chunk)
        #         new_state['messages'] = new_state['messages'] + [
        #             AIMessage(content=content, additional_kwargs={"type": "step"})
        #         ]
        #         yield new_state

    graph_builder = StateGraph(State)
    graph_builder.add_node('flow_orchestration', orchestrate_flow)
    graph_builder.set_entry_point('flow_orchestration')
    graph_builder.add_edge('flow_orchestration', END)
    return graph_builder.compile(checkpointer=MemorySaver())
