"""Configures the LangGraph workflow for query processing in the MissionHelp Demo.

This module defines the retrieval-augmented generation pipeline, integrating the
language model and vector store.
"""

from typing import Generator

import streamlit as st
from langchain import hub
from langgraph.types import Command
from langgraph_supervisor import create_supervisor
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    QuerySQLCheckerTool
)
from langchain.agents import AgentExecutor, create_react_agent
from tools.get_user_permissions import UserPermissionsTool
from tools.get_database_schema import CustomInfoSQLDatabaseTool
from tools.sql_security_toolkit import CustomQuerySQLDatabaseTool
from tools.datetime_toolkit import (
    GetDatetimeNowTool,
    DatetimeShiftWrapperTool
)
from typing import Literal
import json
import logging

from classes import State
from prompts import prompts
from parameters import include_tables

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


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


# 2025-08-05 This is a trial implementation for a supervisor agent.
# This has not yet been trialled.
def supervisor_agent(state: State) -> Command[Literal[
    'get_date_range_agent',
    'END'
]]:
    agent_executor = create_react_agent(
        model=st.session_state.llm,
        prompt=prompts['prompt-007']['template']
    )
    response = agent_executor.invoke({'messages': [
        msg for msg in state["messages"] if msg.type == "human" or (msg.type == "ai" and msg.additional_kwargs.get("final") is True)
    ]})
    new_state = state.copy()
    new_state['messages'] = new_state['messages'] + [
        AIMessage(content=response.content, additional_kwargs={"type": "output"})
    ]
    return Command(
        goto=response['next_agent']
    )


def build_supervisor_graph(
        llm: BaseLanguageModel,
        db,
        table_relationship_graph,
        user_id: int,
        global_hierarchy_access: bool = False
    ) -> StateGraph:
    st.toast("Building LangGraph workflow...", icon=":material/build:")


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
            QuerySQLCheckerTool(db=db, llm=llm),
            GetDatetimeNowTool(),
            DatetimeShiftWrapperTool()
        ]

        # Using create_react_agent from langchain.prebuilt
        # get_date_range_agent = create_react_agent(
        #     name='get_date_range_agent',
        #     model=llm,
        #     tools=[
        #         GetDatetimeNowTool(),
        #         DatetimeShiftTool()
        #     ],
        #     prompt=prompts['prompt-006']['content']
        # )

        # Using create_react_agent from langchain.agents
        # prompt = hub.pull("hwchase17/react-json")
        # agent = create_react_agent(
        #     llm=llm,
        #     tools=tools,
        #     prompt=prompt
        # )

#         supervisor = create_supervisor(
#             supervisor_name='supervisor',
#             model=llm,
#             agents=[get_date_range_agent],
#             prompt=("""
# You are a supervisor managing an agent:
# - an agent deriving date ranges from a user query. Assign tasks requiring date 
# ranges to this agent.
# Assign work to one agent at a time, do not call agents in parallel.
# Do not do any work yourself.
#             """),
#             add_handoff_back_messages=True,
#             output_mode="full_history",
#         ).compile()

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

        agent_executor = create_react_agent(model=llm, tools=tools, prompt=system_message)
        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Streaming of the get_date_range_agent
        # for chunk in get_date_range_agent.stream(
        #     {'messages': [
        #         msg for msg in state["messages"] if msg.type == "human" or (msg.type == "ai" and msg.additional_kwargs.get("final") is True)
        #     ]}):
        #     logger.debug("Streamed chunk: %s", chunk)

        # Invocation of the get_date_range_agent from langchain.prebuilt
        response = agent_executor.invoke(
            {'messages': [
                msg for msg in state["messages"] if msg.type == "human" or (msg.type == "ai" and msg.additional_kwargs.get("final") is True)
            ]})
        logger.debug("Response from get_date_range_agent: %s", response)

        new_state = state.copy()

        for message in response['messages']:
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

    graph_builder = StateGraph(State)
    graph_builder.add_node('flow_orchestration', orchestrate_flow)
    graph_builder.set_entry_point('flow_orchestration')
    graph_builder.add_edge('flow_orchestration', END)
    return graph_builder.compile(checkpointer=MemorySaver())


def build_subagent_graph(
        llm: BaseLanguageModel,
        db,
        table_relationship_graph,
        user_id: int,
        global_hierarchy_access: bool = False
    ) -> StateGraph:
    st.toast("Building LangGraph workflow...", icon=":material/build:")


    def test_subagent(state: State, config: dict) -> Generator[State, None, None]:
        custom_query_sql_database_tool = CustomQuerySQLDatabaseTool(
            db=db,
            table_relationship_graph=dict(table_relationship_graph),
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        get_datetime_now_tool = GetDatetimeNowTool()
        datetime_shift_tool = DatetimeShiftWrapperTool()
        tools = [
            custom_query_sql_database_tool,
            get_datetime_now_tool,
            datetime_shift_tool
        ]
        tool_names = [tool.name for tool in tools]
        with open('instrument_context.json', 'r') as instrument_context_json:
            instrument_context = json.load(instrument_context_json)
        custom_info_sql_database_tool = CustomInfoSQLDatabaseTool()
        table_info = custom_info_sql_database_tool.invoke({'table_names': include_tables})

        prompt = hub.pull("hwchase17/react")
        logger.debug(f'hwchase17/react prompt: {prompt}')
        prompt = PromptTemplate.from_template(prompts['prompt-009']['content'])
        logger.debug(f'Prompt for create_react_agent: {prompt}')
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        user_query = state["messages"][-1].content

        # Log input messages for every query
        logger.debug("Processing query: %s", user_query)
        logger.debug("Input messages to agent_executor: %s",
                     json.dumps([msg.__dict__ for msg in state["messages"]], default=str, indent=2))

        filtered_messages = filter_messages(state["messages"])
        logger.debug("Filtered messages: %s",
                     json.dumps([msg.__dict__ for msg in filtered_messages], default=str, indent=2))
        
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        response = agent_executor.invoke(
            input={
                'input': user_query,
                'tools': tools,
                'tool_names': tool_names,
                'instrument_context': instrument_context,
                'table_info': table_info,
                'get_datetime_now_toolname': get_datetime_now_tool.name,
                'add_or_subtract_datetime_toolname': datetime_shift_tool.name,
                'sql_db_query_toolname': custom_query_sql_database_tool.name,
                'agent_scratchpad': ''
            },
            config=config
        )
        logger.debug("Response from get_date_range_agent: %s", response)

        new_state = state.copy()

        for message in response['messages']:
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

    graph_builder = StateGraph(State)
    graph_builder.add_node('subagent_test', test_subagent)
    graph_builder.set_entry_point('subagent_test')
    graph_builder.add_edge('subagent_test', END)
    return graph_builder.compile(checkpointer=MemorySaver())


def build_tool_graph(
        llm: BaseLanguageModel,
        db,
        table_relationship_graph,
        user_id: int,
        global_hierarchy_access: bool = False
    ) -> StateGraph:
    st.toast("Building LangGraph workflow...", icon=":material/build:")


    def test_tool(state: State, config: dict) -> Generator[State, None, None]:
        tool = CustomQuerySQLDatabaseTool(
            db=db,
            table_relationship_graph=dict(table_relationship_graph),
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        tool = DatetimeShiftWrapperTool()
        question = state["messages"][-1].content

        new_state = state.copy()

        # Invoking with JSON string
        response = tool.invoke('''
        {
            "input_datetime": "08 August 2025 07:21:02 AM",
            "operation": "subtract",
            "value": 1,
            "unit": "days"
        }
        ''')

        new_state['messages'] = new_state['messages'] + [
            AIMessage(content=response, additional_kwargs={"type": "output"})
        ]
        yield new_state

    graph_builder = StateGraph(State)
    graph_builder.add_node('tool_test', test_tool)
    graph_builder.set_entry_point('tool_test')
    graph_builder.add_edge('tool_test', END)
    return graph_builder.compile(checkpointer=MemorySaver())
