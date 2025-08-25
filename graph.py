"""Configures the LangGraph workflow for query processing in the MissionHelp Demo.

This module defines the retrieval-augmented generation pipeline, integrating the
language model and vector store.
"""

from typing import Generator

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    QuerySQLCheckerTool
)
from langchain.agents import AgentExecutor, create_react_agent
from tools.get_database_schema import CustomInfoSQLDatabaseTool
from tools.sql_security_toolkit import GeneralSQLQueryTool
from tools.datetime_toolkit import (
    GetDatetimeNowTool,
    DatetimeShiftWrapperTool
)
# Proxy instrument context tool
from tools.get_instrument_context import InstrumentContextTool
# Sandeep's prototype instrument context tool
# from tools.get_instrument_context_20250818 import EnhancedInstrumentContextTool
from tools.get_trend_info_toolkit import TrendExtractorWrapperTool
from tools.create_output_toolkit import (
    TimeSeriesPlotWrapperTool,
    TimeSeriesPlotTool
)
import json
import logging
import re

from classes import State
from prompts import prompts
from parameters import include_tables, trend_context

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def filter_messages(messages):
    """Filter messages to include only HumanMessages and AIMessages with type='final'."""
    filtered = []
    for msg in messages:
        # Keep HumanMessages (user queries)
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        # Keep AIMessages with type='final' (Final Answer from ReAct agent)
        elif isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") == "final":
            filtered.append(msg)
        else:
            logger.debug("Filtered out message: %s", json.dumps(msg.__dict__, default=str))
    logger.debug("Filtered messages: %s",
                 json.dumps([msg.__dict__ for msg in filtered], default=str, indent=2))
    return filtered


def parse_thought_from_log(log: str) -> str:
    """Extract the thought from an agent's action log.
    
    Args:
        log: The log string from an agent action
        
    Returns:
        The extracted thought, or the original log if no thought pattern is found
    """
    thought_match = re.search(r'(.*?)(?=\nAction:)', log, re.DOTALL)
    return thought_match.group(1).strip() if thought_match else log


def build_supervisor_graph(
        llm: BaseLanguageModel,
        db,
        table_relationship_graph,
        user_id: int,
        global_hierarchy_access: bool = False
    ) -> StateGraph:
    st.toast("Building LangGraph workflow...", icon=":material/build:")


    def test_supervisor(state: State, config: dict) -> Generator[State, None, None]:
        general_sql_query_tool = GeneralSQLQueryTool(
            db=db,
            table_relationship_graph=dict(table_relationship_graph),
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        get_datetime_now_tool = GetDatetimeNowTool()
        datetime_shift_tool = DatetimeShiftWrapperTool()
        # Using proxy instrument context tool
        get_instrument_context_tool = InstrumentContextTool()
        # Using Sandeep's prototype instrument context tool
        # get_instrument_context_tool = EnhancedInstrumentContextTool(
        #     llm=llm,
        #     json_file_path="instrument_context_20250818.json"
        # )
        get_trend_info_tool = TrendExtractorWrapperTool(
            sql_tool=general_sql_query_tool
        )
        plot_time_series_tool = TimeSeriesPlotTool(
            sql_tool=general_sql_query_tool
        )
        plot_time_series_tool = TimeSeriesPlotWrapperTool(
            plot_tool=TimeSeriesPlotTool(sql_tool=general_sql_query_tool)
        )
        tools = [
            general_sql_query_tool,
            get_datetime_now_tool,
            datetime_shift_tool,
            get_instrument_context_tool,
            get_trend_info_tool,
            plot_time_series_tool
        ]
        tool_names = [tool.name for tool in tools]
        with open('instrument_context.json', 'r') as instrument_context_json:
            instrument_context = json.load(instrument_context_json)
        custom_info_sql_database_tool = CustomInfoSQLDatabaseTool()
        table_info = custom_info_sql_database_tool.invoke({'table_names': include_tables})

        # The following two lines of code are to view the format of the 
        # LangChain ReAct prompt.
        # prompt = hub.pull("hwchase17/react")
        # logger.debug(f'hwchase17/react prompt: {prompt}')
        prompt = PromptTemplate.from_template(prompts['prompt-010']['content'])
        logger.debug(f'Prompt for create_react_agent: {prompt}')
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        user_query = state["messages"][-1].content
        chat_history = "\n".join([
            f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in state["messages"][:-1]
        ])

        # Log input messages for every query
        logger.debug("Processing query: %s", user_query)
        logger.debug("Input messages to agent_executor: %s",
                     json.dumps([msg.__dict__ for msg in state["messages"]], default=str, indent=2))

        filtered_messages = filter_messages(state["messages"])
        logger.debug("Filtered messages: %s",
                     json.dumps([msg.__dict__ for msg in filtered_messages], default=str, indent=2))
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

        for chunk in agent_executor.stream(
            input={
                'chat_history': chat_history,
                'input': user_query,
                'tools': tools,
                'tool_names': tool_names,
                'table_info': table_info,
                'trend_context': trend_context,
                'get_datetime_now_toolname': get_datetime_now_tool.name,
                'add_or_subtract_datetime_toolname': datetime_shift_tool.name,
                'general_sql_query_toolname': general_sql_query_tool.name,
                'get_instrument_context_toolname': 
                get_instrument_context_tool.name,
                'get_trend_info_toolname': get_trend_info_tool.name,
                'plot_time_series_toolname': plot_time_series_tool.name,
                'agent_scratchpad': ''
            },
            config=config
        ):
            logger.debug("Received chunk in graph.py: %s", chunk)
            
            if 'steps' in chunk:
                for step in chunk['steps']:
                    logger.debug("Processing step in chunk: %s", json.dumps(step, default=str))
                    action_msg = AIMessage(
                        content=f"[ACTION]:{step.action.tool}"
                    )
                    logger.debug("Created action_msg: %s", action_msg)
                    action_input_msg = AIMessage(
                        content=f"[ACTION_INPUT]:{step.action.tool_input}"
                    )
                    logger.debug("Created action_input_msg: %s", action_input_msg)
                    observation_content = (
                        json.dumps(step.observation, default=str) 
                        if hasattr(step.observation, '__dict__') 
                        else str(step.observation)
                    )
                    observation_msg = AIMessage(
                        content=f"[OBSERVATION]:{observation_content}"
                    )
                    logger.debug("Created observation_msg: %s", observation_msg)
                    yield State(messages=state['messages'] + [
                        action_msg,
                        action_input_msg,
                        observation_msg
                    ])
            
            if 'messages' in chunk:
                for msg in chunk['messages']:
                    logger.debug("Processing message in chunk: %s", json.dumps(msg, default=str))
                    thought_msg = AIMessage(
                        content=f"[THOUGHT]:{msg.content}"
                    )
                    logger.debug("Created thought_msg: %s", thought_msg)
                    yield State(messages=state['messages'] + [thought_msg])

            if 'actions' in chunk:
                for action in chunk['actions']:
                    logger.debug("Processing action in chunk: %s", json.dumps(action, default=str))
                    action_msg = AIMessage(
                        content=action.tool,
                        additional_kwargs={"type": "action"}
                    )
                    logger.debug("Created action_msg: %s", action_msg)
                    action_input_msg = AIMessage(
                        content=action.tool_input,
                        additional_kwargs={"type": "action_input"}
                    )
                    logger.debug("Created action_input_msg: %s", action_input_msg)
                    yield State(messages=state['messages'] + [
                        action_msg,
                        action_input_msg,
                        thought_msg
                    ])

            if 'output' in chunk:
                logger.debug("Processing output in chunk: %s", json.dumps(chunk['output'], default=str))
                final_msg = AIMessage(
                    content=f"[FINAL]:{chunk['output']}"
                )
                logger.debug("Created final_msg: %s", final_msg)
                yield State(messages=state['messages'] + [final_msg])

    graph_builder = StateGraph(State)
    graph_builder.add_node('supervisor_test', test_supervisor)
    graph_builder.set_entry_point('supervisor_test')
    graph_builder.add_edge('supervisor_test', END)
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
        tool = GeneralSQLQueryTool(
            db=db,
            table_relationship_graph=dict(table_relationship_graph),
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )
        tool = DatetimeShiftWrapperTool()
        question = state["messages"][-1].content

        new_state = state.copy()

        thought = f"Invoking tool {tool.name}"
        action = f"{tool.name}"
        action_input = '''
{
    "input_datetime": "08 August 2025 07:21:02 AM",
    "operation": "subtract",
    "value": 1,
    "unit": "days"
}
        '''
        response = tool.invoke(action_input)
        observation = f"Tool Result: {response}"
        final_answer = response

        # Yield each component as a separate message
        components = [
            ("thought", thought),
            ("action", action),
            ("action_input", action_input),
            ("observation", observation),
            ("final", final_answer)
        ]

        for msg_type, msg_content in components:
            new_state['messages'] = new_state['messages'] + [
                AIMessage(content=msg_content, additional_kwargs={"type": msg_type})
            ]
            yield new_state

    graph_builder = StateGraph(State)
    graph_builder.add_node('tool_test', test_tool)
    graph_builder.set_entry_point('tool_test')
    graph_builder.add_edge('tool_test', END)
    return graph_builder.compile(checkpointer=MemorySaver())
