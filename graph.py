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
from langchain.agents import AgentExecutor, create_structured_chat_agent as create_openai_tools_agent
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    QuerySQLCheckerTool
)
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.structured_chat.base import create_structured_chat_agent
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
import asyncio

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
        plot_wrapper_tool = TimeSeriesPlotWrapperTool(
            plot_tool=plot_time_series_tool
        )
        tools = [
            general_sql_query_tool,
            get_datetime_now_tool,
            datetime_shift_tool,
            get_instrument_context_tool,
            get_trend_info_tool,
            plot_wrapper_tool
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

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process_events():
            async for event in agent_executor.astream_events(
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
                    'plot_time_series_toolname': plot_wrapper_tool.name,
                    'agent_scratchpad': ''
                },
                config=config
            ):
                # logger.debug("Received event in graph.py: %s", event)

                # Initialize new_messages for this event
                new_messages = []

                # Handle on_tool_start
                if event['event'] == 'on_tool_start':
                    logger.debug("Processing on_tool_start event: %s", json.dumps(event, default=str))
                    tool_name = event['name']
                    tool_input = json.dumps(event['data']['input'], default=str)
                    new_messages.extend([
                        AIMessage(
                            content=f"[ACTION]:{tool_name}",
                            additional_kwargs={"type": "action"}
                        ),
                        AIMessage(
                            content=f"[ACTION_INPUT]:{tool_input}",
                            additional_kwargs={"type": "action_input"}
                        )
                    ])
                    logger.debug("Created action messages: %s", new_messages)

                # Handle on_tool_end
                elif event['event'] == 'on_tool_end':
                    logger.debug("Processing on_tool_end event: %s", json.dumps(event, default=str))
                    tool_name = event['name']
                    observation = event['data']['output']
                    additional_kwargs = {}
                    obs_content = str(observation)
                    
                    # Handle JSON string or tuple for time_series_plot and time_series_plot_wrapper
                    if tool_name in ['time_series_plot', 'time_series_plot_wrapper']:
                        if isinstance(observation, str):
                            try:
                                parsed = json.loads(observation)
                                if isinstance(parsed, dict) and 'content' in parsed and 'artifacts' in parsed:
                                    obs_content = parsed['content']
                                    additional_kwargs = {'artifacts': parsed['artifacts']}
                                else:
                                    logger.debug(f"Observation not in expected format: {observation}")
                            except json.JSONDecodeError:
                                logger.debug(f"Observation not a valid JSON: {observation}")
                        elif isinstance(observation, tuple) and len(observation) == 2:
                            # Handle tuple (content, artifacts)
                            obs_content, artifacts = observation
                            obs_content = str(obs_content)
                            if isinstance(artifacts, list):
                                additional_kwargs = {'artifacts': artifacts}
                            else:
                                logger.debug(f"Artifacts not a list: {artifacts}")
                            logger.debug("Handled tuple observation")
                        elif isinstance(observation, list) and len(observation) == 2:
                            # Handle list ["Error message", []] as in debug output
                            obs_content, artifacts = observation
                            obs_content = str(obs_content)
                            if isinstance(artifacts, list):
                                additional_kwargs = {'artifacts': artifacts}
                            else:
                                logger.debug(f"Artifacts not a list: {artifacts}")
                            logger.debug("Handled list observation for error")
                        else:
                            logger.debug(f"Unexpected observation type: {type(observation)}")
                    
                    new_messages.append(
                        AIMessage(
                            content=f"[OBSERVATION]:{obs_content}",
                            additional_kwargs=additional_kwargs
                        )
                    )
                    logger.debug("Created observation message: %s", new_messages[-1])

                # Handle on_chat_model_stream
                elif event['event'] == 'on_chat_model_stream':
                    logger.debug("Processing on_chat_model_stream event: %s", json.dumps(event, default=str))
                    chunk = event['data']['chunk']
                    if isinstance(chunk, AIMessageChunk):
                        thought_content = parse_thought_from_log(chunk.content)
                        new_messages.append(
                            AIMessage(
                                content=f"[THOUGHT]:{thought_content}",
                                additional_kwargs={"type": "thought"}
                            )
                        )
                        logger.debug("Created thought message: %s", new_messages[-1])

                # Handle on_chat_model_end
                elif event['event'] == 'on_chat_model_end':
                    logger.debug("Processing on_chat_model_end event: %s", json.dumps(event, default=str))
                    output = event['data']['output']
                    if isinstance(output, AIMessageChunk):
                        thought_content = parse_thought_from_log(output.content)
                        new_messages.append(
                            AIMessage(
                                content=f"[THOUGHT]:{thought_content}",
                                additional_kwargs={"type": "thought"}
                            )
                        )
                        logger.debug("Created thought message: %s", new_messages[-1])

                # Handle on_chain_end
                elif event['event'] == 'on_chain_end':
                    logger.debug("Processing on_chain_end event: %s", json.dumps(event['data']['output'], default=str))
                    output = event['data']['output']
                    if isinstance(output, dict) and 'output' in output:
                        final_content = output['output']
                        new_messages.append(
                            AIMessage(
                                content=f"[FINAL]:{final_content}",
                                additional_kwargs={"type": "final"}
                            )
                        )
                        logger.debug("Created final message: %s", new_messages[-1])

                # Yield a State dictionary with the new messages
                if new_messages:
                    yield State(messages=state['messages'] + new_messages)
                    state['messages'] = state['messages'] + new_messages

        async def run_process_events():
            async for state_update in process_events():
                yield state_update

        try:
            async_gen = run_process_events()
            while True:
                try:
                    state_update = loop.run_until_complete(async_gen.__anext__())
                    yield state_update
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

# The code below works for streaming chunks using the .stream method of agent_executor with the create_react_agent from the agents import.
# The code below is commented to trial the .astream_events method. This is because this async method promises to yield tool calls which are needed to access the tool artifacts, including plots and CSVs.
        # for chunk in agent_executor.stream(
        #     input={
        #         'chat_history': chat_history,
        #         'input': user_query,
        #         'tools': tools,
        #         'tool_names': tool_names,
        #         'table_info': table_info,
        #         'trend_context': trend_context,
        #         'get_datetime_now_toolname': get_datetime_now_tool.name,
        #         'add_or_subtract_datetime_toolname': datetime_shift_tool.name,
        #         'general_sql_query_toolname': general_sql_query_tool.name,
        #         'get_instrument_context_toolname': 
        #         get_instrument_context_tool.name,
        #         'get_trend_info_toolname': get_trend_info_tool.name,
        #         'plot_time_series_toolname': plot_time_series_tool.name,
        #         'agent_scratchpad': ''
        #     },
        #     config=config
        # ):
        #     logger.debug("Received chunk in graph.py: %s", chunk)
            
        #     if 'steps' in chunk:
        #         for step in chunk['steps']:
        #             logger.debug("Processing step in chunk: %s", json.dumps(step, default=str))
        #             action_msg = AIMessage(
        #                 content=f"[ACTION]:{step.action.tool}"
        #             )
        #             logger.debug("Created action_msg: %s", action_msg)
        #             action_input_msg = AIMessage(
        #                 content=f"[ACTION_INPUT]:{step.action.tool_input}"
        #             )
        #             logger.debug("Created action_input_msg: %s", action_input_msg)
        #             observation_content = (
        #                 json.dumps(step.observation, default=str) 
        #                 if hasattr(step.observation, '__dict__') 
        #                 else str(step.observation)
        #             )
        #             observation_msg = AIMessage(
        #                 content=f"[OBSERVATION]:{observation_content}"
        #             )
        #             logger.debug("Created observation_msg: %s", observation_msg)
        #             yield State(messages=state['messages'] + [
        #                 action_msg,
        #                 action_input_msg,
        #                 observation_msg
        #             ])
            
        #     if 'messages' in chunk:
        #         for msg in chunk['messages']:
        #             logger.debug("Processing message in chunk: %s", json.dumps(msg, default=str))
        #             thought_msg = AIMessage(
        #                 content=f"[THOUGHT]:{msg.content}"
        #             )
        #             logger.debug("Created thought_msg: %s", thought_msg)
        #             yield State(messages=state['messages'] + [thought_msg])

        #     if 'actions' in chunk:
        #         for action in chunk['actions']:
        #             logger.debug("Processing action in chunk: %s", json.dumps(action, default=str))
        #             action_msg = AIMessage(
        #                 content=action.tool,
        #                 additional_kwargs={"type": "action"}
        #             )
        #             logger.debug("Created action_msg: %s", action_msg)
        #             action_input_msg = AIMessage(
        #                 content=action.tool_input,
        #                 additional_kwargs={"type": "action_input"}
        #             )
        #             logger.debug("Created action_input_msg: %s", action_input_msg)
        #             yield State(messages=state['messages'] + [
        #                 action_msg,
        #                 action_input_msg,
        #                 thought_msg
        #             ])

        #     if 'output' in chunk:
        #         logger.debug("Processing output in chunk: %s", json.dumps(chunk['output'], default=str))
        #         final_msg = AIMessage(
        #             content=f"[FINAL]:{chunk['output']}"
        #         )
        #         logger.debug("Created final_msg: %s", final_msg)
        #         yield State(messages=state['messages'] + [final_msg])

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
        from datetime import datetime
        
        # Initialize plot tools
        plot_time_series_tool = TimeSeriesPlotTool(
            sql_tool=GeneralSQLQueryTool(
                db=db,
                table_relationship_graph=dict(table_relationship_graph),
                user_id=user_id,
                global_hierarchy_access=global_hierarchy_access
            )
        )
        plot_wrapper_tool = TimeSeriesPlotWrapperTool(plot_tool=plot_time_series_tool)
        
        # Flag to switch between direct and wrapper invocation
        use_wrapper = True  # Set to False to use direct invocation
        
        if use_wrapper:
            # Create the plot request JSON for wrapper
            plot_request = {
                "primary_y_instruments": [
                    {
                        "instrument_id": "0003-L-2",
                        "column_name": "calculation1"
                    }
                ],
                "start_time": "1 January 2025 12:00:00 AM",
                "end_time": "31 May 2025 11:59:59 PM",
                "primary_y_title": "Settlement",
                "primary_y_unit": "mm",
                "highlight_zero": True
            }
            plot_request_string = json.dumps(plot_request, indent=2)
            logger.debug("Invoking plot tool through wrapper with request: %s", plot_request_string)
            result = plot_wrapper_tool.invoke(plot_request_string)
            
        else:
            # Create input parameters for direct invocation
            from dataclasses import dataclass
            
            @dataclass
            class InstrumentColumnPair:
                instrument_id: str
                column_name: str
            
            # Define test data for direct invocation
            start_time = datetime(2025, 1, 1)
            end_time = datetime(2025, 5, 31, 23, 59, 59)
            primary_instruments = [InstrumentColumnPair("0003-L-2", "calculation1")]
            secondary_instruments = []
            review_levels = [-10.0, -5.0, 5.0, 10.0]  # Example review levels
            
            logger.debug("Invoking TimeSeriesPlotTool directly")
            result = plot_time_series_tool._run(
                primary_y_instruments=primary_instruments,
                secondary_y_instruments=secondary_instruments,
                start_time=start_time,
                end_time=end_time,
                primary_y_title="Settlement",
                primary_y_unit="mm",
                secondary_y_title=None,
                secondary_y_unit=None,
                review_level_values=review_levels,
                highlight_zero=True
            )
        
        logger.debug("Tool returned result: %s", result)
        
        if isinstance(result, tuple):
            content, artifacts = result
            logger.debug("Tool returned artifacts: %s", artifacts)
            final_msg = AIMessage(
                content=content,
                additional_kwargs={
                    "type": "final",
                    "artifacts": artifacts
                }
            )
        else:
            final_msg = AIMessage(
                content=str(result),
                additional_kwargs={"type": "final"}
            )
        
        yield State(messages=state['messages'] + [final_msg])

    graph_builder = StateGraph(State)
    graph_builder.add_node('tool_test', test_tool)
    graph_builder.set_entry_point('tool_test')
    graph_builder.add_edge('tool_test', END)
    return graph_builder.compile(checkpointer=MemorySaver())