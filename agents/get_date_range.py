from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from tools.datetime_toolkit import GetDatetimeNowTool
from langchain_core.messages import AIMessage
from typing import Literal, List
from classes import State
from pydantic import BaseModel, Field
import streamlit as st
from prompts import prompts


class DateRangesDict(BaseModel):
    start_date: str = Field(description='''
Date at the start of the date range.
    ''')
    end_date: str = Field(description='''
Date at the end of the date range.
    ''')


class GetDateRangeAgentOutput(BaseModel):
    is_data_request: bool = Field(description='''
True if get_date_range_agent deemed the user\'s query as requesting readings 
data, false if it thought no readings data was requested.
    ''')
    use_most_recent: bool = Field(description='''
False if get_date_range_agent thought the user\'s query referenced a specific 
date, true otherwise.
If true, then the query is assumed to request the most recent data.
    ''')
    date_ranges: List[DateRangesDict] = Field(description='''
If get_date_range_agent thinks the user\'s query referenced a specific date or 
dates, this is a list of the date ranges interpreted by get_date_range_agent 
from the specific date or dates mentioned in the user\'s query.
    ''')


def get_date_range_agent(state: State) -> Command[Literal['supervisor']]:
    agent_executor = create_react_agent(
        model=st.session_state.llm,
        tools=[
            GetDatetimeNowTool()
        ],
        prompt=prompts['prompt-006']['content']
    )
    response = agent_executor.invoke({'messages': [
        msg for msg in state["messages"] if msg.type == "human" or (msg.type == "ai" and msg.additional_kwargs.get("final") is True)
    ]})
    new_state = state.copy()
    new_state['messages'] = new_state['messages'] + [
        AIMessage(content=response.content, additional_kwargs={"type": "step"})
    ]
    return Command(
        goto="supervisor",
        update=new_state,
    )
