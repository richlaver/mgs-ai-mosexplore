"""Configures the LangGraph workflow for query processing in the MissionHelp Demo.

This module defines the retrieval-augmented generation pipeline, integrating the
language model and vector store.
"""

from typing import Generator

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool, ListSQLDatabaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor
import time
import logging

from classes import State, QueryOutput

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def build_graph_agent(llm, db) -> StateGraph:
    """Builds the LangGraph workflow for query processing.

    Args:
        llm: Language model instance.
        db: SQL database instance.

    Returns:
        Compiled LangGraph instance.
    """
    st.toast("Building LangGraph workflow...", icon=":material/build:")


    def generate_response(state: State) -> Generator[State, None, None]:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        system_message = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect="MySQL",
            top_k=5,
        )
        question = state["messages"][-1].content

        agent = create_react_agent(llm, tools, prompt=system_message)
        logging.debug('agent:')
        logging.debug(agent)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        logging.debug('agent_executor:')
        logging.debug(agent_executor)
        logging.debug(f'Question: {question}')

        new_state = state.copy()
        for chunk in agent_executor.stream({'messages': [HumanMessage(content=question)]}):
            logging.debug('Streamed chunk in generate_response: ')
            logging.debug(chunk)
            content = ''
            if 'actions' in chunk:
                for action in chunk['actions']:
                    try:
                        content = f'Calling Tool: `{action.tool}` with input `{action.tool_input}`'
                        new_state['messages'] = new_state['messages'] + [AIMessage(content=content)]
                        yield new_state
                    except Exception as e:
                        logging.error(f'Action in chunk: {action}')
            elif 'steps' in chunk:
                for step in chunk['steps']:
                    content = f'Tool Result: `{step.observation}`'
                    new_state['messages'] = new_state['messages'] + [AIMessage(content=content)]
                    yield new_state
            elif 'output' in chunk:
                content = f'Final Output: {chunk['output']}'
                new_state['messages'] = new_state['messages'] + [AIMessage(content=content)]
                yield new_state
            else:
                raise ValueError()    

    graph_builder = StateGraph(State)
    graph_builder.add_node('response_generation', generate_response)
    graph_builder.set_entry_point('response_generation')
    graph_builder.set_finish_point('response_generation')
    return graph_builder.compile(checkpointer=MemorySaver())


def build_graph(llm, db) -> StateGraph:
    """Builds the LangGraph workflow for query processing.

    Args:
        llm: Language model instance.
        db: SQL database instance.

    Returns:
        Compiled LangGraph instance.
    """
    st.toast("Building LangGraph workflow...", icon=":material/build:")
    

    def generate_sql_query(state: State) -> dict:
        start_time = time.time()
        system_message_content = """
            Given an input question, create a syntactically correct {dialect} query to
            run to help find the answer. Unless the user specifies in his question a
            specific number of examples they wish to obtain, always limit your query to
            at most {top_k} results. You can order the results by a relevant column to
            return the most interesting examples in the database.

            Never query for all the columns from a specific table, only ask for a the
            few relevant columns given the question.

            Pay attention to use only the column names that you can see in the schema
            description. Be careful to not query for columns that do not exist. Also,
            pay attention to which column is in which table.

            Only use the following tables:
            {table_info}
        """
        user_prompt = "Question: {input}"
        query_prompt_template = ChatPromptTemplate([
            ('system', system_message_content),
            ('human', user_prompt)
        ])
        question = state["messages"][-1].content
        logging.debug(f"Question: {question}")
        table_info = db.get_table_info()
        logging.debug(f"Dialect: {db.dialect}")
        logging.debug(f"Table info: {table_info}")
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": question,
            }
        )
        logging.debug(f"Generated prompt: {prompt}")
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        logging.debug(f"LLM Output: {result}")
        logging.debug(f"Extracted SQL query: {result['query']}")
        new_state = state.copy()
        new_state["messages"] = new_state["messages"] + [AIMessage(
            content=result['query']
        )]
        query_time = time.time() - start_time
        new_state["timings"].append({"node": "generate_sql_query", "time": query_time, "component": "sql_generation"})
        logging.debug(f"New state: {new_state}")
        return new_state


    def execute_sql_query(state: State) -> dict:
        """Execute SQL query."""
        start_time = time.time()
        logging.debug(f"State messages in execute_query: {state["messages"]}")
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        query = state["messages"][-1].content
        logging.debug(f"Query read from messages: {query}")
        result = execute_query_tool.invoke(query)
        logging.debug(f"Result: {result}")
        execution_time = time.time() - start_time
        new_state = state.copy()
        new_state["messages"] = new_state["messages"] + [AIMessage(
            content=result
        )]
        logging.debug("New state stored in sql execution: {}")
        new_state["timings"].append({"node": "execute_sql_query", "time": execution_time, "component": "sql_execution"})
        return new_state
    

    def generate_response(state: State):
        """Answer question using retrieved information as context."""
        start_time = time.time()
        question = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content
        logging.debug(f"Question in generate_response: {question}")
        sql_query = [msg for msg in state['messages'] if isinstance(msg, AIMessage)][-2].content
        logging.debug(f"SQL query in generate_response {sql_query}")
        sql_result = [msg for msg in state['messages'] if isinstance(msg, AIMessage)][-1].content
        logging.debug(f"SQL result in generate_response {sql_result}")
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {question}\n'
            f'SQL Query: {sql_query}\n'
            f'SQL Result: {sql_result}'
        )
        logging.debug(f"Prompt in generate_response: {prompt}")
        response = llm.invoke(prompt)
        logging.debug(f"Response: {response}")
        generation_time = time.time() - start_time
        new_state = state.copy()
        new_state["messages"] = new_state["messages"] + [AIMessage(
            content=response.content
        )]
        logging.debug("New state stored in response generation: {}")
        new_state["timings"].append({"node": "generate_response", "time": generation_time, "component": "response_generation"})
        return new_state
    

    # Initialize the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("sql_generation", generate_sql_query)
    graph_builder.add_node("sql_execution", execute_sql_query)
    graph_builder.add_node("response_generation", generate_response)
    graph_builder.set_entry_point("sql_generation")
    graph_builder.add_edge("sql_generation", "sql_execution")
    graph_builder.add_edge("sql_execution", "response_generation")
    graph_builder.add_edge("response_generation", END)
    return graph_builder.compile(checkpointer=MemorySaver())