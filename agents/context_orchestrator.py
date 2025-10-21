import operator
from typing import List, Annotated

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from pydantic import BaseModel

from agents.instrument_validator import instrument_validator
from agents.database_expert import database_expert
from classes import Context

class ContextState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add] = []
    context: Context
    clarification_request: str = ""
    instruments_validated: bool = False
    db_context_provided: bool = False

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer control to {agent_name}."

    @tool(name, description=description)
    def handoff_tool(payload: dict | None = None) -> str:
        return f"Transferred to {agent_name}"

    return handoff_tool

transfer_to_instrument_validator = create_handoff_tool(
    agent_name="instrument_validator",
    description="Transfer to instrument validator for identifying valid and potentially invalid instrument IDs in the query."
)

transfer_to_database_expert = create_handoff_tool(
    agent_name="database_expert",
    description="Transfer to database expert for retrieving essential information on how to access data in the database to answer the query."
)

def get_context_graph(llm: BaseLanguageModel, db: any) -> any:
    supervisor_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a supervisor that delegates by calling tools to transfer control to agents.\n"
            "Routing policy:\n"
            "- If instruments_validated is false, call transfer_to_instrument_validator.\n"
            "- Else if db_context_provided is false, call transfer_to_database_expert.\n"
            "- Else, do not call any tools.\n"
            "Current status: instruments_validated={instruments_validated}, db_context_provided={db_context_provided}"
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [transfer_to_instrument_validator, transfer_to_database_expert]
    agent = create_tool_calling_agent(llm, tools, supervisor_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)

    def supervisor_node(state: ContextState):
        user_input = getattr(state.context, "retrospective_query", "")
        scratch_lines = []
        if state.instruments_validated:
            scratch_lines.append("INTERNAL: Instrument validation completed.")
        if state.db_context_provided:
            scratch_lines.append("INTERNAL: Database context retrieval completed.")
        inputs = {
            "input": user_input,
            "instruments_validated": state.instruments_validated,
            "db_context_provided": state.db_context_provided,
            "agent_scratchpad": "\n".join(scratch_lines),
        }

        result = agent_executor.invoke(inputs)
        target = None
        try:
            steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
            if steps:
                last_action = steps[-1][0]
                last_tool = getattr(last_action, "tool", None)
                if last_tool == "transfer_to_instrument_validator":
                    target = "instrument_validator"
                elif last_tool == "transfer_to_database_expert":
                    target = "database_expert"
        except Exception:
            target = None

        if target is None:
            return Command(goto=END)

        return Command(goto=target)

    graph = StateGraph(ContextState)
    graph.add_node("supervisor", supervisor_node, destinations=("instrument_validator", "database_expert", END))
    graph.add_node("instrument_validator", lambda state: instrument_validator(state, llm, db))
    graph.add_node("database_expert", lambda state: database_expert(state, llm, db))
    graph.add_edge(START, "supervisor")
    graph.add_edge("instrument_validator", "database_expert")
    graph.add_edge("database_expert", "supervisor")
    return graph.compile()