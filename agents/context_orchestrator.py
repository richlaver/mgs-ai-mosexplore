from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

from agents.instrument_validator import instrument_validator
from agents.database_expert import database_expert
from agents.period_expert import period_expert
from classes import ContextState

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

transfer_to_period_expert = create_handoff_tool(
    agent_name="period_expert",
    description="Transfer to period expert for deducing relevant date ranges from the query."
)

def get_context_graph(llm: BaseLanguageModel, db: any) -> any:
    tools = [transfer_to_instrument_validator, transfer_to_database_expert, transfer_to_period_expert]

    def supervisor_node(state: ContextState):
        has_clar_reqs = False
        try:
            if isinstance(state.context, dict):
                clar = state.context.get("clarification_requests") or []
                has_clar_reqs = bool(clar)
            else:
                clar = getattr(state.context, "clarification_requests", []) or []
                has_clar_reqs = bool(clar)
        except Exception:
            has_clar_reqs = False

        if has_clar_reqs:
            return Command(goto=END)

        user_input = (
            state.context.get("retrospective_query", "")
            if isinstance(state.context, dict)
            else getattr(state.context, "retrospective_query", "")
        )
        scratch_lines = []
        if state.period_deduced:
            scratch_lines.append("INTERNAL: Period deduction completed.")
        if state.instruments_validated:
            scratch_lines.append("INTERNAL: Instrument validation completed.")
        if state.db_context_provided:
            scratch_lines.append("INTERNAL: Database context retrieval completed.")
        agent_scratchpad = "\n".join(scratch_lines)

        system_content = f"""
You are a supervisor that delegates by calling tools to transfer control to agents.
STRICT ROUTING CONTRACT (follow exactly, highest precedence first):
1) If has_clarification_requests is true:
    - Do NOT call any tools.
    - Return NO tool calls.
2) If ALL are true, do NOT call any tools:
    - period_deduced == true
    - instruments_validated == true
    - db_context_provided == true
3) Otherwise (has_clarification_requests is false):
    - If period_deduced == false, call transfer_to_period_expert.
    - If instruments_validated == false, call transfer_to_instrument_validator.
    - If instruments_validated == true AND db_context_provided == false, call transfer_to_database_expert.

Notes:
- Call tools in parallel when multiple conditions in (3) apply.
- If unsure, prefer calling fewer tools; never call any tool when rule (1) applies.
- Do not explain your reasoning; just make the appropriate tool calls or none.

Current status:
- period_deduced={state.period_deduced}
- instruments_validated={state.instruments_validated}
- db_context_provided={state.db_context_provided}
- has_clarification_requests={has_clar_reqs}
{agent_scratchpad}
"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]

        llm_with_tools = llm.bind_tools(tools)
        try:
            response = llm_with_tools.invoke(messages)
        except Exception:
            return Command(goto=END)

        requested_targets = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc['name']
                if tool_name == "transfer_to_instrument_validator":
                    requested_targets.append("instrument_validator")
                elif tool_name == "transfer_to_database_expert":
                    requested_targets.append("database_expert")
                elif tool_name == "transfer_to_period_expert":
                    requested_targets.append("period_expert")

        allowed_targets = set()
        if not state.period_deduced:
            allowed_targets.add("period_expert")
        if not state.instruments_validated:
            allowed_targets.add("instrument_validator")
        if state.instruments_validated and not state.db_context_provided:
            allowed_targets.add("database_expert")

        final_targets = set(requested_targets).intersection(allowed_targets) if requested_targets else allowed_targets

        if not final_targets:
            return Command(goto=END)

        return Command(goto=[Send(target, state) for target in final_targets])

    graph = StateGraph(ContextState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("instrument_validator", lambda state: instrument_validator(state, llm, db))
    graph.add_node("database_expert", lambda state: database_expert(state, llm, db))
    graph.add_node("period_expert", lambda state: period_expert(state, llm, db))
    graph.add_edge(START, "supervisor")
    graph.add_edge("instrument_validator", "supervisor")
    graph.add_edge("database_expert", "supervisor")
    graph.add_edge("period_expert", "supervisor")
    return graph.compile()