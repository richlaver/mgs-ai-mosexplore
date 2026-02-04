from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

from agents.instrument_id_validator import instrument_id_validator
from agents.instrument_type_validator import instrument_type_validator
from agents.database_expert import database_expert
from agents.period_expert import period_expert
from agents.project_insider import project_insider
from agents.platform_expert import platform_expert
from classes import ContextState

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer control to {agent_name}."

    @tool(name, description=description)
    def handoff_tool(payload: dict | None = None) -> str:
        return f"Transferred to {agent_name}"

    return handoff_tool

transfer_to_instrument_id_validator = create_handoff_tool(
    agent_name="instrument_id_validator",
    description="Transfer to instrument validator for identifying valid and potentially invalid instrument IDs in the query."
)

transfer_to_instrument_type_validator = create_handoff_tool(
    agent_name="instrument_type_validator",
    description="Transfer to instrument type validator for identifying instrument types and subtypes mentioned in the query."
)

transfer_to_database_expert = create_handoff_tool(
    agent_name="database_expert",
    description="Transfer to database expert for retrieving essential information on how to access data in the database to answer the query."
)

transfer_to_period_expert = create_handoff_tool(
    agent_name="period_expert",
    description="Transfer to period expert for deducing relevant date ranges from the query."
)

transfer_to_platform_expert = create_handoff_tool(
    agent_name="platform_expert",
    description="Transfer to platform expert for retrieving platform-specific terminology semantics and database guidance."
)

transfer_to_project_insider = create_handoff_tool(
    agent_name="project_insider",
    description="Transfer to project insider for retrieving ad-hoc insights specific to the project."
)

def get_context_graph(llms: Dict[str, BaseLanguageModel], db: any, selected_project_key: str | None) -> any:
    tools = [
        transfer_to_instrument_id_validator,
        transfer_to_instrument_type_validator,
        transfer_to_database_expert,
        transfer_to_period_expert,
        transfer_to_platform_expert,
        transfer_to_project_insider,
    ]

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
        if state.instrument_ids_validated:
            scratch_lines.append("INTERNAL: Instrument validation completed.")
        if state.instrument_types_validated:
            scratch_lines.append("INTERNAL: Instrument type validation completed.")
        if state.db_context_provided:
            scratch_lines.append("INTERNAL: Database context retrieval completed.")
        if state.platform_context_provided:
            scratch_lines.append("INTERNAL: Platform semantics retrieval completed.")
        if state.project_specifics_retrieved:
            scratch_lines.append("INTERNAL: Project-specific insights retrieval completed.")
        agent_scratchpad = "\n".join(scratch_lines)

        system_content = f"""
You are a supervisor that delegates by calling tools to transfer control to agents.
STRICT ROUTING CONTRACT (follow exactly, highest precedence first):
1) If has_clarification_requests is true:
    - Do NOT call any tools.
    - Return NO tool calls.
2) If ALL are true, do NOT call any tools:
    - period_deduced == true
    - instrument_ids_validated == true
    - instrument_types_validated == true
    - db_context_provided == true
    - platform_context_provided == true
    - project_specifics_retrieved == true
3) Otherwise (has_clarification_requests is false):
    - If period_deduced == false, call transfer_to_period_expert.
    - If instrument_ids_validated == false, call transfer_to_instrument_id_validator.
    - If instrument_types_validated == false, call transfer_to_instrument_type_validator.
    - If platform_context_provided == false, call transfer_to_platform_expert.
    - If project_specifics_retrieved == false, call transfer_to_project_insider.
    - If instrument_ids_validated == true AND instrument_types_validated == true AND db_context_provided == false, call transfer_to_database_expert.

Notes:
- Call tools in parallel when multiple conditions in (3) apply.
- If unsure, prefer calling fewer tools; never call any tool when rule (1) applies.
- Do not explain your reasoning; just make the appropriate tool calls or none.

Current status:
- period_deduced={state.period_deduced}
- instrument_ids_validated={state.instrument_ids_validated}
- instrument_types_validated={state.instrument_types_validated}
- db_context_provided={state.db_context_provided}
- platform_context_provided={state.platform_context_provided}
- project_specifics_retrieved={state.project_specifics_retrieved}
- has_clarification_requests={has_clar_reqs}
{agent_scratchpad}
"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]

        llm_with_tools = llms["BALANCED"].bind_tools(tools)
        try:
            response = llm_with_tools.invoke(messages)
        except Exception:
            return Command(goto=END)

        requested_targets = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc['name']
                if tool_name == "transfer_to_instrument_id_validator":
                    requested_targets.append("instrument_id_validator")
                elif tool_name == "transfer_to_instrument_type_validator":
                    requested_targets.append("instrument_type_validator")
                elif tool_name == "transfer_to_database_expert":
                    requested_targets.append("database_expert")
                elif tool_name == "transfer_to_period_expert":
                    requested_targets.append("period_expert")
                elif tool_name == "transfer_to_platform_expert":
                    requested_targets.append("platform_expert")
                elif tool_name == "transfer_to_project_insider":
                    requested_targets.append("project_insider")

        allowed_targets = set()
        if not state.period_deduced:
            allowed_targets.add("period_expert")
        if not state.instrument_ids_validated:
            allowed_targets.add("instrument_id_validator")
        if not state.instrument_types_validated:
            allowed_targets.add("instrument_type_validator")
        if not state.platform_context_provided:
            allowed_targets.add("platform_expert")
        if not state.project_specifics_retrieved:
            allowed_targets.add("project_insider")
        if state.instrument_ids_validated and state.instrument_types_validated and not state.db_context_provided:
            allowed_targets.add("database_expert")

        final_targets = set(requested_targets).intersection(allowed_targets) if requested_targets else allowed_targets

        if not final_targets:
            return Command(goto=END)

        return Command(goto=[Send(target, state) for target in final_targets])

    graph = StateGraph(ContextState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("instrument_id_validator", lambda state: instrument_id_validator(state, llms['FAST'], db))
    graph.add_node("instrument_type_validator", lambda state: instrument_type_validator(state, selected_project_key))
    graph.add_node("database_expert", lambda state: database_expert(state, llms["FAST"], db, selected_project_key))
    graph.add_node("period_expert", lambda state: period_expert(state, llms['FAST'], db))
    graph.add_node("project_insider", lambda state: project_insider(state, selected_project_key))
    graph.add_node("platform_expert", platform_expert)
    graph.add_edge(START, "supervisor")
    graph.add_edge("instrument_id_validator", "supervisor")
    graph.add_edge("instrument_type_validator", "supervisor")
    graph.add_edge("database_expert", "supervisor")
    graph.add_edge("period_expert", "supervisor")
    graph.add_edge("project_insider", "supervisor")
    graph.add_edge("platform_expert", "supervisor")
    return graph.compile()