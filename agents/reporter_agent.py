from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from classes import AgentState, Context, Execution
from typing import List
import json
import logging

logger = logging.getLogger(__name__)

def reporter_agent(
    llm,
    context: Context,
    messages: List,
    executions: List[Execution],
) -> List:
    """
    Generate a concise final response based on context and execution outputs from multiple agents.

    Args:
        llm: Chat LLM for response synthesis.
        context: Context including retrospective_query and edge_case.
        messages: Running list of messages (preserved but not used for extraction).
        executions: List of execution runs from AgentState.

    Returns:
        Updated list of messages, including a final AIMessage and unique artefact messages with stage="final".
    """
    logger.info("Starting reporter_agent")

    updated_messages = messages.copy() if messages else []

    retrospective_query = context.retrospective_query if context else "Unable to process query"
    is_edge_case = context.edge_case if context else False

    best_ex = next((ex for ex in executions if ex.is_best), None)
    if best_ex is None:
        best_ex = next((ex for ex in executions if ex.final_response is not None), None)

    execution_outputs = best_ex.final_response.content if (best_ex and best_ex.final_response) else ""

    plot_dict = {}
    csv_dict = {}
    for ex in ([best_ex] if best_ex else []):
        for art in ex.artefacts:
            process = art.additional_kwargs.get("process")
            desc = art.content
            artefact_id = art.additional_kwargs.get("artefact_id")
            if process == "plot":
                if desc not in plot_dict:
                    plot_dict[desc] = {"description": desc, "artefact_id": artefact_id}
            elif process == "csv":
                if desc not in csv_dict:
                    csv_dict[desc] = {"description": desc, "artefact_id": artefact_id}

    plot_artefacts = list(plot_dict.values())
    csv_artefacts = list(csv_dict.values())

    logger.info(f"execution_outputs: {execution_outputs}")
    logger.info(f"Plot artefacts: {plot_artefacts}")
    logger.info(f"CSV artefacts: {csv_artefacts}")

    reporter_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful assistant generating a concise, polite, and coherent response to a user's query based on provided data. Use markdown only when helpful for clarity. Integrate:

- Query: the user's rephrased query incorporating chat history.
- Edge Case note only if relevant.
- Best Execution Output: Use the provided content directly to answer the query, rewriting for clarity and flow.
- Plots/CSVs: If provided, reference them by description, noting they'll be displayed below or available for download. Do not reference plots with captions e.g. "Plot 1" because there are none. Do not mention any artefact IDs.

Instructions:
- Keep the response concise, self-contained, and directly answer the query.

Output: return a single markdown-formatted string when needed for clarity.
        """),
        ("human", """
**Query**: {retrospective_query}
**Edge Case**: {is_edge_case}
**Execution Outputs**: {execution_outputs}
**Plots**: {plots_info}
**CSV Files**: {csvs_info}
        """)
    ])

    # Prepare inputs for the LLM
    plots_info = [f"Plot {i+1}: {p['description']}" for i, p in enumerate(plot_artefacts)]
    plots_info_str = "\n".join(plots_info) if plots_info else ""

    csvs_info = [f"Data File {i+1}: {c['description']}" for i, c in enumerate(csv_artefacts)]
    csvs_info_str = "\n".join(csvs_info) if csvs_info else ""

    final_response = None
    try:
        response_chain = reporter_prompt | llm
        final_response = response_chain.invoke({
            "retrospective_query": retrospective_query,
            "is_edge_case": "Yes" if is_edge_case else "No",
            "execution_outputs": execution_outputs,
            "plots_info": plots_info_str,
            "csvs_info": csvs_info_str
        }).content
    except Exception as e:
        logger.error("Failed to generate LLM response: %s", str(e))
        final_response = execution_outputs or "I'm unable to provide a response at this time."

    if not final_response or not str(final_response).strip():
        final_response = execution_outputs or "I'm unable to provide a response at this time."

    updated_messages.append(AIMessage(
        name="Reporter",
        content=final_response,
        additional_kwargs={
            "stage": "final",
            "process": "response"
        }
    ))

    for plot in plot_artefacts:
        updated_messages.append(AIMessage(
            name="Reporter",
            content=f"Plot artefact available (id={plot['artefact_id']})",
            additional_kwargs={
                "stage": "final",
                "process": "plot",
                "artefact_id": plot['artefact_id']
            }
        ))

    for csv_file in csv_artefacts:
        updated_messages.append(AIMessage(
            name="Reporter",
            content=f"CSV artefact available (id={csv_file['artefact_id']})",
            additional_kwargs={
                "stage": "final",
                "process": "csv",
                "artefact_id": csv_file['artefact_id']
            }
        ))

    logger.info("Completed reporter_agent with %d messages", len(updated_messages))
    logger.info("updated_messages from reporter_agent: %s", [msg.content for msg in updated_messages])

    return updated_messages