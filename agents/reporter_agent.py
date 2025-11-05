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

    # Collect successful executions (those with final_response)
    successful_executions = [ex for ex in executions if ex.final_response is not None]

    if successful_executions:
        execution_progresses = [
            ex.final_response.content + ("\nErrors: " + ex.error_summary if ex.error_summary else "")
            for ex in successful_executions
        ]
    else:
        # Fallback to error summaries if no successful executions
        execution_progresses = [
            "Errors: " + ex.error_summary
            for ex in executions if ex.error_summary
        ]

    execution_outputs = "\n\n".join(
        f"Output {i+1}:\n{prog}" for i, prog in enumerate(execution_progresses)
    )

    # Collect artefacts from successful executions, dedup plots and CSVs by description
    plot_dict = {}
    csv_dict = {}
    for ex in successful_executions:
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
You are a helpful assistant generating a concise, polite, and coherent response to a user's query based on provided data. Use markdown for structure (e.g., headings with **, lists with -) only if the response is long, complex, or requires separation of distinct components for clarity. Otherwise, keep the response simple and unstructured for conciseness and readability. The response should integrate:

- **Query**: The user's rephrased query incorporating chat history.
- **Edge Case**: If the query is an edge case, include a brief note about potential limitations.
- **Execution Outputs**: Ensemble outputs from multiple executions following these rules:
  - Ignore any obviously erroneous results (e.g., those with errors or nonsensical/invalid outputs).
  - If the remaining results are unanimous, accept that result.
  - If they disagree, accept the majority result if 60 percent or more of the executions agree on it; otherwise, discard the result and note the lack of consensus in the response.
  - Make the ensembled result coherent and flowing in the final response.
- **Plots**: If plots are provided, reference them by description, noting they will be displayed below in the UI.
- **CSV Files**: If CSV files are provided, reference them by description, noting they are available for download/viewing below in the UI.

**Instructions:**
- Keep the response polite, and helpful.
- Keep the response concise but include ALL relevant ensembled execution results even if they are large.
- Do not explicitly state the absence of plots or CSVs, as they are only generated when warranted.
- If plots or CSVs are included, integrate them naturally into the response.
- Do not include suggestions for follow-on queries.
- Ensure the response is self-contained and answers the query directly.
- If all outputs are errors or no consensus is reached, summarize the issues politely and suggest the query may need refinement.

**Output Format:**
Return a single markdown-formatted string, using sectioning only when necessary for clarity.
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
        edge_case_note = " (Note: This query was identified as an edge case, so the response may be limited.)" if is_edge_case else ""
        final_response = f"Unable to ensemble results due to an issue.{edge_case_note}"
        if execution_outputs:
            final_response += f"\nRaw outputs:\n{execution_outputs}"
        if plots_info:
            final_response += f" See the following plots below: {', '.join(plots_info)}."
        if csvs_info:
            final_response += f" Data files are available for download: {', '.join(csvs_info)}."

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