from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from classes import AgentState, Context, CodingAttempt
from typing import List, Dict
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def reporter_agent(
    llm,
    context: Context,
    coding_attempts: List[CodingAttempt],
    messages: List
) -> List:
    """
    Generates a final response to the user's query using an LLM, based on sandbox outputs and context.
    Returns an updated list of messages with streamed AIMessage objects.

    Args:
        llm: ChatVertexAI instance (gemini-2.0-flash-001, temperature=0.1) for generating the response.
        context: Context object containing retrospective_query and edge_case.
        coding_attempts: List of CodingAttempt objects, with the latest containing execution_output.
        messages: Current list of messages to append to.

    Returns:
        Updated list of messages with streamed AIMessage objects.
    """
    logger.info("Starting reporter_agent")

    # Initialize the messages list to append to
    updated_messages = messages.copy() if messages else []

    # Get the retrospective query and edge case status
    retrospective_query = context.retrospective_query if context else "Unable to process query"
    is_edge_case = context.edge_case if context else False

    # Get the latest coding attempt's execution output
    execution_outputs = []
    if coding_attempts:
        current_attempt = coding_attempts[-1]  # Equivalent to get_current_coding_attempt
        execution_outputs = current_attempt.execution_output if current_attempt else []

    # Initialize response components
    execution_progress = ""
    plot_artefacts = []
    csv_artefacts = []

    # Process execution outputs
    for output in execution_outputs:
        metadata = output.get("metadata", {})
        output_type = metadata.get("type")
        content = output.get("content", "")

        if output_type in ["progress", "error", "final"]:
            execution_progress += content + "\n"
        elif output_type == "plot":
            try:
                plot_data = json.loads(content)
                plot_artefacts.append({
                    "tool_name": plot_data.get("tool_name"),
                    "description": plot_data.get("description"),
                    "artefact_id": plot_data.get("artefact_id")
                })
            except json.JSONDecodeError:
                logger.error("Failed to parse plot output: %s", content)
        elif output_type == "csv":
            try:
                csv_data = json.loads(content)
                csv_artefacts.append({
                    "description": csv_data.get("description"),
                    "artefact_id": csv_data.get("artefact_id")
                })
            except json.JSONDecodeError:
                logger.error("Failed to parse csv output: %s", content)

    # Define the LLM prompt for generating the final response
    reporter_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful assistant generating a concise, polite, and coherent response to a user's query based on provided data. Use markdown for structure (e.g., headings with **, lists with -) only if the response is long, complex, or requires separation of distinct components for clarity. Otherwise, keep the response simple and unstructured for conciseness and readability. The response should integrate:

- **Query**: The user's rephrased query incorporating chat history.
- **Edge Case**: If the query is an edge case, include a brief note about potential limitations.
- **Execution Progress**: Summarize progress of the executed code and any errors.
- **Plots**: If plots are provided, reference them by description, noting they will be displayed below in the UI.
- **CSV Files**: If CSV files are provided, reference them by description, noting they are available for download/viewing below in the UI.

**Instructions:**
- Keep the response concise, polite, and helpful.
- Do not explicitly state the absence of plots or CSVs, as they are only generated when warranted.
- If plots or CSVs are included, integrate them naturally into the response.
- Do not include suggestions for follow-on queries.
- Ensure the response is self-contained and answers the query directly.

**Output Format:**
Return a single markdown-formatted string, using sectioning only when necessary for clarity.
        """),
        ("human", """
**Query**: {retrospective_query}
**Edge Case**: {is_edge_case}
**Execution Progress**: {execution_progress}
**Plots**: {plots_info}
**CSV Files**: {csvs_info}
        """)
    ])

    # Prepare inputs for the LLM
    plots_info = []
    for i, plot in enumerate(plot_artefacts, 1):
        description = plot.get("description", "Plot")
        plots_info.append(f"Plot {i}: {description}")
    plots_info_str = "\n".join(plots_info) if plots_info else ""

    csvs_info = []
    for i, csv in enumerate(csv_artefacts, 1):
        description = csv.get("description", "Data file")
        csvs_info.append(f"Data File {i}: {description}")
    csvs_info_str = "\n".join(csvs_info) if csvs_info else ""

    # Generate the response using the LLM
    try:
        response_chain = reporter_prompt | llm
        final_response = response_chain.invoke({
            "retrospective_query": retrospective_query,
            "is_edge_case": "Yes" if is_edge_case else "No",
            "execution_progress": execution_progress,
            "plots_info": plots_info_str,
            "csvs_info": csvs_info_str
        }).content
    except Exception as e:
        logger.error("Failed to generate LLM response: %s", str(e))
        # Fallback response if LLM fails
        edge_case_note = " (Note: This query was identified as an edge case, so the response may be limited.)" if is_edge_case else ""
        final_response = f"{execution_progress}{edge_case_note}"
        if plots_info:
            final_response += f" See the following plots below: {', '.join(plots_info)}."
        if csvs_info:
            final_response += f" Data files are available for download: {', '.join(csvs_info)}."

    for raw in execution_outputs:
        metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
        output_type = metadata.get("type", "raw")
        content = raw.get("content", "") if isinstance(raw, dict) else str(raw)
        # Provide a concise prefix for non-final artefacts while retaining original content.
        if output_type == "plot":
            try:
                jd = json.loads(content)
                display_text = f"Plot artefact created: {jd.get('description','(no description)')} (id={jd.get('artefact_id')})"
            except Exception:
                display_text = f"Plot artefact (unparsed): {content[:200]}"  # fallback
        elif output_type == "csv":
            try:
                jd = json.loads(content)
                display_text = f"Data file created: {jd.get('description','(no description)')} (id={jd.get('artefact_id')})"
            except Exception:
                display_text = f"Data file artefact (unparsed): {content[:200]}"
        elif output_type == "final":
            display_text = content
        else:
            display_text = content

        updated_messages.append(AIMessage(
            name="Reporter",
            content=display_text,
            additional_kwargs={
                "stage": "execution_output",
                "process": output_type,
                **({"artefact_id": jd.get("artefact_id")} if output_type in {"plot", "csv"} and 'jd' in locals() and isinstance(jd, dict) else {})
            }
        ))

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
            content=f"Plot artefact available (id={plot.get('artefact_id')})",
            additional_kwargs={
                "stage": "final",
                "process": "plot",
                "artefact_id": plot.get("artefact_id")
            }
        ))

    for csv in csv_artefacts:
        updated_messages.append(AIMessage(
            name="Reporter",
            content=f"CSV artefact available (id={csv.get('artefact_id')})",
            additional_kwargs={
                "stage": "final",
                "process": "csv",
                "artefact_id": csv.get("artefact_id")
            }
        ))

    logger.info("Completed reporter_agent with %d messages", len(updated_messages))

    return updated_messages