from datetime import datetime
import json
import logging
import time
from typing import Dict, List, Optional, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from tools.create_output_toolkit import MapPlotTool
from tools.sql_security_toolkit import GeneralSQLQueryTool
from tools.artefact_toolkit import WriteArtefactTool

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MapPlotSandboxAgentInput(BaseModel):
    prompt: str = Field(description="""
Natural language description of the map plot including:
- Data type ('readings' or 'review_levels').
- Plot type ('value_at_time' or 'change_over_period').
- Time range (e.g., 'from 1 January 2025 12:00:00 PM to 31 January 2025 11:59:59 PM').
- Series details (MUST include instrument types, subtypes, fields, names, units).
- Center (instrument ID or easting/northing) and radius.
- Optional: Excludes, buffer hours.
                        """)

class MapPlotSandboxAgentState(TypedDict):
    prompt: str
    tool_inputs: Optional[Dict]
    tool_error: Optional[str]
    tool_result: Optional[Dict]
    attempt_count: int
    messages: List[AIMessage]
    next_path: Optional[str]

def create_map_plot_sandbox_subgraph(llm: BaseLanguageModel, sql_tool: GeneralSQLQueryTool, write_artefact_tool: WriteArtefactTool, thread_id: str, user_id: int) -> CompiledStateGraph:
    logger.debug("Entering create_map_plot_sandbox_subgraph")
    plot_tool = MapPlotTool(sql_tool=sql_tool)
    logger.debug("MapPlotTool initialized")

    generate_prompt = ChatPromptTemplate.from_template("""
    You are an expert at formulating inputs for the MapPlotTool.

    Tool input schema (must match exactly; output as JSON):
    - data_type: Str, either 'readings' or 'review_levels'.
    - plot_type: Str, either 'value_at_time' or 'change_over_period'.
    - start_time: Optional str in 'D Month YYYY H:MM:SS AM/PM' (e.g., '1 January 2025 12:00:00 PM'). Required for 'change_over_period'.
    - end_time: Str in same format.
    - buffer_period_hours: Optional int, default 72.
    - series: List of dicts with 'instrument_type' (str), 'instrument_subtype' (str), 'database_field_name' (str like 'dataN' or 'calculationN'), 'measured_quantity_name' (str), 'abbreviated_unit' (str).
    - center_instrument_id: Optional str.
    - center_easting: Optional float.
    - center_northing: Optional float.
    - radius_meters: Int.
    - exclude_instrument_ids: Optional list of str, default [].

    User prompt:
    {prompt}

    Current date: {current_date}

    Previous failed inputs (if "None", no previous failure):
    {last_failed_inputs}

    Error from previous (if "None", no error; fix issues, keep correct parts):
    {last_error}

    Task:
    1. Parse prompt for data_type, plot_type, times, series details, center, radius, etc.
    2. Ensure times match format; start < end if applicable.
    3. Validate constraints (e.g., series details must match DB structure).
    4. If there was an error, correct the previous inputs based on the error.
    5. If unclear, infer reasonably but note assumptions.
    6. Output ONLY valid JSON matching schema (no extra text).
    """)
    logger.debug("Generate prompt template created")

    generate_chain = generate_prompt | llm | JsonOutputParser()
    logger.debug("Generate chain assembled")

    def generate_inputs(state: MapPlotSandboxAgentState) -> MapPlotSandboxAgentState:
        logger.debug("Entering generate_inputs with state: %s", json.dumps(state, default=str))
        t0 = time.perf_counter()
        messages = state.get('messages', [])
        messages.append(AIMessage(name="MapPlotAgent", content="Generating tool inputs...", additional_kwargs={"stage": "intermediate", "process": "action"}))
        last_failed_inputs = json.dumps(state.get('tool_inputs')) if state.get('tool_error') else "None"
        last_error = state.get('tool_error', "None")
        inputs = {"prompt": state['prompt'], "last_failed_inputs": last_failed_inputs, "last_error": last_error, "current_date": datetime.now().strftime('%B %d, %Y')}
        logger.debug("Inputs for generate_chain: %s", json.dumps(inputs, default=str))
        try:
            tool_inputs = generate_chain.invoke(inputs)
            logger.debug("[generate_inputs] produced inputs: %s", json.dumps(tool_inputs, indent=2))
            messages.append(AIMessage(name="MapPlotAgent", content=f"Generated inputs: {json.dumps(tool_inputs)}", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.debug("generate_inputs completed successfully in %.2f seconds", time.perf_counter() - t0)
            return {"tool_inputs": tool_inputs, "tool_error": None, "messages": messages}
        except Exception as e:
            logger.exception("Exception in generate_inputs: %s", str(e))
            error_msg = f"Error generating inputs: {str(e)}. Ensure output is valid JSON matching schema."
            messages.append(AIMessage(name="MapPlotAgent", content=error_msg, additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.debug("generate_inputs failed in %.2f seconds", time.perf_counter() - t0)
            return {"tool_inputs": None, "tool_error": error_msg, "messages": messages}

    def call_tool(state: MapPlotSandboxAgentState) -> MapPlotSandboxAgentState:
        logger.debug("Entering call_tool with state: %s", json.dumps(state, default=str))
        messages = state.get('messages', [])
        messages.append(AIMessage(name="MapPlotAgent", content="Calling MapPlotTool...", additional_kwargs={"stage": "intermediate", "process": "action"}))
        tool_inputs = state.get('tool_inputs')
        if not tool_inputs:
            error_msg = state.get('tool_error', "No inputs generated.")
            messages.append(AIMessage(name="MapPlotAgent", content=error_msg, additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.warning("call_tool aborted: No tool_inputs available")
            return {"tool_error": error_msg, "messages": messages}
        try:
            logger.debug("Invoking plot_tool.run with inputs: %s", json.dumps(tool_inputs, default=str))
            result = plot_tool.run(tool_inputs)
            content = None
            artefacts = []
            try:
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        content, artefacts = result[0], result[1]
                    elif len(result) == 1:
                        content = result[0]
                    else:
                        content = ""
                elif isinstance(result, dict):
                    content = result.get("content") or result.get("text") or result.get("message") or ""
                    artefacts = result.get("artefacts") or result.get("artifacts") or []
                else:
                    content = str(result)
            except Exception as e:
                logger.exception("Failed to interpret tool result type: %s", str(e))
                content = str(result)
                artefacts = []
            logger.debug("plot_tool.run returned content type: %s; artefacts type: %s", type(content).__name__, type(artefacts).__name__)
            logger.debug("plot_tool.run content (truncated): %s", (content[:500] if isinstance(content, str) else str(content)))
            logger.debug("plot_tool.run artefacts keys: %s", [list(a.keys()) for a in artefacts] if isinstance(artefacts, list) else artefacts)
            if content.startswith("Error:"):
                raise ValueError(content)
            tool_result = {"content": content, "artefacts": artefacts}
            messages.append(AIMessage(name="MapPlotAgent", content="Tool called successfully.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.debug("call_tool completed successfully")
            return {"tool_result": tool_result, "messages": messages}
        except json.JSONDecodeError as e:
            logger.exception("JSONDecodeError in call_tool: %s", str(e))
            error_msg = f"Invalid JSON: {str(e)}"
            messages.append(AIMessage(name="MapPlotAgent", content=error_msg, additional_kwargs={"stage": "intermediate", "process": "observation"}))
            return {"tool_error": error_msg, "messages": messages}
        except Exception as e:
            logger.exception("Exception in call_tool: %s", str(e))
            error_msg = str(e)
            messages.append(AIMessage(name="MapPlotAgent", content=error_msg, additional_kwargs={"stage": "intermediate", "process": "observation"}))
            return {"tool_error": error_msg, "messages": messages}

    def decide_next_node(state: MapPlotSandboxAgentState) -> MapPlotSandboxAgentState:
        logger.debug("Entering decide_next_node with state: %s", json.dumps(state, default=str))
        messages = state.get('messages', [])
        attempt_count = state.get('attempt_count', 0)
        tool_error = state.get('tool_error')
        tool_result = state.get('tool_result')
        should_retry = bool(tool_error) and not tool_result
        new_attempt = attempt_count + 1
        if should_retry and new_attempt < 5:
            messages.append(AIMessage(name="MapPlotAgent", content=f"Retrying due to: {tool_error[:100]}... Attempt {new_attempt}/5.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.info("Deciding to retry: next_path=generate_inputs, attempt=%d", new_attempt)
            return {"attempt_count": new_attempt, "next_path": "generate_inputs", "messages": messages}
        else:
            messages.append(AIMessage(name="MapPlotAgent", content="Proceeding to parse result." if tool_result else "Max attempts reached; returning None.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.info("Deciding to proceed: next_path=parse_result")
            return {"next_path": "parse_result", "messages": messages}

    def parse_result(state: MapPlotSandboxAgentState) -> MapPlotSandboxAgentState:
        logger.debug("Entering parse_result with state: %s", json.dumps(state, default=str))
        messages = state.get('messages', [])
        messages.append(AIMessage(name="MapPlotAgent", content="Parsing final result...", additional_kwargs={"stage": "intermediate", "process": "action"}))
        tool_result = state.get('tool_result')
        if not tool_result or not tool_result.get('artefacts'):
            messages.append(AIMessage(name="MapPlotAgent", content="No map generated.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.warning("parse_result: No map generated")
            return {"tool_result": None, "messages": messages}
        artefact_id = None
        generating_parameters = state.get('tool_inputs', {})
        for a in tool_result['artefacts']:
            if not isinstance(a, dict) or 'type' not in a or 'content' not in a or a['type'] != 'Plotly object':
                continue
            blob = a['content']
            try:
                output = write_artefact_tool._run(
                    blob=blob,
                    thread_id=thread_id,
                    user_id=user_id,
                    generating_tool='MapPlotTool',
                    generating_parameters=generating_parameters,
                    description='Plotly JSON for map plot'
                )
                if output['error']:
                    raise ValueError(output['error'])
                artefact_id = output['artefact_id']
                break
            except Exception as e:
                error_msg = f"Failed to store Plotly JSON artefact: {str(e)}"
                messages.append(AIMessage(name="MapPlotAgent", content=error_msg, additional_kwargs={"stage": "intermediate", "process": "observation"}))
                logger.exception(error_msg)
                return {"tool_result": None, "messages": messages}
        if artefact_id:
            messages.append(AIMessage(name="MapPlotAgent", content="Map artefact stored successfully.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.debug("parse_result: Map generated and stored successfully")
            return {"tool_result": artefact_id, "messages": messages}
        else:
            messages.append(AIMessage(name="MapPlotAgent", content="No Plotly JSON artefact stored.", additional_kwargs={"stage": "intermediate", "process": "observation"}))
            logger.warning("parse_result: No Plotly JSON artefact stored")
            return {"tool_result": None, "messages": messages}

    graph = StateGraph(MapPlotSandboxAgentState)
    graph.add_node("generate_inputs", generate_inputs)
    graph.add_node("call_tool", call_tool)
    graph.add_node("decide_next", decide_next_node)
    graph.add_node("parse_result", parse_result)

    graph.set_entry_point("generate_inputs")
    graph.add_edge("generate_inputs", "call_tool")
    graph.add_edge("call_tool", "decide_next")
    graph.add_conditional_edges(
        "decide_next",
        lambda state: state.get("next_path", "parse_result"),
        {"generate_inputs": "generate_inputs", "parse_result": "parse_result"}
    )
    graph.add_edge("parse_result", END)

    compiled_graph = graph.compile()
    logger.debug("Graph compiled successfully")
    return compiled_graph

class MapPlotSandboxAgentTool(BaseTool):
    name: str = "map_plot_sandbox_agent"
    description: str = """
    Agent to generate a map plot safely and store its Plotly JSON as an artefact. 
    Input: Natural language prompt which MUST include ALL of:
    - Data type ('readings' or 'review_levels').
    - Plot type ('value_at_time' or 'change_over_period').
    - Time range (e.g., 'from 1 January 2025 12:00:00 PM to 31 January 2025 11:59:59 PM').
    - Series details (instrument types, subtypes, fields, names, units).
    - Center (instrument ID or easting/northing) and radius.
    - Optional: Excludes, buffer hours.
    Returns: String artefact_id of the stored Plotly JSON or None if failed.
    """
    args_schema: type[BaseModel] = MapPlotSandboxAgentInput
    llm: BaseLanguageModel = Field(...)
    sql_tool: GeneralSQLQueryTool = Field(...)
    write_artefact_tool: WriteArtefactTool = Field(...)
    thread_id: str = Field(...)
    user_id: int = Field(...)

    def _run(self, prompt: str) -> Optional[str]:
        logger.debug("Entering MapPlotSandboxAgentTool._run with prompt: %s", prompt)
        graph = create_map_plot_sandbox_subgraph(self.llm, self.sql_tool, self.write_artefact_tool, self.thread_id, self.user_id)
        initial_state = {
            "prompt": prompt,
            "attempt_count": 0,
            "messages": [AIMessage(name="MapPlotAgent", content="Starting map generation.", additional_kwargs={"stage": "intermediate", "process": "action"})]
        }
        logger.debug("Initial state: %s", json.dumps(initial_state, default=str))
        try:
            final_state = graph.invoke(initial_state)
            logger.debug("Graph invocation completed with final_state: %s", json.dumps(final_state, default=str))
            return final_state.get('tool_result', None)
        except Exception as e:
            logger.exception("Exception during graph.invoke in _run: %s", str(e))
            return None

    def invoke(self, input=None, **kwargs):
        """Normalize various call styles into the expected single input.

        Supported forms:
        - invoke(prompt=...)  -> treated as the tool input string
        - invoke(input=...)   -> standard Runnable interface
        - invoke({"prompt": ...}) -> dict input using args_schema
        - invoke("...")      -> plain string input
        """
        if input is None and "prompt" in kwargs:
            input = kwargs.pop("prompt")
        if isinstance(input, dict) and set(input.keys()) == {"prompt"}:
            input = input["prompt"]
        return super().invoke(input, **kwargs)

def map_plot_sandbox_agent(llm: BaseLanguageModel, sql_tool: GeneralSQLQueryTool, write_artefact_tool: WriteArtefactTool, thread_id: str, user_id: int) -> BaseTool:
    logger.debug("Creating MapPlotSandboxAgentTool instance")
    return MapPlotSandboxAgentTool(llm=llm, sql_tool=sql_tool, write_artefact_tool=write_artefact_tool, thread_id=thread_id, user_id=user_id)