"""Standalone runner to test the map_plot_sandbox_agent tool.

This script:
- Initializes llm and db using functions from setup.py
- Builds the table relationship graph via setup.build_relationship_graph
- Constructs the GeneralSQLQueryTool
- Constructs the map_plot_sandbox_agent tool with user_id and thread_id
- Lets you select from multiple predefined prompt scenarios, pass a one-off prompt,
  or read a prompt from a file.
- If the tool returns an artefact_id, retrieves the Plotly JSON using ReadArtefactsTool
  and generates an HTML file 'map_plot_<timestamp>.html' in the current directory.

Quick usage:
    # Default scenario
    python run_map_plot_sandbox_agent.py

    # List scenarios
    python run_map_plot_sandbox_agent.py --list

    # Run a named scenario
    python run_map_plot_sandbox_agent.py --scenario basic_map

    # One-off prompt inline
    python run_map_plot_sandbox_agent.py --prompt "Plot readings value at time for settlements around 0003-L-2 at 1 March 2025 12:00:00 PM. Radius: 500m"

    # One-off prompt from a file
    python run_map_plot_sandbox_agent.py --prompt-file ./my_prompt.txt

    # Or via env var PROMPT_SCENARIO
    PROMPT_SCENARIO=basic_map python run_map_plot_sandbox_agent.py

Note: This expects your Streamlit secrets/environment for DB and LLM to be configured
      the same way as the app (see setup.get_llm and setup.get_db).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime

# Make Streamlit UI calls (st.toast) no-ops when not running inside a Streamlit app
try:
    import streamlit as st  # type: ignore
    try:
        st.toast  # attribute check
    except Exception:
        st.toast = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    else:
        # Always neutralize to avoid UI-context errors in plain Python
        st.toast = lambda *args, **kwargs: None  # type: ignore[attr-defined]
except Exception:
    st = None  # Not strictly needed; kept for clarity

import plotly.io as pio  # For generating HTML from Plotly JSON

# Set logging to DEBUG to display detailed logs in terminal
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# map_prompt = f"Plot readings as a map:\n" \
#     f"- Time range: 2025-03-03T00:00:00Z to 2025-03-31T23:59:59Z\n" \
#     f"- Buffer for missing readings: 7 days\n" \
#     f"- Plot centred on instrument ID 0003-L-2\n" \
#     f"- Plot extent: 200 metres radius\n" \
#     f"- Series 1:\n" \
#     f"  * Instrument type: LP\n" \
#     f"  * Instrument subtype: MOVEMENT\n" \
#     f"  * Database field name: calculation1\n" \
#     f"  * Label: Settlement\n" \
#     f"  * Unit: mm" #Hanoi Live
map_prompt = f"Plot readings as a map:\n" \
    f"- Time range: 2025-09-11T00:00:00Z to 2025-10-11T23:59:59Z\n" \
    f"- Buffer for missing readings: 7 days\n" \
    f"- Plot centred on instrument ID 1202/D/TSM1/12A(N)\n" \
    f"- Plot extent: 200 metres radius\n" \
    f"- Series 1:\n" \
    f"  * Instrument type: TSM\n" \
    f"  * Instrument subtype: DEFAULT\n" \
    f"  * Database field name: calculation1\n" \
    f"  * Label: Settlement\n" \
    f"  * Unit: mm" #LPP
# Registry of reusable prompt scenarios for quick testing
# SCENARIOS: dict[str, dict[str, str]] = {
#     "basic_map": {
#         "description": "Basic map plot for readings around instrument 0003-L-2.",
#         "prompt": map_prompt,
#     },
#     "change_over_period": {
#         "description": "Plot change of readings over period.",
#         "prompt": (
#             "Plot readings change over period for settlements (instrument_type: LP, subtype: MOVEMENT, field: calculation1, name: Settlement, unit: mm) "
#             "from 1 January 2025 12:00:00 PM to 18 November 2025 11:59:59 PM around 0003-L-2. "
#             "Radius: 1000 meters. Buffer: 14 days."
#         ),
#     },
#     "review_levels_at_time": {
#         "description": "Plot review levels at time.",
#         # "prompt": (
#         #     "Plot review_levels value at time for settlements (instrument_type: LP, subtype: MOVEMENT, field: data1, name: Settlement, unit: mm) "
#         #     "around center instrument 0003-L-2 at 14 November 2025 12:00:00 PM. Radius: 300 meters."
#         # ),
#         "prompt": (
#             "data_type='review_levels' plot_type='value_at_time' start_time=None end_time=datetime.datetime(2025, 12, 5, 11, 48, 10) buffer_period_hours=72 series=[SeriesDict(instrument_type='LP', instrument_subtype='LP', database_field_name='data1', measured_quantity_name='Settlement Marker Status', abbreviated_unit='status')] center_instrument_id='1523-1-L-01' center_easting=None center_northing=None radius_meters=500.0 exclude_instrument_ids=[]"
#         ),
#     },
#     "review_levels_change": {
#         "description": "Plot review levels change over time.",
#         "prompt": (
#             "Plot review_levels value change over time for settlements (instrument_type: LP, subtype: MOVEMENT, field: data1, name: Settlement, unit: mm) "
#             "around center instrument 0003-L-2 from 14 May 2025 12:00:00 PM to 14 November 2025 12:00:00 PM. Radius: 300 meters."
#         ),
#     },
# } #Hanoi Live
SCENARIOS: dict[str, dict[str, str]] = {
    "basic_map": {
        "description": "Basic map plot for readings around instrument 1202/D/TSM1/12A(N).",
        "prompt": map_prompt,
    },
    "change_over_period": {
        "description": "Plot change of readings over period.",
        "prompt": (
            "Plot readings change over period for settlements (instrument_type: TSM, subtype: DEFAULT, field: calculation1, name: Settlement, unit: mm) "
            "from 11 September 2025 12:00:00 PM to 12 January 2026 11:59:59 PM around 1202/D/TSM1/12A(N). "
            "Radius: 1000 meters. Buffer: 30 days."
        ),
    },
    "review_levels_at_time": {
        "description": "Plot review levels at time.",
        # "prompt": (
        #     "Plot review_levels value at time for settlements (instrument_type: LP, subtype: MOVEMENT, field: data1, name: Settlement, unit: mm) "
        #     "around center instrument 0003-L-2 at 14 November 2025 12:00:00 PM. Radius: 300 meters."
        # ),
        "prompt": (
            "data_type='review_levels' plot_type='value_at_time' start_time=None end_time=datetime.datetime(2026, 1, 12, 11, 48, 10) buffer_period_hours=150 series=[SeriesDict(instrument_type='TSM', instrument_subtype='DEFAULT', database_field_name='calculation1', measured_quantity_name='Settlement Marker Status', abbreviated_unit='status')] center_instrument_id='1202/D/TSM1/12A(N)' center_easting=None center_northing=None radius_meters=500.0 exclude_instrument_ids=[]"
        ),
    },
    "review_levels_change": {
        "description": "Plot review levels change over time.",
        "prompt": (
            "Plot review_levels value change over time for settlements (instrument_type: TSM, subtype: DEFAULT, field: calculation1, name: Settlement, unit: mm) "
            "around center instrument 1202/D/TSM1/12A(N) from 11 September 2025 12:00:00 PM to 12 January 2026 12:00:00 PM. Radius: 300 meters."
        ),
    },
} #LPP


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run map_plot_sandbox_agent against predefined or custom prompts.")
    parser.add_argument(
        "--scenario",
        "-s",
        help="Name of a predefined prompt scenario (use --list to see options).",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        help="Provide a one-off prompt string to run instead of a scenario.",
    )
    parser.add_argument(
        "--prompt-file",
        "-f",
        help="Path to a text file containing a one-off prompt to run instead of a scenario.",
    )
    return parser.parse_args(argv)


def _resolve_prompt(args: argparse.Namespace) -> tuple[str, str]:
    """Return (prompt_text, source_label).

    Precedence:
      1) --prompt
      2) --prompt-file
      3) --scenario or $PROMPT_SCENARIO
      4) default scenario 'basic_map'
    """
    # 1) One-off prompt inline
    if args.prompt:
        return args.prompt.strip(), "--prompt"

    # 2) One-off prompt from file
    if args.prompt_file:
        path = args.prompt_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Prompt file is empty: {path}")
        return content, f"--prompt-file:{os.path.basename(path)}"

    # 3) Scenario by flag or env var
    scenario = args.scenario or os.getenv("PROMPT_SCENARIO")
    if scenario:
        if scenario not in SCENARIOS:
            valid = ", ".join(sorted(SCENARIOS.keys()))
            raise KeyError(f"Unknown scenario '{scenario}'. Valid options: {valid}")
        return SCENARIOS[scenario]["prompt"].strip(), f"scenario:{scenario}"

    # 4) Default scenario
    default = "basic_map"
    return SCENARIOS[default]["prompt"].strip(), f"scenario:{default}"


def main() -> None:
    # 0) Parse args and handle scenario listing
    args = _parse_args()
    if args.list:
        print("Available scenarios:\n")
        for key in sorted(SCENARIOS.keys()):
            desc = SCENARIOS[key].get("description", "")
            print(f"- {key}: {desc}")
        return

    # 1) Resolve prompt from args/env/scenario registry
    try:
        PROMPT, source = _resolve_prompt(args)
    except Exception as e:
        print(f"Error selecting prompt: {e}")
        sys.exit(2)

    # 2) Heavy imports (deferred so --list can run without full env)
    try:
        from setup import get_llms, get_db, build_relationship_graph, get_blob_db, get_metadata_db  # type: ignore
        # Prefer table_info from table_info.py as requested; fallback to parameters.py if needed
        try:
            from table_info import table_info  # type: ignore
        except Exception:
            from parameters import table_info  # type: ignore
        from agents.map_plot_sandbox_agent import map_plot_sandbox_agent  # type: ignore
        from tools.sql_security_toolkit import GeneralSQLQueryTool  # type: ignore
        from tools.artefact_toolkit import ReadArtefactsTool, WriteArtefactTool  # type: ignore
    except Exception as e:
        print(f"Error importing runtime dependencies: {e}")
        sys.exit(3)

    # 3) Initialize LLM and DB (configured as per setup.py)
    llms = get_llms()
    if not llms:
        raise RuntimeError("setup.get_llms() returned no models; cannot proceed")
    llm = (
        llms.get("BALANCED")
        or llms.get("FAST")
        or llms.get("CODING")
        or next(iter(llms.values()))
    )
    metadata_db = get_metadata_db()

    # 4) Build the relationship graph from table_info via setup
    relationship_graph = build_relationship_graph(table_info)

    # 5) Construct the GeneralSQLQueryTool
    st.session_state.selected_project_key = "project_data.18_167_246_137__db_lpp"
    user_id = 1
    global_hierarchy_access = True
    sql_tool = GeneralSQLQueryTool(
        db=get_db(),
        table_relationship_graph=relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )

    # 6) Construct the artefact tools
    blob_db = get_blob_db()
    write_artefact_tool = WriteArtefactTool(blob_db=blob_db, metadata_db=metadata_db)
    read_artefact_tool = ReadArtefactsTool(blob_db=blob_db, metadata_db=metadata_db)

    # 7) Construct the map_plot_sandbox_agent tool with user_id and thread_id
    thread_id = str(uuid.uuid4())
    tool = map_plot_sandbox_agent(
        llm=llm,
        sql_tool=sql_tool,
        write_artefact_tool=write_artefact_tool,
        thread_id=thread_id,
        user_id=user_id,
    )

    # 8) Invoke the tool with the prompt
    print("Running map_plot_sandbox_agent...\n")
    print(f"Prompt source: {source}")
    print(f"Prompt: {PROMPT}")
    print()
    try:
        result = tool._run(PROMPT)
    except Exception as e:
        print(f"Error invoking tool: {e}")
        sys.exit(1)

    # 9) Display and handle results
    if result is None:
        print("No map generated (None).")
        return

    # Assume result is an artefact_id string; retrieve Plotly JSON and generate HTML
    try:
        # Retrieve the artefact using ReadArtefactsTool
        read_output = read_artefact_tool._run(
            metadata_only=False,
            artefact_ids=[result],
            thread_ids=[thread_id],
            user_ids=[user_id]
        )
        if not read_output['success']:
            raise ValueError(f"Error retrieving artefact: {read_output.get('error', 'Unknown error')}")
        artefacts = read_output['artefacts']
        if not artefacts:
            raise ValueError("No artefacts retrieved.")
        blob = artefacts[0].get('blob')
        if not blob:
            raise ValueError("No blob content in retrieved artefact.")
        plotly_json = blob.decode('utf-8') if isinstance(blob, bytes) else blob
        fig = pio.from_json(plotly_json)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"map_plot_{ts}.html"
        pio.write_html(fig, out_filename, full_html=True)
        print(f"Map generated and saved to '{out_filename}' with artefact_id: {result}")
    except Exception as e:
        print(f"Error generating HTML from artefact_id {result}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
