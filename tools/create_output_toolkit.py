from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict, validator
from typing import List, Dict, Optional, Union, Tuple, Type
from .sql_security_toolkit import GeneralSQLQueryTool
from .get_trend_info_toolkit import parse_datetime, format_datetime
import plotly.graph_objects as go
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import datetime as datetime_module
import numpy as np
import re
import uuid
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class InstrumentColumnPair(BaseModel):
    """Model for specifying an instrument and its associated data column for plotting."""
    instrument_id: str = Field(
        ...,
        description="The unique identifier of the instrument in the database (e.g., 'INST001'). Must be a non-empty string matching an existing instrument ID in the 'mydata' table's 'instr_id' column."
    )
    column_name: str = Field(
        ...,
        description="The name of the database column containing the instrument's data. Must be in the format 'dataN' or 'calculationN' where N is a positive integer (e.g., 'data1', 'calculation2'). For 'dataN', the value is directly retrieved from the column. For 'calculationN', the value is extracted from the 'custom_fields' JSON column using JSON_EXTRACT."
    )

    @validator('column_name')
    def validate_column_name(cls, v):
        if not re.match(r'^(data|calculation)\d+$', v):
            raise ValueError("Column name must be 'data<n>' or 'calculation<n>'")
        return v

class TimeSeriesPlotInput(BaseModel):
    """Input model for plotting time series data from a database, specifying instruments, time range, and plot configuration."""
    primary_y_instruments: List[InstrumentColumnPair] = Field(
        ...,
        description="A list of instrument-column pairs to plot on the primary (left) y-axis. Each pair is an object with 'instrument_id' (a string, e.g., 'INST001') and 'column_name' (a string, e.g., 'data1' or 'calculation1'). At least one pair is required. The total number of pairs (primary + secondary) cannot exceed 7. Example: [{'instrument_id': 'INST001', 'column_name': 'data1'}, {'instrument_id': 'INST002', 'column_name': 'calculation1'}]."
    )
    secondary_y_instruments: List[InstrumentColumnPair] = Field(
        default_factory=list,
        description="An optional list of instrument-column pairs to plot on the secondary (right) y-axis. Each pair is an object with 'instrument_id' (a string, e.g., 'INST003') and 'column_name' (a string, e.g., 'data2' or 'calculation2'). If provided, review levels cannot be used. The total number of pairs (primary + secondary) cannot exceed 7. Example: [{'instrument_id': 'INST003', 'column_name': 'data2'}]."
    )
    start_time: datetime = Field(
        ...,
        description="The start of the time range for the data to plot, as a datetime string in the format 'D Month YYYY H:MM:SS AM/PM' (e.g., '1 January 2025 12:00:00 PM'). Must be earlier than end_time and match the format of the 'date1' column in the database."
    )
    end_time: datetime = Field(
        ...,
        description="The end of the time range for the data to plot, as a datetime string in the format 'D Month YYYY H:MM:SS AM/PM' (e.g., '31 May 2025 2:00:00 PM'). Must be later than start_time and match the format of the 'date1' column in the database."
    )
    primary_y_title: str = Field(
        ...,
        description="The title for the primary (left) y-axis, describing the data being plotted (e.g., 'Temperature'). Must be a non-empty string. Example: 'Temperature'."
    )
    primary_y_unit: str = Field(
        ...,
        description="The unit for the primary (left) y-axis, displayed in parentheses in the axis title (e.g., '°C'). Must be a non-empty string. Example: '°C'."
    )
    secondary_y_title: Optional[str] = Field(
        None,
        description="The title for the secondary (right) y-axis, if secondary_y_instruments is provided (e.g., 'Pressure'). Required if secondary_y_instruments is non-empty, otherwise optional. Example: 'Pressure'."
    )
    secondary_y_unit: Optional[str] = Field(
        None,
        description="The unit for the secondary (right) y-axis, displayed in parentheses in the axis title (e.g., 'kPa'). Required if secondary_y_instruments is non-empty, otherwise optional. Example: 'kPa'."
    )
    review_level_values: List[float] = Field(
        default_factory=list,
        description="An optional list of float values to plot as horizontal dashed lines on the primary y-axis, representing thresholds or review levels (e.g., [10.0, -5.0]). Maximum 3 positive and 3 negative values allowed. Cannot be used if secondary_y_instruments is non-empty. Example: [10.0, 5.0, -5.0]."
    )
    highlight_zero: bool = Field(
        False,
        description="Whether to highlight the zero line on the primary y-axis with a light grey line. Only applicable if secondary_y_instruments is empty. Set to true to enable, false to disable. Example: true."
    )

class BaseSQLQueryTool(BaseModel):
    """Base tool for SQL database interaction."""
    sql_tool: GeneralSQLQueryTool = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TimeSeriesPlotTool(BaseTool, BaseSQLQueryTool):
    """Tool for plotting time series data with Plotly."""
    name: str = "time_series_plot"
    description: str = """
    Creates an interactive Plotly time series plot and CSV file from instrumentation data.
    Supports multiple time series with different column names, dual y-axes, review levels, and customizable gridlines.
    Returns a tuple of (content, artefacts) with response_format='content_and_artifact'.
    The artefacts include the Plotly JSON and CSV file.
    """
    args_schema: Type[TimeSeriesPlotInput] = TimeSeriesPlotInput
    response_format: str = "content_and_artifact"

    def _get_x_grid_settings(self, start_time: datetime, end_time: datetime) -> Dict:
        """Determine x-axis grid settings and formatting."""
        time_delta_days = (end_time - start_time).total_seconds() / (24 * 3600)
        year_seconds = 365.25 * 24 * 3600
        month_seconds = year_seconds / 12
        day_seconds = 24 * 3600
        hour_seconds = 3600
        minute_seconds = 60

        grid_settings = [
            (35 * year_seconds, {'major_spacing': 10 * year_seconds, 'minor_spacing': 2 * year_seconds, 'major_format': '%Y', 'start_unit': 'year'}),
            (15 * year_seconds, {'major_spacing': 5 * year_seconds, 'minor_spacing': 1 * year_seconds, 'major_format': '%Y', 'start_unit': 'year'}),
            (7 * year_seconds, {'major_spacing': 2 * year_seconds, 'minor_spacing': 0.5 * year_seconds, 'major_format': '%Y', 'start_unit': 'year'}),
            (4 * year_seconds, {'major_spacing': 1 * year_seconds, 'minor_spacing': 0.25 * year_seconds, 'major_format': '%Y', 'start_unit': 'year'}),
            (2 * year_seconds, {'major_spacing': 6 * month_seconds, 'minor_spacing': 2 * month_seconds, 'major_format': '%b-%Y', 'start_unit': 'month'}),
            (1 * year_seconds, {'major_spacing': 3 * month_seconds, 'minor_spacing': 1 * month_seconds, 'major_format': '%b-%Y', 'start_unit': 'month'}),
            (6 * month_seconds, {'major_spacing': 2 * month_seconds, 'minor_spacing': 1 * month_seconds, 'major_format': '%b-%Y', 'start_unit': 'month'}),
            (2 * month_seconds, {'major_spacing': 1 * month_seconds, 'minor_spacing': 0.5 * month_seconds, 'major_format': '%b-%Y', 'start_unit': 'month'}),
            (1 * month_seconds, {'major_spacing': 2 * 7 * day_seconds, 'minor_spacing': 7 * day_seconds, 'major_format': '%d-%b-%Y', 'start_unit': 'day'}),
            (10 * day_seconds, {'major_spacing': 7 * day_seconds, 'minor_spacing': 1 * day_seconds, 'major_format': '%d-%b-%Y', 'start_unit': 'day'}),
            (2 * day_seconds, {'major_spacing': 1 * day_seconds, 'minor_spacing': 0.5 * day_seconds, 'major_format': '%d-%b-%Y', 'start_unit': 'midnight'}),
            (12 * hour_seconds, {'major_spacing': 12 * hour_seconds, 'minor_spacing': 3 * hour_seconds, 'major_format': '%H:%M-%d-%b-%Y', 'start_unit': 'midnight'}),
            (6 * hour_seconds, {'major_spacing': 6 * hour_seconds, 'minor_spacing': 2 * hour_seconds, 'major_format': '%H:%M-%d-%b-%Y', 'start_unit': 'midnight'}),
            (3 * hour_seconds, {'major_spacing': 3 * hour_seconds, 'minor_spacing': 1 * hour_seconds, 'major_format': '%H:%M-%d-%b-%Y', 'start_unit': 'hour'}),
            (1 * hour_seconds, {'major_spacing': 1 * hour_seconds, 'minor_spacing': 30 * minute_seconds, 'major_format': '%H:%M-%d-%b-%Y', 'start_unit': 'hour'}),
            (20 * minute_seconds, {'major_spacing': 10 * minute_seconds, 'minor_spacing': 5 * minute_seconds, 'major_format': '%M:%H-%d-%b-%Y', 'start_unit': 'minute'}),
            (5 * minute_seconds, {'major_spacing': 5 * minute_seconds, 'minor_spacing': 1 * minute_seconds, 'major_format': '%M:%H-%d-%b-%Y', 'start_unit': 'minute'}),
            (0, {'major_spacing': 1 * minute_seconds, 'minor_spacing': 20, 'major_format': '%M:%H-%d-%b-%Y', 'start_unit': 'minute'})
        ]

        time_delta = (end_time - start_time).total_seconds()
        for threshold, settings in grid_settings:
            if time_delta > threshold:
                settings['major_dtick'] = settings['major_spacing'] * 1000  # Plotly uses milliseconds
                settings['minor_dtick'] = settings['minor_spacing'] * 1000
                return settings
        return grid_settings[-1][1]

    def _get_y_grid_settings(self, y_min: float, y_max: float) -> Dict:
        """Determine y-axis grid settings."""
        y_range = y_max - y_min
        if y_range == 0:
            return {'major_step': 1, 'minor_step': 0.5}
        
        steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        major_step = 1
        for step in steps:
            n_grids = y_range / step
            if 4 <= n_grids <= 7:
                major_step = step
                break
        
        if major_step in [0.2, 2, 20, 200]:
            minor_step = major_step / 2
        elif major_step in [0.5, 5, 50, 500]:
            minor_step = major_step / 5
        else:
            minor_step = major_step / 2
        
        return {'major_step': major_step, 'minor_step': minor_step}

    def _run(
        self,
        primary_y_instruments: List[InstrumentColumnPair],
        secondary_y_instruments: List[InstrumentColumnPair],
        start_time: datetime,
        end_time: datetime,
        primary_y_title: str,
        primary_y_unit: str,
        secondary_y_title: Optional[str],
        secondary_y_unit: Optional[str],
        review_level_values: List[float],
        highlight_zero: bool
    ) -> Tuple[str, List[Dict]]:
        logger.debug(f"TimeSeriesPlotTool._run called with {len(primary_y_instruments)} primary and {len(secondary_y_instruments)} secondary instruments")
        try:
            # Group instruments by column name for efficient querying
            column_groups = {}
            all_instrument_ids = []
            for instr in primary_y_instruments + secondary_y_instruments:
                col = instr.column_name
                if col not in column_groups:
                    column_groups[col] = []
                column_groups[col].append(instr.instrument_id)
                all_instrument_ids.append(instr.instrument_id)
            
            # Query for each column name
            time_series_data = {id: [] for id in all_instrument_ids}
            for column_name, instr_ids in column_groups.items():
                query_template = """
                SELECT 
                    date1 as timestamp,
                    {column_name} as value,
                    instr_id
                FROM mydata m
                WHERE m.instr_id IN ({instrument_ids})
                AND m.date1 BETWEEN '{start_time}' AND '{end_time}'
                AND {column_name} IS NOT NULL
                AND {column_name} != ''
                ORDER BY instr_id, date1;
                """ if column_name.startswith('data') else """
                SELECT 
                    date1 as timestamp,
                    JSON_EXTRACT(custom_fields, '$.{column_name}') as value,
                    instr_id
                FROM mydata m
                WHERE m.instr_id IN ({instrument_ids})
                AND m.date1 BETWEEN '{start_time}' AND '{end_time}'
                AND custom_fields IS NOT NULL
                AND JSON_VALID(custom_fields)
                AND JSON_EXTRACT(custom_fields, '$.{column_name}') IS NOT NULL
                AND JSON_EXTRACT(custom_fields, '$.{column_name}') != 'null'
                AND JSON_EXTRACT(custom_fields, '$.{column_name}') != ''
                ORDER BY instr_id, date1;
                """
                
                instrument_id_str = ','.join(f"'{id}'" for id in instr_ids)
                query = query_template.format(
                    column_name=column_name,
                    instrument_ids=instrument_id_str,
                    start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time=end_time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                logger.debug(f"Executing SQL query for {column_name}:\n{query}")
                results = self.sql_tool._run(query)
                logger.debug(f"Query results: {results}")
                
                if results == "No data was found in the database matching the specified search criteria.":
                    continue
                
                def clean_numeric_string(val_str):
                    cleaned = str(val_str).strip()
                    while cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
                        cleaned = cleaned[1:-1].strip()
                    if re.match(r'^-?\d*\.?\d+$', cleaned):
                        return cleaned
                    return None

                try:
                    parsed_data = eval(results, {"__builtins__": {}, "datetime": datetime_module}, {})
                    for dt, val_str, instr_id in parsed_data:
                        if instr_id not in time_series_data:
                            continue
                        cleaned_val = clean_numeric_string(val_str)
                        if cleaned_val is None:
                            continue
                        try:
                            val = float(cleaned_val)
                            time_series_data[instr_id].append((dt, val))
                        except ValueError:
                            continue
                except Exception as e:
                    content = f"Error processing plot: Error processing database results for {column_name}: {str(e)}"
                    return content, []
            
            if not any(time_series_data.values()):
                content = "No data found: No valid data points found for given instruments"
                return content, []
            
            logger.debug(f"Processed data for instruments: {list(time_series_data.keys())}")
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Colors for time series
            primary_colors = ['#1f77b4', '#4b9cd3', '#87ceeb', '#add8e6']  # Blue hues
            secondary_colors = ['#ff69b4', '#ff85c0', '#ffb6c1', '#ffc1cc']  # Pink hues
            
            # Plot time series for primary y-axis
            y_min, y_max = float('inf'), float('-inf')
            for i, instr in enumerate(primary_y_instruments):
                instr_id = instr.instrument_id
                if time_series_data[instr_id]:
                    times, values = zip(*time_series_data[instr_id])
                    y_min = min(y_min, min(values))
                    y_max = max(y_max, max(values))
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=values,
                        mode='lines+markers',
                        name=f"Instrument {instr_id} ({instr.column_name})",
                        line=dict(color=primary_colors[i % len(primary_colors)])
                    ))
            
            # Plot time series for secondary y-axis
            secondary_y_min, secondary_y_max = float('inf'), float('-inf')
            for i, instr in enumerate(secondary_y_instruments):
                instr_id = instr.instrument_id
                if time_series_data[instr_id]:
                    times, values = zip(*time_series_data[instr_id])
                    secondary_y_min = min(secondary_y_min, min(values))
                    secondary_y_max = max(secondary_y_max, max(values))
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=values,
                        mode='lines+markers',
                        name=f"Instrument {instr_id} ({instr.column_name})",
                        line=dict(color=secondary_colors[i % len(secondary_colors)]),
                        yaxis='y2'
                    ))
            
            # Plot review levels if provided
            if review_level_values:
                pos_levels = sorted([x for x in review_level_values if x > 0], reverse=True)
                neg_levels = sorted([x for x in review_level_values if x < 0])
                colors = ['#8B0000', '#FF4500', '#006400']
                for i, level in enumerate(pos_levels + neg_levels):
                    color_idx = min(i % 3, 2)
                    fig.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color=colors[color_idx],
                        annotation_text=f"Review Level {level}",
                        annotation_position="top right"
                    )
                    y_min = min(y_min, level)
                    y_max = max(y_max, level)
            
            # Highlight zero line
            if highlight_zero and not secondary_y_instruments:
                fig.add_hline(
                    y=0,
                    line_color='lightgrey',
                    line_width=1
                )
                y_min = min(y_min, 0)
                y_max = max(y_max, 0)
            
            # Set axis properties
            x_grid = self._get_x_grid_settings(start_time, end_time)
            primary_y_grid = self._get_y_grid_settings(y_min, y_max)
            
            layout = {
                'xaxis': {
                    'tickformat': x_grid['major_format'],
                    'dtick': x_grid['major_dtick'],
                    'minor': {'dtick': x_grid['minor_dtick'], 'showgrid': True, 'gridcolor': 'lightgrey'},
                    'showgrid': True,
                    'gridcolor': 'grey'
                },
                'yaxis': {
                    'title': f"{primary_y_title} ({primary_y_unit})",
                    'dtick': primary_y_grid['major_step'],
                    'minor': {'dtick': primary_y_grid['minor_step'], 'showgrid': True, 'gridcolor': 'lightgrey'},
                    'showgrid': True,
                    'gridcolor': 'grey'
                }
            }
            
            if secondary_y_instruments:
                layout['yaxis2'] = {
                    'title': f"{secondary_y_title} ({secondary_y_unit})",
                    'overlaying': 'y',
                    'side': 'right',
                    'dtick': self._get_y_grid_settings(secondary_y_min, secondary_y_max)['major_step'],
                    'minor': {
                        'dtick': self._get_y_grid_settings(secondary_y_min, secondary_y_max)['minor_step'],
                        'showgrid': True,
                        'gridcolor': 'lightgrey'
                    },
                    'showgrid': True,
                    'gridcolor': 'grey'
                }
            
            fig.update_layout(
                showlegend=True,
                template="plotly_white",
                **layout
            )
            
            # Create CSV
            all_data = []
            for instr in primary_y_instruments + secondary_y_instruments:
                instr_id = instr.instrument_id
                column_name = instr.column_name
                for dt, val in time_series_data[instr_id]:
                    all_data.append({
                        'Instrument ID': instr_id,
                        'Column Name': column_name,
                        'Timestamp': format_datetime(dt),
                        'Value': val,
                        'Y-Axis': 'Primary' if instr in primary_y_instruments else 'Secondary'
                    })
            df = pd.DataFrame(all_data)
            csv_filename = f"time_series_{uuid.uuid4()}.csv"
            csv_content = df.to_csv(index=False)
            
            # Prepare artefacts
            plotly_filename = f"time_series_{uuid.uuid4()}.json"
            artefacts = [
                {
                    'artifact_id': str(uuid.uuid4()),
                    'filename': plotly_filename,
                    'type': 'Plotly object',
                    'description': f"Time series plot for instruments {', '.join(all_instrument_ids)} from {format_datetime(start_time)} to {format_datetime(end_time)}",
                    'content': fig.to_json()
                },
                {
                    'artifact_id': str(uuid.uuid4()),
                    'filename': csv_filename,
                    'type': 'CSV',
                    'description': f"CSV data for time series plot of instruments {', '.join(all_instrument_ids)}",
                    'content': csv_content
                }
            ]
            
            content = f"Generated time series plot and CSV for instruments {', '.join(all_instrument_ids)} from {format_datetime(start_time)} to {format_datetime(end_time)}."
            return content, artefacts
        
        except Exception as e:
            logger.error(f"Error in TimeSeriesPlotTool: {str(e)}")
            content = f"Error processing plot: {str(e)}"
            return content, []

class TimeSeriesPlotWrapperInput(BaseModel):
    """Input model for the time series plotting wrapper, accepting a JSON string to create plots."""
    input_json: str = Field(
        ...,
        description="""A JSON string specifying the parameters for a time series plot. The JSON must be a single object containing the following fields:
        - 'primary_y_instruments': A list of objects, each with 'instrument_id' (string, e.g., 'INST001') and 'column_name' (string, e.g., 'data1' or 'calculation1'). At least one object is required. Example: [{'instrument_id': 'INST001', 'column_name': 'data1'}].
        - 'secondary_y_instruments': An optional list of objects, each with 'instrument_id' and 'column_name', for the secondary y-axis. Example: [{'instrument_id': 'INST002', 'column_name': 'data2'}].
        - 'start_time': A datetime string in the format 'D Month YYYY H:MM:SS AM/PM' (e.g., '1 January 2025 12:00:00 PM') for the data start time.
        - 'end_time': A datetime string in the format 'D Month YYYY H:MM:SS AM/PM' (e.g., '31 May 2025 2:00:00 PM') for the data end time, must be later than start_time.
        - 'primary_y_title': A string for the primary y-axis title (e.g., 'Temperature').
        - 'primary_y_unit': A string for the primary y-axis unit (e.g., '°C').
        - 'secondary_y_title': An optional string for the secondary y-axis title, required if secondary_y_instruments is non-empty (e.g., 'Pressure').
        - 'secondary_y_unit': An optional string for the secondary y-axis unit, required if secondary_y_instruments is non-empty (e.g., 'kPa').
        - 'review_level_values': An optional list of floats for review level lines on the primary y-axis (e.g., [10.0, -5.0]). Max 3 positive and 3 negative values. Cannot be used with secondary_y_instruments.
        - 'highlight_zero': An optional boolean to highlight the zero line on the primary y-axis (default: false). Only used if secondary_y_instruments is empty.
        The total number of instrument-column pairs (primary + secondary) cannot exceed 7. Example JSON:
        ```json
        {
            "primary_y_instruments": [{"instrument_id": "INST001", "column_name": "data1"}, {"instrument_id": "INST002", "column_name": "calculation1"}],
            "secondary_y_instruments": [{"instrument_id": "INST003", "column_name": "data2"}],
            "start_time": "1 August 2025 12:00:00 AM",
            "end_time": "31 August 2025 11:59:59 PM",
            "primary_y_title": "Temperature",
            "primary_y_unit": "°C",
            "secondary_y_title": "Pressure",
            "secondary_y_unit": "kPa",
            "review_level_values": [],
            "highlight_zero": false
        }
        The JSON string must be valid and properly formatted."""
    )

class TimeSeriesPlotWrapperTool(BaseTool):
    """Wrapper for TimeSeriesPlotTool to handle JSON input."""
    name: str = "time_series_plot_wrapper"
    description: str = """
    Wrapper that accepts JSON string to create time series plots.
    Saves the plot as an HTML file in the same directory as this tool for verification and returns a JSON string containing content and artefacts with response_format='content_and_artifact'.
    The artefacts include the Plotly JSON, CSV, and HTML file.
    """
    args_schema: Type[TimeSeriesPlotWrapperInput] = TimeSeriesPlotWrapperInput
    plot_tool: TimeSeriesPlotTool = Field(exclude=True)
    response_format: str = "content"

    def _normalize_boolean_values(self, json_str: str) -> str:
        """Normalize Python-style boolean values to JSON-style boolean values."""
        # Replace Python's True/False with JSON's true/false, being careful to match whole words only
        json_str = re.sub(r':\s*True\b', ': true', json_str)
        json_str = re.sub(r':\s*False\b', ': false', json_str)
        return json_str

    def _run(self, input_json: str) -> str:
        logger.debug(f"TimeSeriesPlotWrapperTool._run called")
        logger.debug(f"Input JSON: {input_json}")
        try:
            # Remove JSON code block markers if present
            input_json = re.sub(r'^```json\s*\n|\s*```$', '', input_json, flags=re.MULTILINE)
            # Replace single quotes with double quotes
            input_json = input_json.replace("'", "\"")
            # Replace "None" with "null" for JSON compatibility
            input_json = re.sub(r':\s*None\b', ': null', input_json)
            # Normalize boolean values
            input_json = self._normalize_boolean_values(input_json)
            input_dict = json.loads(input_json)
            
            required_fields = ['primary_y_instruments', 'start_time', 'end_time', 'primary_y_title', 'primary_y_unit']
            missing = [f for f in required_fields if f not in input_dict]
            if missing:
                content = f"Error processing input: Missing required fields: {', '.join(missing)}"
                return content, []
            logger.debug(f"Parsed input dictionary: {input_dict}")
            
            try:
                start_time = parse_datetime(input_dict['start_time'])
                end_time = parse_datetime(input_dict['end_time'])
                primary_y_instruments = [InstrumentColumnPair(**pair) for pair in input_dict['primary_y_instruments']]
                secondary_y_instruments = [InstrumentColumnPair(**pair) for pair in input_dict.get('secondary_y_instruments', [])]
            except ValueError as e:
                content = f"Error processing input: {str(e)}"
                return content, []

            plot_tool_result = self.plot_tool._run(
                primary_y_instruments=primary_y_instruments,
                secondary_y_instruments=secondary_y_instruments,
                start_time=start_time,
                end_time=end_time,
                primary_y_title=input_dict['primary_y_title'],
                primary_y_unit=input_dict['primary_y_unit'],
                secondary_y_title=input_dict.get('secondary_y_title'),
                secondary_y_unit=input_dict.get('secondary_y_unit'),
                review_level_values=input_dict.get('review_level_values', []),
                highlight_zero=input_dict.get('highlight_zero', False)
            )
            logger.debug(f"Plot tool result: {plot_tool_result}")

            content, artifacts = plot_tool_result
            result = {'content': content, 'artifacts': artifacts}
            return json.dumps(result)
        except Exception as e:
            content = f"Error processing input: {str(e)}"
            result = {'content': content, 'artifacts': []}
            return json.dumps(result)