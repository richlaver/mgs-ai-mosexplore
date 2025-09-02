import json
import logging
import math
import os
import re
import uuid
from datetime import datetime, timedelta
import datetime as datetime_module
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, validator
from pyproj.crs import CRS
from pyproj.transformer import Transformer

from .sql_security_toolkit import GeneralSQLQueryTool
from .get_trend_info_toolkit import parse_datetime, format_datetime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def clean_numeric_string(val):
    if not isinstance(val, (str, int, float)):
        return None
    cleaned = str(val).strip()
    while cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
        cleaned = cleaned[1:-1].strip()
    if re.match(r'^-?\d*\.?\d+(?:[eE][-+]?\d+)?$', cleaned):
        return cleaned
    return None

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

# Hanoi local coordinate system: HN-72, approximated with EPSG:4147 for zone
hn72_crs = CRS.from_epsg(4147)  # HN-72
wgs84_crs = CRS.from_epsg(4326)  # WGS84
transformer = Transformer.from_crs(hn72_crs, wgs84_crs, always_xy=True)

def easting_northing_to_lat_lon(easting: float, northing: float) -> Tuple[float, float]:
    """Convert easting and northing from HN-72 to latitude and longitude using WGS84."""
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

class SeriesDict(BaseModel):
    """Model for specifying a data series to plot on the map.
    
    This defines one type of instrument data to include in the map plot, represented as a dictionary.
    
    Examples:
    - {"instrument_type": "VWP", "instrument_subtype": "DEFAULT", "database_field_name": "calculation1", "measured_quantity_name": "Pore Pressure", "abbreviated_unit": "kPa"}
    
    Constraints:
    - All fields must be non-empty strings.
    - database_field_name must be in format 'dataN' or 'calculationN' where N is integer.
    """
    instrument_type: str = Field(
        ...,
        description="The type of instrument, corresponding to 'type1' in instrum table (e.g., 'VWP' for Vibrating Wire Piezometer). Must match existing types in the database."
    )
    instrument_subtype: str = Field(
        ...,
        description="The subtype of instrument, corresponding to 'subtype1' in instrum table (e.g., 'DEFAULT'). Must match existing subtypes."
    )
    database_field_name: str = Field(
        ...,
        description="The field name in mydata table to plot, e.g., 'data1' for raw data or 'calculation1' for computed values. For 'calculationN', value is extracted from custom_fields JSON."
    )
    measured_quantity_name: str = Field(
        ...,
        description="Descriptive name for the quantity being measured, used in legend (e.g., 'Settlement')."
    )
    abbreviated_unit: str = Field(
        ...,
        description="Abbreviated unit for the quantity, used in legend (e.g., 'mm')."
    )

    @validator('database_field_name')
    def validate_field_name(cls, v):
        if not re.match(r'^(data|calculation)\d+$', v):
            raise ValueError("database_field_name must be 'dataN' or 'calculationN'")
        return v

class MapPlotInput(BaseModel):
    """Input schema for the map plotting tool.
    
    Use this tool to create a geospatial plot of instrument data on a map using Plotly's Mapbox.
    The tool extracts data via SQL queries, converts local coordinates to lat/lon, filters spatially, and plots markers colored by values or review statuses.
    
    When to use:
    - When the user wants a map view of instrument readings or review levels.
    - For visualizing spatial patterns, clusters, or changes over time in a geographic context.
    - Supports up to 3 series (instrument type/subtype/field combinations).
    - For 'readings': Plots numerical values or changes, with sequential or divergent colormaps.
    - For 'review_levels': Plots breach statuses as colors (green/amber/red/grey), or worsened statuses.
    
    Input formatting:
    - Times as datetime objects, but in wrapper, parse from strings like '1 January 2025 12:00:00 PM'.
    - series as list of dicts in JSON, e.g., [{"instrument_type": "VWP", "instrument_subtype": "DEFAULT", ...}]
    - Center: Either instrument_id or easting+northing.
    
    Constraints:
    - Max 3 series.
    - buffer_period_hours <= half the period for changes, no limit for single time.
    - radius_meters > 0.
    - For review_levels, max 3 levels per side (pos/neg).
    
    Output:
    - Returns (content str, list of artifacts: Plotly JSON and CSV).
    - Content includes any suggested outliers for exclusion (for readings only).
    """
    data_type: str = Field(
        ...,
        description="Type of data: 'review_levels' for breach statuses, 'readings' for numerical values. Determines coloring and data extraction logic."
    )
    plot_type: str = Field(
        ...,
        description=" 'value_at_time' for snapshot at end_time, 'change_over_period' for delta or worsened status between start_time and end_time."
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Start datetime for 'change_over_period'. Required for changes, ignored for 'value_at_time'. Format in wrapper: 'D Month YYYY H:MM:SS AM/PM'."
    )
    end_time: datetime = Field(
        ...,
        description="End datetime (or single time for 'value_at_time'). Must be after start_time if provided."
    )
    buffer_period_hours: int = Field(
        default=72,
        description="Hours before specified times to search for nearest reading. Max half period for changes. Example: 72 for three days."
    )
    series: List[SeriesDict] = Field(
        ...,
        description="List of up to 3 series to plot. Each specifies instrument type/subtype/field and legend info."
    )
    center_instrument_id: Optional[str] = Field(
        None,
        description="Instrument ID to center map on (fetches its location). Alternative to easting/northing."
    )
    center_easting: Optional[float] = Field(
        None,
        description="Easting coordinate for map center (local system). Required if no center_instrument_id."
    )
    center_northing: Optional[float] = Field(
        None,
        description="Northing coordinate for map center. Required if no center_instrument_id."
    )
    radius_meters: float = Field(
        ...,
        description="Radius for spatial filter (square bounding box of side 2*radius). Data outside ignored. >0."
    )
    exclude_instrument_ids: List[str] = Field(
        default_factory=list,
        description="List of instrument IDs to exclude from plot and data extraction."
    )

    @validator('buffer_period_hours')
    def validate_buffer(cls, v, values):
        if 'plot_type' in values and values['plot_type'] == 'change_over_period' and 'start_time' in values and 'end_time' in values:
            period_hours = (values['end_time'] - values['start_time']).total_seconds() / 3600
            if v > period_hours / 2:
                raise ValueError("Buffer cannot exceed half the period for changes")
        return v

    @validator('data_type')
    def validate_data_type(cls, v):
        if v not in ['review_levels', 'readings']:
            raise ValueError("data_type must be 'review_levels' or 'readings'")
        return v

    @validator('plot_type')
    def validate_plot_type(cls, v):
        if v not in ['value_at_time', 'change_over_period']:
            raise ValueError("plot_type must be 'value_at_time' or 'change_over_period'")
        return v

    @validator('series')
    def validate_series(cls, v):
        if len(v) == 0 or len(v) > 3:
            raise ValueError("1-3 series required")
        return v

class BaseSQLQueryTool(BaseModel):
    sql_tool: GeneralSQLQueryTool = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pyproj import Proj, transform

from .sql_security_toolkit import GeneralSQLQueryTool
from .get_trend_info_toolkit import format_datetime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def clean_numeric_string(val):
    if not isinstance(val, (str, int, float)):
        return None
    cleaned = str(val).strip()
    while cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
        cleaned = cleaned[1:-1].strip()
    if re.match(r'^-?\d*\.?\d+(?:[eE][-+]?\d+)?$', cleaned):
        return cleaned
    return None

hn72_crs = CRS.from_epsg(3405)  # VN-2000 / Vietnam TM-3 zone 105-45
wgs84_crs = CRS.from_epsg(4326)  # WGS84
transformer = Transformer.from_crs(hn72_crs, wgs84_crs, always_xy=True)

def easting_northing_to_lat_lon(easting: float, northing: float) -> Tuple[float, float]:
    """Convert easting and northing from VN-2000 to latitude and longitude using WGS84."""
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

class SeriesDict(BaseModel):
    """Model for specifying a data series to plot on the map."""
    instrument_type: str = Field(
        ...,
        description="The type of instrument, corresponding to 'type1' in instrum table (e.g., 'VWP'). Must match existing types in the database."
    )
    instrument_subtype: str = Field(
        ...,
        description="The subtype of instrument, corresponding to 'subtype1' in instrum table (e.g., 'DEFAULT'). Must match existing subtypes."
    )
    database_field_name: str = Field(
        ...,
        description="The field name in mydata table to plot, e.g., 'data1' or 'calculation1'. For 'calculationN', value is extracted from custom_fields JSON."
    )
    measured_quantity_name: str = Field(
        ...,
        description="Descriptive name for the quantity being measured, used in legend (e.g., 'Settlement')."
    )
    abbreviated_unit: str = Field(
        ...,
        description="Abbreviated unit for the quantity, used in legend (e.g., 'mm')."
    )

    @validator('database_field_name')
    def validate_field_name(cls, v):
        if not re.match(r'^(data|calculation)\d+$', v):
            raise ValueError("database_field_name must be 'dataN' or 'calculationN'")
        return v

class MapPlotInput(BaseModel):
    """Input schema for the map plotting tool."""
    data_type: str = Field(
        ...,
        description="Type of data: 'review_levels' for breach statuses, 'readings' for numerical values."
    )
    plot_type: str = Field(
        ...,
        description="'value_at_time' for snapshot at end_time, 'change_over_period' for delta or worsened status."
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Start datetime for 'change_over_period'. Required for changes, ignored for 'value_at_time'."
    )
    end_time: datetime = Field(
        ...,
        description="End datetime (or single time for 'value_at_time'). Must be after start_time if provided."
    )
    buffer_period_hours: int = Field(
        default=72,
        description="Hours before specified times to search for nearest reading."
    )
    series: List[SeriesDict] = Field(
        ...,
        description="List of up to 3 series to plot."
    )
    center_instrument_id: Optional[str] = Field(
        None,
        description="Instrument ID to center map on."
    )
    center_easting: Optional[float] = Field(
        None,
        description="Easting coordinate for map center."
    )
    center_northing: Optional[float] = Field(
        None,
        description="Northing coordinate for map center."
    )
    radius_meters: float = Field(
        ...,
        description="Radius for spatial filter (square bounding box of side 2*radius). >0."
    )
    exclude_instrument_ids: List[str] = Field(
        default_factory=list,
        description="List of instrument IDs to exclude."
    )

    @validator('buffer_period_hours')
    def validate_buffer(cls, v, values):
        if 'plot_type' in values and values['plot_type'] == 'change_over_period' and 'start_time' in values and 'end_time' in values:
            period_hours = (values['end_time'] - values['start_time']).total_seconds() / 3600
            if v > period_hours / 2:
                raise ValueError("Buffer cannot exceed half the period for changes")
        return v

    @validator('data_type')
    def validate_data_type(cls, v):
        if v not in ['review_levels', 'readings']:
            raise ValueError("data_type must be 'review_levels' or 'readings'")
        return v

    @validator('plot_type')
    def validate_plot_type(cls, v):
        if v not in ['value_at_time', 'change_over_period']:
            raise ValueError("plot_type must be 'value_at_time' or 'change_over_period'")
        return v

    @validator('series')
    def validate_series(cls, v):
        if len(v) == 0 or len(v) > 3:
            raise ValueError("1-3 series required")
        return v

class BaseSQLQueryTool(BaseModel):
    sql_tool: GeneralSQLQueryTool = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class MapPlotTool(BaseTool, BaseSQLQueryTool):
    """Tool for plotting instrument readings or review status on a map with Plotly."""
    name: str = "map_plot"
    description: str = """
    Creates an interactive Plotly map plot and CSV file from instrumentation data.
    Supports readings or review status, for a single time or change over a period, within a geographic radius.
    Returns a tuple of (content, artefacts) with response_format='content_and_artifact'.
    The artefacts include the Plotly JSON, CSV, and HTML file.
    """
    args_schema: Type[MapPlotInput] = MapPlotInput
    response_format: str = "content_and_artifact"

    def _get_center_coords(self, center_instrument_id: Optional[str], center_easting: Optional[float], center_northing: Optional[float]) -> Tuple[float, float]:
        if center_instrument_id:
            query = f"""
            SELECT l.easting, l.northing
            FROM instrum i
            JOIN location l ON i.location_id = l.id
            WHERE i.instr_id = '{center_instrument_id}';
            """
            result = self.sql_tool._run(query)
            if result == "No data was found in the database matching the specified search criteria.":
                raise ValueError(f"No location found for instrument ID {center_instrument_id}")
            try:
                data = eval(result, {"__builtins__": {}, "datetime": datetime_module}, {})
                easting, northing = data[0]
                easting_clean = clean_numeric_string(easting)
                northing_clean = clean_numeric_string(northing)
                if None in (easting_clean, northing_clean):
                    raise ValueError(f"Invalid easting or northing for instrument ID {center_instrument_id}")
                return float(easting_clean), float(northing_clean)
            except Exception as e:
                raise ValueError(f"Error processing location for instrument ID {center_instrument_id}: {str(e)}")
        elif center_easting is not None and center_northing is not None:
            return center_easting, center_northing
        else:
            raise ValueError("Either center_instrument_id or both center_easting and center_northing must be provided")

    def _get_bounds(self, center_e: float, center_n: float, radius_m: float) -> Tuple[float, float, float, float]:
        min_e = center_e - radius_m
        max_e = center_e + radius_m
        min_n = center_n - radius_m
        max_n = center_n + radius_m
        return min_e, max_e, min_n, max_n

    def _get_review_config(self, field: str) -> Dict:
        pos_values = [100, 50, 20]
        neg_values = [-20, -50, -100]
        severity_map = {100: 3, 50: 2, 20: 1, -20: 1, -50: 2, -100: 3}
        color_map = {100: 'red', 50: 'orange', 20: 'yellow', -20: 'yellow', -50: 'orange', -100: 'red'}
        return {'pos_values': pos_values, 'neg_values': neg_values, 'severity_map': severity_map, 'color_map': color_map}

    def _get_readings(self, inputs: MapPlotInput, s: SeriesDict, min_e: float, max_e: float, min_n: float, max_n: float, exclude_clause: str, 
                  start_time_str: Optional[str], end_time_str: str, buffer_start: Optional[str], buffer_end: str) -> List[Tuple]:
        field = s.database_field_name
        is_data = field.startswith('data')
        
        # Define value extraction logic
        if is_data:
            field_extract = field
            value_extract = field
            extra_conditions = f"AND {field} != ''"
        else:
            field_path = f'$.{field}'
            value_extract = f"CASE WHEN m.custom_fields IS NOT NULL AND LENGTH(m.custom_fields) > 0 AND JSON_VALID(m.custom_fields) THEN JSON_EXTRACT(m.custom_fields, '{field_path}') ELSE NULL END"
            extra_conditions = f"AND m.custom_fields IS NOT NULL AND LENGTH(m.custom_fields) > 0 AND JSON_VALID(m.custom_fields) " \
                            f"AND {value_extract} IS NOT NULL " \
                            f"AND {value_extract} != 'null' " \
                            f"AND {value_extract} != ''"
        
        safe_eval = lambda s: eval(s, {"__builtins__": {}}, {}) if s != "No data was found in the database matching the specified search criteria." else []

        if inputs.plot_type == 'value_at_time':
            query = f"""
            SELECT i.instr_id, l.easting, l.northing, {value_extract} as value
            FROM instrum i
            JOIN location l ON i.location_id = l.id
            JOIN mydata m ON i.instr_id = m.instr_id
            WHERE i.type1 = '{s.instrument_type}' AND i.subtype1 = '{s.instrument_subtype}'
            AND l.easting BETWEEN {min_e} AND {max_e}
            AND l.northing BETWEEN {min_n} AND {max_n}
            {exclude_clause}
            AND m.date1 = (SELECT MAX(date1) FROM mydata WHERE instr_id = i.instr_id AND date1 BETWEEN '{buffer_end}' AND '{end_time_str}')
            {extra_conditions};
            """
            logger.debug(f"Executing readings query: {query}")
            raw_result = self.sql_tool._run(query)
            logger.debug(f"Raw SQL result: {raw_result!r}")
            raw_data = safe_eval(raw_result)
            logger.debug(f"Parsed eval result: {raw_data}")
            if not raw_data:
                logger.warning(f"No data returned for series {s.instrument_type}_{s.instrument_subtype}_{s.database_field_name}")
                return []
            data = []
            for item in raw_data:
                logger.debug(f"Processing item: {item}")
                if len(item) != 4:
                    logger.warning(f"Skipping item with incorrect length: {item}")
                    continue
                id_, e, n, v = item
                logger.debug(f"Raw values: id={id_!r}, easting={e!r}, northing={n!r}, value={v!r}")
                e_clean = clean_numeric_string(e)
                n_clean = clean_numeric_string(n)
                v_clean = clean_numeric_string(v)
                logger.debug(f"Cleaned values: easting={e_clean!r}, northing={n_clean!r}, value={v_clean!r}")
                if None in (e_clean, n_clean, v_clean):
                    logger.warning(f"Invalid cleaned values: easting={e!r}, northing={n!r}, value={v!r}")
                    continue
                try:
                    data.append((id_, float(e_clean), float(n_clean), float(v_clean)))
                    logger.debug(f"Appended valid data point: {id_}, {e_clean}, {n_clean}, {v_clean}")
                except ValueError as e:
                    logger.warning(f"ValueError converting values: {e}, item: {item}")
                    continue
            logger.debug(f"Collected {len(data)} valid data points")
            return data
        else:
            query_end = f"""
            SELECT i.instr_id, {value_extract} as value_end
            FROM instrum i JOIN location l ON i.location_id = l.id JOIN mydata m ON i.instr_id = m.instr_id
            WHERE i.type1 = '{s.instrument_type}' AND i.subtype1 = '{s.instrument_subtype}'
            AND l.easting BETWEEN {min_e} AND {max_e} AND l.northing BETWEEN {min_n} AND {max_n}
            {exclude_clause}
            AND m.date1 = (SELECT MAX(date1) FROM mydata WHERE instr_id = i.instr_id AND date1 BETWEEN '{buffer_end}' AND '{end_time_str}')
            {extra_conditions};
            """
            logger.debug(f'End readings SQL extraction result: {self.sql_tool._run(query_end)}')
            raw_end = safe_eval(self.sql_tool._run(query_end))
            result_end = []
            for item in raw_end:
                if len(item) != 2:
                    continue
                instr_id, val_end = item
                val_end_clean = clean_numeric_string(val_end)
                if val_end_clean is None:
                    continue
                try:
                    result_end.append((instr_id, float(val_end_clean)))
                except ValueError:
                    continue

            query_start = f"""
            SELECT i.instr_id, {value_extract} as value_start
            FROM instrum i JOIN location l ON i.location_id = l.id JOIN mydata m ON i.instr_id = m.instr_id
            WHERE i.type1 = '{s.instrument_type}' AND i.subtype1 = '{s.instrument_subtype}'
            AND l.easting BETWEEN {min_e} AND {max_e} AND l.northing BETWEEN {min_n} AND {max_n}
            {exclude_clause}
            AND m.date1 = (SELECT MAX(date1) FROM mydata WHERE instr_id = i.instr_id AND date1 BETWEEN '{buffer_start}' AND '{start_time_str}')
            {extra_conditions};
            """
            raw_start = safe_eval(self.sql_tool._run(query_start))
            result_start = []
            for item in raw_start:
                if len(item) != 2:
                    continue
                instr_id, val_start = item
                val_start_clean = clean_numeric_string(val_start)
                if val_start_clean is None:
                    continue
                try:
                    result_start.append((instr_id, float(val_start_clean)))
                except ValueError:
                    continue

            changes = {}
            for instr_id, val_end in result_end:
                val_start = next((v for id_, v in result_start if id_ == instr_id), None)
                if val_start is not None:
                    changes[instr_id] = val_end - val_start
            if not changes:
                return []
            instr_ids_str = ','.join(f"'{id_}'" for id_ in changes)
            loc_query = f"""
            SELECT i.instr_id, l.easting, l.northing
            FROM instrum i JOIN location l ON i.location_id = l.id
            WHERE i.instr_id IN ({instr_ids_str});
            """
            logger.debug(f'Location query results: {self.sql_tool._run(loc_query)}')
            raw_locs = safe_eval(self.sql_tool._run(loc_query))
            loc_dict = {}
            for item in raw_locs:
                if len(item) != 3:
                    continue
                id_, e, n = item
                e_clean = clean_numeric_string(e)
                n_clean = clean_numeric_string(n)
                if None in (e_clean, n_clean):
                    continue
                try:
                    loc_dict[id_] = (float(e_clean), float(n_clean))
                except ValueError:
                    continue
            return [(id_, loc_dict[id_][0], loc_dict[id_][1], changes[id_]) for id_ in changes if id_ in loc_dict]

    def _get_review_status(self, value: float, review_config: Dict) -> Tuple[int, str]:
        pos = review_config['pos_values']
        neg = review_config['neg_values']
        abs_thresh = 0
        if value > 0:
            for t in pos:
                if value > t:
                    abs_thresh = max(abs_thresh, t)
        elif value < 0:
            for t in neg:
                if value < t:
                    abs_thresh = max(abs_thresh, abs(t))
        severity = review_config['severity_map'].get(abs_thresh, 0)
        color = review_config['color_map'].get(abs_thresh, 'grey')
        return severity, color

    def _get_reviews(self, inputs: MapPlotInput, s: SeriesDict, min_e: float, max_e: float, min_n: float, max_n: float, exclude_clause: str, 
                     start_time_str: Optional[str], end_time_str: str, buffer_start: Optional[str], buffer_end: str) -> List[Tuple]:
        review_config = self._get_review_config(s.database_field_name)
        field = s.database_field_name
        is_data = field.startswith('data')
        field_extract = field if is_data else f"JSON_EXTRACT(m.custom_fields, '$.{field}')"  # AMENDED: Added m. to custom_fields
        extra_conditions = f"AND {field_extract} != ''" if is_data else f"AND m.custom_fields IS NOT NULL AND JSON_VALID(m.custom_fields) AND {field_extract} IS NOT NULL AND {field_extract} != 'null' AND {field_extract} != ''"  # AMENDED: Added m. to custom_fields
        safe_eval = lambda s: eval(s, {"__builtins__": {}}, {}) if s != "No data was found in the database matching the specified search criteria." else []

        if inputs.plot_type == 'value_at_time':
            query = f"""
            SELECT i.instr_id, l.easting, l.northing, {field_extract} as value
            FROM instrum i
            JOIN location l ON i.location_id = l.id
            JOIN mydata m ON i.instr_id = m.instr_id
            JOIN review_instruments ri ON i.instr_id = ri.instr_id
            WHERE i.type1 = '{s.instrument_type}' AND i.subtype1 = '{s.instrument_subtype}'
            AND l.easting BETWEEN {min_e} AND {max_e}
            AND l.northing BETWEEN {min_n} AND {max_n}
            {exclude_clause}
            AND ri.review_field = '{s.database_field_name}'
            AND m.date1 = (SELECT MAX(date1) FROM mydata WHERE instr_id = i.instr_id AND date1 BETWEEN '{buffer_end}' AND '{end_time_str}')
            AND {field_extract} IS NOT NULL {extra_conditions};
            """
            raw_result = self.sql_tool._run(query)
            raw_data = safe_eval(raw_result)
            data = []
            for item in raw_data:
                if len(item) != 4:
                    continue
                id_, e, n, v = item
                e_clean = clean_numeric_string(e)
                n_clean = clean_numeric_string(n)
                v_clean = clean_numeric_string(v)
                if None in (e_clean, n_clean, v_clean):
                    continue
                try:
                    v_float = float(v_clean)
                    data.append((id_, float(e_clean), float(n_clean), self._get_review_status(v_float, review_config)[1]))
                except ValueError:
                    continue
            return data
        else:
            query_end = f"""
            SELECT i.instr_id, {field_extract} as value_end
            FROM instrum i JOIN location l ON i.location_id = l.id JOIN mydata m ON i.instr_id = m.instr_id
            JOIN review_instruments ri ON i.instr_id = ri.instr_id
            WHERE i.type1 = '{s.instrument_type}' AND i.subtype1 = '{s.instrument_subtype}'
            AND l.easting BETWEEN {min_e} AND {max_e} AND l.northing BETWEEN {min_n} AND {max_n}
            {exclude_clause}
            AND ri.review_field = '{s.database_field_name}'
            AND m.date1 = (SELECT MAX(date1) FROM mydata WHERE instr_id = i.instr_id AND date1 BETWEEN '{buffer_end}' AND '{end_time_str}')
            AND {field_extract} IS NOT NULL {extra_conditions};
            """
            raw_end = safe_eval(self.sql_tool._run(query_end))
            result_end = []
            for item in raw_end:
                if len(item) != 2:
                    continue
                instr_id, val_end = item
                val_end_clean = clean_numeric_string(val_end)
                if val_end_clean is None:
                    continue
                try:
                    result_end.append((instr_id, float(val_end_clean)))
                except ValueError:
                    continue

            query_start = f"""
            SELECT i.instr_id, {field_extract} as value_start
            FROM instrum i JOIN location l ON i.location_id = l.id JOIN mydata m ON i.instr_id = m.instr_id
            JOIN review_instruments ri ON i.instr_id = ri.instr_id
            WHERE i.type1 = '{s.instrument_type}' AND i.subtype1 = '{s.instrument_subtype}'
            AND l.easting BETWEEN {min_e} AND {max_e} AND l.northing BETWEEN {min_n} AND {max_n}
            {exclude_clause}
            AND ri.review_field = '{s.database_field_name}'
            AND m.date1 = (SELECT MAX(date1) FROM mydata WHERE instr_id = i.instr_id AND date1 BETWEEN '{buffer_start}' AND '{start_time_str}')
            AND {field_extract} IS NOT NULL {extra_conditions};
            """
            raw_start = safe_eval(self.sql_tool._run(query_start))
            result_start = []
            for item in raw_start:
                if len(item) != 2:
                    continue
                instr_id, val_start = item
                val_start_clean = clean_numeric_string(val_start)
                if val_start_clean is None:
                    continue
                try:
                    result_start.append((instr_id, float(val_start_clean)))
                except ValueError:
                    continue

            statuses = {}
            for instr_id, val_end in result_end:
                val_start = next((v for id_, v in result_start if id_ == instr_id), None)
                if val_start is not None:
                    sev_start, _ = self._get_review_status(val_start, review_config)
                    sev_end, color_end = self._get_review_status(val_end, review_config)
                    if sev_end > sev_start:
                        statuses[instr_id] = color_end
                    else:
                        statuses[instr_id] = 'grey'
            if not statuses:
                return []
            instr_ids_str = ','.join(f"'{id_}'" for id_ in statuses)
            loc_query = f"""
            SELECT i.instr_id, l.easting, l.northing
            FROM instrum i JOIN location l ON i.location_id = l.id
            WHERE i.instr_id IN ({instr_ids_str});
            """
            raw_locs = safe_eval(self.sql_tool._run(loc_query))
            loc_dict = {}
            for item in raw_locs:
                if len(item) != 3:
                    continue
                id_, e, n = item
                e_clean = clean_numeric_string(e)
                n_clean = clean_numeric_string(n)
                if None in (e_clean, n_clean):
                    continue
                try:
                    loc_dict[id_] = (float(e_clean), float(n_clean))
                except ValueError:
                    continue
            return [(id_, loc_dict.get(id_, (0,0))[0], loc_dict.get(id_, (0,0))[1], statuses[id_]) for id_ in statuses if id_ in loc_dict]

    def _run(
    self,
    data_type: str,
    plot_type: str,
    start_time: Optional[datetime],
    end_time: datetime,
    buffer_period_hours: int,
    series: List[SeriesDict],
    center_instrument_id: Optional[str],
    center_easting: Optional[float],
    center_northing: Optional[float],
    radius_meters: float,
    exclude_instrument_ids: List[str]
) -> Tuple[str, List[Dict]]:
        logger.debug(f"Starting MapPlotTool._run with inputs: data_type={data_type}, plot_type={plot_type}, start_time={start_time}, end_time={end_time}")
        inputs = MapPlotInput(
            data_type=data_type,
            plot_type=plot_type,
            start_time=start_time,
            end_time=end_time,
            buffer_period_hours=buffer_period_hours,
            series=series,
            center_instrument_id=center_instrument_id,
            center_easting=center_easting,
            center_northing=center_northing,
            radius_meters=radius_meters,
            exclude_instrument_ids=exclude_instrument_ids
        )
        try:
            logger.debug("Fetching center coordinates")
            center_e, center_n = self._get_center_coords(inputs.center_instrument_id, inputs.center_easting, inputs.center_northing)
            logger.debug(f"Center coords: easting={center_e}, northing={center_n}")
            min_e, max_e, min_n, max_n = self._get_bounds(center_e, center_n, inputs.radius_meters)
            logger.debug(f"Bounds: min_e={min_e}, max_e={max_e}, min_n={min_n}, max_n={max_n}")
            exclude_str = ','.join(f"'{id}'" for id in inputs.exclude_instrument_ids)
            exclude_clause = f"AND i.instr_id NOT IN ({exclude_str})" if exclude_str else ""
            start_time_str = inputs.start_time.strftime("%Y-%m-%d %H:%M:%S") if inputs.start_time else None
            end_time_str = inputs.end_time.strftime("%Y-%m-%d %H:%M:%S")
            buffer_end = (inputs.end_time - timedelta(hours=inputs.buffer_period_hours)).strftime("%Y-%m-%d %H:%M:%S")
            buffer_start = (inputs.start_time - timedelta(hours=inputs.buffer_period_hours)).strftime("%Y-%m-%d %H:%M:%S") if inputs.start_time else None
            logger.debug(f"Time strings: start_time={start_time_str}, end_time={end_time_str}, buffer_start={buffer_start}, buffer_end={buffer_end}")

            data = {}
            suggestions = []
            for s in inputs.series:
                key = f"{s.instrument_type}_{s.instrument_subtype}_{s.database_field_name}"
                logger.debug(f"Processing series: {key}")
                if inputs.data_type == 'readings':
                    logger.debug("Fetching readings")
                    points = self._get_readings(inputs, s, min_e, max_e, min_n, max_n, exclude_clause, start_time_str, end_time_str, buffer_start, buffer_end)
                    if points:
                        vals = [v for _, _, _, v in points]
                        logger.debug(f"Values for outlier detection: {vals}")
                        mean = np.mean(vals)
                        std = np.std(vals)
                        outliers = [id for id, _, _, v in points if abs(v - mean) > 3 * std]
                        logger.debug(f"Outliers detected: {outliers}")
                        suggestions.extend(outliers)
                else:
                    logger.debug("Fetching reviews")
                    points = self._get_reviews(inputs, s, min_e, max_e, min_n, max_n, exclude_clause, start_time_str, end_time_str, buffer_start, buffer_end)
                data[key] = []
                for p in points:
                    id, e, n, v = p
                    try:
                        lat, lon = easting_northing_to_lat_lon(e, n)
                        data[key].append((id, lat, lon, v))
                        logger.debug(f"Converted coords for {id}: lat={lat}, lon={lon}, value={v}")
                    except Exception as e:
                        logger.warning(f"Failed to convert coords for {id}: easting={e}, northing={n}, error: {str(e)}")
                        continue
                logger.debug(f"Collected {len(data[key])} points for series {key}")

            if not any(data.values()):
                logger.warning("No data found for any series")
                return "No data found", []

            center_lat, center_lon = easting_northing_to_lat_lon(center_e, center_n)
            fig = go.Figure()
            for i, s in enumerate(inputs.series):
                key = f"{s.instrument_type}_{s.instrument_subtype}_{s.database_field_name}"
                points = data.get(key, [])
                if not points:
                    continue
                instr_ids, lats, lons, vals = zip(*points)
                if inputs.data_type == 'readings':
                    max_abs = max(abs(max(vals)), abs(min(vals))) if vals else 1
                    if inputs.plot_type == 'change_over_period':
                        cmap = pc.diverging.PiYG
                        norm = [(v / max_abs) for v in vals]
                        colors = [cmap[int(((nv + 1) / 2) * (len(cmap)-1))] for nv in norm]
                        colorscale = [[(nv + 1) / 2, cmap[int(((nv + 1) / 2) * (len(cmap)-1))]] for nv in np.linspace(-1, 1, len(cmap))]
                        colorbar_title = f"{s.measured_quantity_name} ({s.abbreviated_unit})"
                        tickvals = [min(vals), 0, max(vals)] if min(vals) < 0 < max(vals) else [min(vals), max(vals)]
                        ticktext = [f"{v:.2f}" for v in tickvals]
                    else:
                        cmap = pc.sequential.Reds if i==0 else pc.sequential.Blues if i==1 else pc.sequential.Greens
                        min_v, max_v = min(vals), max(vals)
                        norm = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.5 for v in vals]
                        colors = [cmap[int(n * (len(cmap)-1))] for n in norm]
                        colorscale = [[n, cmap[int(n * (len(cmap)-1))]] for n in np.linspace(0, 1, len(cmap))]
                        colorbar_title = f"{s.measured_quantity_name} ({s.abbreviated_unit})"
                        tickvals = [min(vals), max(vals)]
                        ticktext = [f"{min(vals):.2f}", f"{max(vals):.2f}"]
                    text = [f"{id}: {v:.2f} {s.abbreviated_unit}" for id, v in zip(instr_ids, vals)]
                    fig.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons, mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=10, 
                            color=vals, 
                            colorscale=colorscale, 
                            showscale=True,
                            colorbar=dict(
                                title=colorbar_title,
                                x=1 + 0.15 * i,  # Position colorbars side by side
                                xanchor="left",
                                len=0.3,
                                y=0.5 - 0.15 * i,  # Stagger vertically to avoid overlap
                                yanchor="middle",
                                tickvals=tickvals,
                                ticktext=ticktext
                            )
                        ),
                        text=text,
                        name=f"{s.measured_quantity_name} ({s.abbreviated_unit})"
                    ))
                else:
                    colors = vals
                    text = [f"{id}: {v}" for id, v in zip(instr_ids, vals)]
                    unique_colors = list(set(vals))
                    colorscale = [[0, 'grey'], [0.33, 'yellow'], [0.66, 'orange'], [1, 'red']] if len(unique_colors) > 1 else [[0, colors[0]], [1, colors[0]]]
                    colorbar_title = f"{s.measured_quantity_name} Status"
                    tickvals = unique_colors
                    ticktext = unique_colors
                    fig.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons, mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=10, 
                            color=vals, 
                            colorscale=colorscale, 
                            showscale=True,
                            colorbar=dict(
                                title=colorbar_title,
                                x=1 + 0.15 * i,
                                xanchor="left",
                                len=0.3,
                                y=0.5 - 0.15 * i,
                                yanchor="middle",
                                tickvals=tickvals,
                                ticktext=ticktext
                            )
                        ),
                        text=text,
                        name=f"{s.measured_quantity_name} ({s.abbreviated_unit})"
                    ))

            time_str = f"at {format_datetime(inputs.end_time)}" if inputs.plot_type == 'value_at_time' else f"change from {format_datetime(inputs.start_time)} to {format_datetime(inputs.end_time)}"
            fig.update_layout(
                title=f"{inputs.data_type.capitalize()} {time_str}",
                mapbox_style="open-street-map",
                mapbox_center={"lat": center_lat, "lon": center_lon},
                mapbox_zoom=15,
                showlegend=False  # Disable legend
            )

            all_data = []
            for s in inputs.series:
                key = f"{s.instrument_type}_{s.instrument_subtype}_{s.database_field_name}"
                for id, lat, lon, val in data.get(key, []):
                    all_data.append({
                        'Instrument ID': id,
                        'Latitude': lat,
                        'Longitude': lon,
                        'Value/Color': val,
                        'Series': s.measured_quantity_name,
                        'Unit': s.abbreviated_unit
                    })
            df = pd.DataFrame(all_data)
            csv_content = df.to_csv(index=False)

            html_filename = f"map_plot_{uuid.uuid4()}.html"
            html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
            script_dir = os.path.dirname(os.path.abspath(__file__))
            html_filepath = os.path.join(script_dir, html_filename)
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            artefacts = [
                {
                    'artifact_id': str(uuid.uuid4()),
                    'filename': f"map_plot_{uuid.uuid4()}.json",
                    'type': 'Plotly object',
                    'description': 'Map plot JSON',
                    'content': fig.to_json()
                },
                {
                    'artifact_id': str(uuid.uuid4()),
                    'filename': f"map_data_{uuid.uuid4()}.csv",
                    'type': 'CSV',
                    'description': 'Map data CSV',
                    'content': csv_content
                }
            ]

            logger.debug("Plotly figure and CSV generated successfully")
            content = f"Generated map plot." + (f" Suggested outliers to exclude: {', '.join(suggestions)}" if suggestions and inputs.data_type == 'readings' else "")
            logger.debug(f"Returning content: {content}, artifacts count: {len(artefacts)}")
            return content, artefacts
        except Exception as e:
            logger.error(f"Error in MapPlotTool._run: {str(e)}", exc_info=True)
            return f"Error generating map: {str(e)}", []

class MapPlotWrapperInput(BaseModel):
    """Input for wrapper: JSON string of MapPlotInput fields.
    
    Example JSON:
    {"data_type": "readings", "plot_type": "change_over_period", "start_time": "1 August 2025 12:00:00 AM", "end_time": "31 August 2025 11:59:59 PM", ...}
    """
    input_json: str = Field(
        ...,
        description="Valid JSON string matching MapPlotInput schema. Times as strings in 'D Month YYYY H:MM:SS AM/PM'."
    )

class MapPlotWrapperTool(BaseTool):
    name: str = "map_plot_wrapper"
    description: str = """
    Wrapper to parse JSON input for map_plot tool.
    Handles string to datetime parsing.
    Returns JSON {'content': str, 'artifacts': list}.
    Use when agent provides JSON for map plot.
    """
    args_schema: Type[MapPlotWrapperInput] = MapPlotWrapperInput
    plot_tool: MapPlotTool = Field(exclude=True)
    response_format: str = "content"

    def _run(self, input_json: str) -> str:
        logger.debug(f"MapPlotWrapperTool._run called")
        logger.debug(f"Input JSON: {input_json}")
        try:
            # Remove JSON code block markers if present
            input_json = re.sub(r'^```json\s*\n|\s*```$', '', input_json, flags=re.MULTILINE)
            # Replace single quotes with double quotes for valid JSON
            input_json = input_json.replace("'", '"')
            # Replace "None" with "null" for JSON compatibility
            input_json = re.sub(r':\s*None\b', ': null', input_json)
            input_dict = json.loads(input_json)
            logger.debug(f"Parsed input dictionary: {input_dict}")

            # Extract required fields (will raise KeyError if missing)
            data_type = input_dict['data_type']
            plot_type = input_dict['plot_type']
            end_time_str = input_dict['end_time']
            series_list = input_dict['series']
            radius_meters = input_dict['radius_meters']

            # Extract optional fields with defaults
            start_time_str = input_dict.get('start_time')
            buffer_period_hours = input_dict.get('buffer_period_hours', 72)
            center_instrument_id = input_dict.get('center_instrument_id')
            center_easting = input_dict.get('center_easting')
            center_northing = input_dict.get('center_northing')
            exclude_instrument_ids = input_dict.get('exclude_instrument_ids', [])

            # Parse datetimes
            start_time = parse_datetime(start_time_str) if start_time_str else None
            end_time = parse_datetime(end_time_str)

            # Convert series to Pydantic models
            series = [SeriesDict(**s) for s in series_list]

            # Call the underlying tool with explicit arguments
            content, artefacts = self.plot_tool._run(
                data_type=data_type,
                plot_type=plot_type,
                start_time=start_time,
                end_time=end_time,
                buffer_period_hours=buffer_period_hours,
                series=series,
                center_instrument_id=center_instrument_id,
                center_easting=center_easting,
                center_northing=center_northing,
                radius_meters=radius_meters,
                exclude_instrument_ids=exclude_instrument_ids
            )
            logger.debug(f"Plot tool result: {content}, {artefacts}")

            return json.dumps({'content': content, 'artifacts': artefacts})
        except KeyError as e:
            return json.dumps({'content': f"Error processing input: Missing required field {str(e)}", 'artifacts': []})
        except ValueError as e:
            return json.dumps({'content': f"Error processing input: {str(e)}", 'artifacts': []})
        except Exception as e:
            return json.dumps({'content': f"Error: {str(e)}", 'artifacts': []})