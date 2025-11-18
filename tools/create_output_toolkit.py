import json
import logging
import re
import uuid
from datetime import datetime, timedelta
import datetime as datetime_module
import decimal
import math
from typing import Dict, List, Optional, Tuple, Type, Union, Any

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, validator
from pyproj.crs import CRS
from pyproj.transformer import Transformer

from tools.sql_security_toolkit import GeneralSQLQueryTool
from tools.review_level_toolkit import (
    GetMapSeriesReviewSnapshotTool,
    GetMapSeriesReviewChangeTool,
    GetReviewSchemaTool,
)

logger = logging.getLogger(__name__)

def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string in format 'D Month YYYY H:MM:SS AM/PM'."""
    # Try Windows format (no leading zeros)
    try:
        return datetime.strptime(dt_str, "%#d %B %Y %#I:%M:%S %p")
    except ValueError:
        pass
        
    # Try padded format
    try:
        return datetime.strptime(dt_str, "%d %B %Y %I:%M:%S %p")
    except ValueError:
        pass
            
    raise ValueError(
        f"Invalid datetime format: {dt_str}. "
        "Expected format: 'D Month YYYY H:MM:SS AM/PM' "
        "e.g. '2 August 2025 2:30:00 PM' or '02 August 2025 02:30:00 PM'"
    )

def format_datetime(dt: datetime) -> str:
    """Format datetime object to string in format 'D Month YYYY H:MM:SS AM/PM'."""
    return dt.strftime("%d %B %Y %I:%M:%S %p").lstrip("0").replace(" 0", " ")

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
    secondary_y_instruments: Optional[List[InstrumentColumnPair]] = Field(
        default_factory=list,
        description="An optional list of instrument-column pairs to plot on the secondary (right) y-axis. Each pair is an object with 'instrument_id' (a string, e.g., 'INST003') and 'column_name' (a string, e.g., 'data2' or 'calculation2'). The total number of pairs (primary + secondary) cannot exceed 7. Example: [{'instrument_id': 'INST003', 'column_name': 'data2'}]."
    )
    start_time: Union[str, datetime] = Field(
        ...,
        description="The start of the time range for the data to plot, as a datetime string in the format 'D Month YYYY H:MM:SS AM/PM' (e.g., '1 January 2025 12:00:00 PM'). Must be earlier than end_time and match the format of the 'date1' column in the database."
    )
    end_time: Union[str, datetime] = Field(
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
    highlight_zero: Optional[bool] = Field(
        False,
        description="Whether to highlight the zero line on the primary y-axis with a light grey line. Only applicable if secondary_y_instruments is empty. Set to true to enable, false to disable. Example: true."
    )

    @validator('start_time', 'end_time', pre=True)
    def _parse_datetime_fields(cls, v):
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return parse_datetime(v)
        return v

    @validator('secondary_y_instruments', pre=True)
    def _coerce_none_to_list(cls, v):
        return [] if v is None else v

    @validator('highlight_zero', pre=True)
    def _coerce_none_to_bool(cls, v):
        if v is None:
            return False
        if isinstance(v, str):
            val = v.strip().lower()
            if val in ('true', '1', 'yes'):
                return True
            if val in ('false', '0', 'no'):
                return False
        return v

class BaseSQLQueryTool(BaseModel):
    """Base tool for SQL database interaction."""
    sql_tool: GeneralSQLQueryTool = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TimeSeriesPlotTool(BaseTool, BaseSQLQueryTool):
    """Tool for plotting time series data with Plotly."""
    name: str = "time_series_plot_tool"
    description: str = """
Creates an interactive Plotly time series plot and CSV file from instrumentation data. Supports multiple time series with different column names, dual y-axes and customizable gridlines. Review levels are superposed. Returns a tuple of (content, artefacts) with response_format='content_and_artifact'. The artefacts include the Plotly JSON and CSV file.

Input schema:
- primary_y_instruments: List of objects with 'instrument_id' (str, e.g., 'INST001') and 'column_name' (str, e.g., 'data1' or 'calculation1'). At least one required. Total instruments (primary + secondary) <= 7.
- secondary_y_instruments: Optional list of objects with 'instrument_id' and 'column_name'.
- start_time: Datetime string in 'D Month YYYY H:MM:SS AM/PM' (e.g., '1 January 2025 12:00:00 PM'). Required.
- end_time: Datetime string in same format, later than start_time. Required.
- primary_y_title: String for primary y-axis title (e.g., 'Temperature'). Required.
- primary_y_unit: String for primary y-axis unit (e.g., '°C'). Required.
- secondary_y_title: Optional string for secondary y-axis title, required if secondary_y_instruments non-empty.
- secondary_y_unit: Optional string for secondary y-axis unit, required if secondary_y_instruments non-empty.
- highlight_zero: Boolean to highlight zero line on primary y-axis. Only if secondary_y_instruments empty.

Example: {"primary_y_instruments": [{"instrument_id": "INST001", "column_name": "data1"}], "start_time": "1 January 2025 12:00:00 AM", "end_time": "31 January 2025 11:59:59 PM", "primary_y_title": "Temperature", "primary_y_unit": "°C", "highlight_zero": true}
"""
    args_schema: Type[TimeSeriesPlotInput] = TimeSeriesPlotInput
    response_format: str = "content"

    def _get_x_grid_settings(self, start_time: datetime, end_time: datetime) -> Dict:
        """Determine x-axis grid settings and formatting."""
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
                start_unit = settings.get('start_unit')
                major_spacing = settings['major_spacing']
                minor_spacing = settings['minor_spacing']

                tickmode = 'linear'
                tick0_dt = start_time

                if start_unit == 'year':
                    tick0_dt = datetime(start_time.year, 1, 1)
                    months = max(1, int(round(major_spacing / month_seconds)))
                    settings['major_dtick'] = f"M{months}"
                    minor_months = int(round(minor_spacing / month_seconds))
                    if minor_months >= 1:
                        settings['minor_dtick'] = f"M{minor_months}"
                    else:
                        settings['minor_dtick'] = int(minor_spacing * 1000)
                elif start_unit == 'month':
                    tick0_dt = datetime(start_time.year, start_time.month, 1)
                    months = max(1, int(round(major_spacing / month_seconds)))
                    settings['major_dtick'] = f"M{months}"
                    minor_months = int(round(minor_spacing / month_seconds))
                    if minor_months >= 1:
                        settings['minor_dtick'] = f"M{minor_months}"
                    else:
                        settings['minor_dtick'] = int(minor_spacing * 1000)
                elif start_unit in ('day', 'midnight'):
                    tick0_dt = datetime(start_time.year, start_time.month, start_time.day)
                    settings['major_dtick'] = int(major_spacing * 1000)
                    settings['minor_dtick'] = int(minor_spacing * 1000)
                elif start_unit == 'hour':
                    tick0_dt = datetime(start_time.year, start_time.month, start_time.day, start_time.hour)
                    settings['major_dtick'] = int(major_spacing * 1000)
                    settings['minor_dtick'] = int(minor_spacing * 1000)
                elif start_unit == 'minute':
                    tick0_dt = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute)
                    settings['major_dtick'] = int(major_spacing * 1000)
                    settings['minor_dtick'] = int(minor_spacing * 1000)
                else:
                    settings['major_dtick'] = int(major_spacing * 1000)
                    settings['minor_dtick'] = int(minor_spacing * 1000)

                settings['tickmode'] = tickmode
                settings['tick0'] = tick0_dt.strftime("%Y-%m-%d %H:%M:%S")
                return settings
        return grid_settings[-1][1]

    def _get_y_grid_settings(self, y_min: float, y_max: float) -> Dict:
        """Determine y-axis grid settings."""
        y_range = y_max - y_min
        logger.info(f"Calculating y-grid settings for range: {y_min} to {y_max} (range={y_range})")
        if y_range == 0:
            return {'major_step': 1, 'minor_step': 0.5}
        
        steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        major_step = 1
        for step in steps:
            n_grids = y_range / step
            if 4 <= n_grids <= 7:
                major_step = step
                break
        logger.info(f"Determined major_step: {major_step}")
        
        if major_step in [0.2, 2, 20, 200]:
            minor_step = major_step / 2
        elif major_step in [0.5, 5, 50, 500]:
            minor_step = major_step / 5
        else:
            minor_step = major_step / 2
        logger.info(f"Determined minor_step: {minor_step}")
        
        return {'major_step': major_step, 'minor_step': minor_step}

    def _run(
        self,
        primary_y_instruments: List[InstrumentColumnPair],
        start_time: datetime,
        end_time: datetime,
        primary_y_title: str,
        primary_y_unit: str,
        secondary_y_instruments: Optional[List[InstrumentColumnPair]] = None,
        secondary_y_title: Optional[str] = None,
        secondary_y_unit: Optional[str] = None,
        highlight_zero: bool = False
    ) -> Tuple[str, List[Dict]]:
        secondary_y_instruments = secondary_y_instruments or []
        logger.info(
            f"TimeSeriesPlotTool._run called with {len(primary_y_instruments)} primary and {len(secondary_y_instruments)} secondary instruments"
        )
        try:
            if start_time >= end_time:
                raise ValueError("start_time must be earlier than end_time. Correct the time range.")
            if secondary_y_instruments and (secondary_y_title is None or secondary_y_unit is None):
                raise ValueError("secondary_y_title and secondary_y_unit are required when secondary_y_instruments is provided.")
            total_instr = len(primary_y_instruments) + len(secondary_y_instruments)
            if total_instr == 0:
                raise ValueError("At least one instrument-column pair required in primary_y_instruments.")
            if total_instr > 7:
                raise ValueError("Total instrument-column pairs (primary + secondary) cannot exceed 7.")
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
                
                logger.info(f"Executing SQL query for {column_name}:\n{query}")
                results = self.sql_tool._run(query)
                logger.info(f"Query results: {results}")
                
                if results == "No data was found in the database matching the specified search criteria.":
                    continue
                
                try:
                    parsed_data: List[dict] = eval(
                        results,
                        {"__builtins__": {}},
                        {
                            "datetime": datetime_module,
                            "date": datetime_module.date,
                            "time": datetime_module.time,
                            "Decimal": decimal.Decimal,
                            "bytes": bytes,
                            "bytearray": bytearray
                        }
                    )
                    for row in parsed_data:
                        dt = row['timestamp']
                        val_str = row['value']
                        instr_id = row['instr_id']
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
                raise ValueError("No valid data found for any instruments in the specified time range. Check instrument IDs, column names, or expand the time range.")
            
            logger.info(f"Processed data for instruments: {list(time_series_data.keys())}")
            logger.info(f"Parsed data into time series data: {time_series_data}")

            # Create Plotly figure
            fig = go.Figure()
            logger.info("Plotly figure initialized")
            
            # Colors for time series
            primary_colors = ['#1f77b4', '#4b9cd3', '#87ceeb', '#add8e6']  # Blue hues
            secondary_colors = ['#ff69b4', '#ff85c0', '#ffb6c1', '#ffc1cc']  # Pink hues
            
            # Plot time series for primary y-axis
            y_min, y_max = float('inf'), float('-inf')
            for i, instr in enumerate(primary_y_instruments):
                instr_id = instr.instrument_id
                if instr_id not in time_series_data or not time_series_data[instr_id]:
                    logger.info(f"No data for primary instrument {instr_id}")
                    continue
                times, values = zip(*time_series_data[instr_id])
                logger.info(f"Primary instrument {instr_id}: {len(times)} data points")
                y_min = min(y_min, min(values))
                y_max = max(y_max, max(values))
                if y_min == y_max:
                    y_min -= 1
                    y_max += 1
                if secondary_y_instruments:
                    axis_suffix = f" {primary_y_title}".lower()
                    display_name = f"{instr_id}{axis_suffix}"
                else:
                    display_name = f"{instr_id}"
                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=primary_colors[i % len(primary_colors)]),
                    hovertemplate=f"%{{x|%-d %b %Y %-I:%M%p}}<br>%{{y:.5f}} {primary_y_unit}<extra></extra>"
                ))
                logger.info(f"Added primary trace for {instr_id}")
            logger.info(f"Primary traces added | y_min={y_min}, y_max={y_max}")

            # Plot time series for secondary y-axis
            secondary_y_min, secondary_y_max = float('inf'), float('-inf')
            for i, instr in enumerate(secondary_y_instruments):
                instr_id = instr.instrument_id
                if instr_id not in time_series_data or not time_series_data[instr_id]:
                    logger.info(f"No data for secondary instrument {instr_id}")
                    continue
                times, values = zip(*time_series_data[instr_id])
                logger.info(f"Secondary instrument {instr_id}: {len(times)} data points")
                secondary_y_min = min(secondary_y_min, min(values))
                secondary_y_max = max(secondary_y_max, max(values))
                sec_suffix_title = (secondary_y_title or "").lower()
                display_name = f"{instr_id} {sec_suffix_title}" if sec_suffix_title else f"{instr_id}"
                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=secondary_colors[i % len(secondary_colors)]),
                    yaxis='y2',
                    hovertemplate=f"%{{x|%-d %b %Y %-I:%M%p}}<br>%{{y:.5f}} {secondary_y_unit if secondary_y_unit is not None else ''}<extra></extra>"
                ))
                logger.info(f"Added secondary trace for {instr_id}")
            logger.info(f"Secondary traces added | y_min={secondary_y_min}, y_max={secondary_y_max}")

            # Highlight zero line
            if highlight_zero and not secondary_y_instruments:
                fig.add_hline(
                    y=0,
                    line_color='#607164',
                    line_width=1
                )
                y_min = min(y_min, 0)
                y_max = max(y_max, 0)
                logger.info("Added zero highlight hline")

            # Set axis properties
            x_grid = self._get_x_grid_settings(start_time, end_time)
            logger.info(f"x_grid calculated: {x_grid}")
            primary_y_grid = self._get_y_grid_settings(y_min, y_max)
            logger.info(f"primary_y_grid calculated: {primary_y_grid}")

            if primary_y_grid['major_step'] <= 0 or primary_y_grid['minor_step'] <= 0:
                logger.error(f"Invalid primary y-grid steps: major={primary_y_grid['major_step']}, minor={primary_y_grid['minor_step']}")
                raise ValueError("Invalid y-axis grid steps calculated")
            
            # Add review level horizontal lines
            try:
                review_lines_added = False
                if not secondary_y_instruments and primary_y_instruments:
                    major_step = primary_y_grid.get('major_step', 1)
                    y_grid_min = math.floor(y_min / major_step) * major_step
                    y_grid_max = math.ceil(y_max / major_step) * major_step
                    if len(primary_y_instruments) == 1:
                        target_column = primary_y_instruments[0].column_name
                        schema_tool = GetReviewSchemaTool(sql_tool=self.sql_tool)
                        schema_df = schema_tool._run(primary_y_instruments[0].instrument_id, target_column)
                        if isinstance(schema_df, str) and schema_df.startswith("ERROR"):
                            logger.warning(f"GetReviewSchemaTool returned error: {schema_df}")
                        elif schema_df is not None and hasattr(schema_df, 'empty') and not schema_df.empty:
                            for _, row in schema_df.iterrows():
                                try:
                                    yval = float(row.get('review_value'))
                                except Exception:
                                    continue
                                color = str(row.get('review_color') or '#FF0000')
                                name = str(row.get('review_name') or '')
                                fig.add_hline(
                                    y=yval,
                                    line_color=color,
                                    line_width=2,
                                    layer='below',
                                    annotation_text=name,
                                    annotation_position='top left',
                                    annotation=dict(font_color='#607164')
                                )
                                review_lines_added = True
                            if review_lines_added:
                                logger.info("Added review level horizontal lines (single-series fast path)")
                    else:
                        primary_columns = {p.column_name for p in primary_y_instruments}
                        if len(primary_columns) == 1:
                            target_column = next(iter(primary_columns))
                            instr_ids = [p.instrument_id for p in primary_y_instruments]
                            instr_ids_str = ",".join(f"'{i}'" for i in instr_ids)
                            meta_query = f"""
                            SELECT instr_id, type1, subtype1
                            FROM instrum
                            WHERE instr_id IN ({instr_ids_str});
                            """
                            logger.info(f"Executing instrument metadata query: {meta_query}")
                            meta_result = self.sql_tool._run(meta_query)
                            logger.info(f"Instrument metadata query result: {meta_result}")
                            def _safe_parse(raw: Any) -> List[Dict[str, Any]]:
                                if raw is None:
                                    return []
                                if isinstance(raw, list):
                                    return [dict(r) if isinstance(r, dict) else r for r in raw]
                                s = str(raw)
                                try:
                                    parsed = json.loads(s)
                                    if isinstance(parsed, list):
                                        return [dict(r) if isinstance(r, dict) else r for r in parsed]
                                except Exception:
                                    pass
                                try:
                                    parsed = eval(s, {"__builtins__": {}})
                                    if isinstance(parsed, list):
                                        return [dict(r) if isinstance(r, dict) else r for r in parsed]
                                except Exception:
                                    return []
                                return []
                            meta_rows = _safe_parse(meta_result)
                            id_to_meta: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
                            for r in meta_rows:
                                if isinstance(r, dict):
                                    rid = r.get('instr_id') or r.get('i.instr_id') or r.get('INSTR_ID')
                                    t = r.get('type1') or r.get('i.type1') or r.get('TYPE1')
                                    st = r.get('subtype1') or r.get('i.subtype1') or r.get('SUBTYPE1')
                                elif isinstance(r, (list, tuple)) and len(r) >= 3:
                                    rid, t, st = r[0], r[1], r[2]
                                else:
                                    continue
                                if rid is not None:
                                    id_to_meta[str(rid)] = (None if t is None else str(t), None if st is None else str(st))
                            types = set(); subtypes = set()
                            for pid in instr_ids:
                                t, st = id_to_meta.get(pid, (None, None))
                                if t is None or st is None:
                                    types.add(None); subtypes.add(None)
                                else:
                                    types.add(t); subtypes.add(st)
                            if len(types) == 1 and None not in types and len(subtypes) == 1 and None not in subtypes:
                                schema_tool = GetReviewSchemaTool(sql_tool=self.sql_tool)
                                schema_df = schema_tool._run(primary_y_instruments[0].instrument_id, target_column)
                                if isinstance(schema_df, str) and schema_df.startswith("ERROR"):
                                    logger.warning(f"GetReviewSchemaTool returned error: {schema_df}")
                                elif schema_df is not None and hasattr(schema_df, 'empty') and not schema_df.empty:
                                    for _, row in schema_df.iterrows():
                                        try:
                                            yval = float(row.get('review_value'))
                                        except Exception:
                                            continue
                                        color = str(row.get('review_color') or '#FF0000')
                                        name = str(row.get('review_name') or '')
                                        fig.add_hline(
                                            y=yval,
                                            line_color=color,
                                            line_width=2,
                                            layer='below',
                                            annotation_text=name,
                                            annotation_position='top left',
                                            annotation=dict(font_color='#607164')
                                        )
                                        review_lines_added = True
                                    if review_lines_added:
                                        logger.info("Added review level horizontal lines beneath data traces")
                            else:
                                logger.info("Skipping review lines: primary series do not share the same type/subtype or metadata missing")
                        else:
                            logger.info("Skipping review lines: primary series have differing database field names")
                else:
                    logger.info("Skipping review lines: secondary axis present or no primary series")
            except Exception as e:
                logger.warning(f"Review lines section failed: {e}")
            
            major_step = primary_y_grid['major_step']
            minor_step = primary_y_grid['minor_step']
            y_grid_min = math.floor(y_min / major_step) * major_step
            y_grid_max = math.ceil(y_max / major_step) * major_step

            yaxis_config = {
                'title': {'text': f"{primary_y_title} ({primary_y_unit})", 'font': {'color': '#607164'}},
                'dtick': major_step,
                'minor': {'dtick': minor_step, 'showgrid': True, 'gridcolor': '#D8DEDA'},
                'showgrid': True,
                'gridcolor': '#B2BEB5',
                'tickfont': {'color': '#607164'}
            }
            if y_min <= 0 <= y_max:
                yaxis_config.update({
                    'tickmode': 'linear',
                    'tick0': 0,
                    'range': [y_grid_min, y_grid_max]
                })

            layout = {
                'xaxis': {
                    'tickformat': x_grid['major_format'],
                    'dtick': x_grid['major_dtick'],
                    'tickmode': x_grid.get('tickmode', 'linear'),
                    'tick0': x_grid.get('tick0'),
                    'minor': {'dtick': x_grid['minor_dtick'], 'showgrid': True, 'gridcolor': '#D8DEDA'},
                    'showgrid': True,
                    'gridcolor': '#B2BEB5',
                    'tickfont': {'color': '#607164'}
                },
                'yaxis': yaxis_config
            }
            
            if secondary_y_instruments:
                secondary_y_grid = self._get_y_grid_settings(secondary_y_min, secondary_y_max)
                logger.info(f"secondary_y_grid calculated: {secondary_y_grid}")
                if secondary_y_grid['major_step'] <= 0 or secondary_y_grid['minor_step'] <= 0:
                    logger.error(f"Invalid secondary y-grid steps: major={secondary_y_grid['major_step']}, minor={secondary_y_grid['minor_step']}")
                    raise ValueError("Invalid secondary y-axis grid steps calculated")
                layout['yaxis2'] = {
                    'title': {'text': f"{secondary_y_title} ({secondary_y_unit})", 'font': {'color': '#607164'}},
                    'overlaying': 'y',
                    'side': 'right',
                    'dtick': secondary_y_grid['major_step'],
                    'minor': {
                        'dtick': secondary_y_grid['minor_step'],
                        'showgrid': True,
                        'gridcolor': '#D8DEDA'
                    },
                    'showgrid': True,
                    'gridcolor': '#B2BEB5',
                    'tickfont': {'color': '#607164'}
                }
            
            fig.update_layout(
                showlegend=True,
                template="plotly_white",
                font={'color': '#607164'},
                legend={'font': {'color': '#607164'}},
                **layout
            )
            logger.info("Figure layout updated")

            if not fig.data:
                logger.error("Plotly figure has no traces")
                raise ValueError("No traces added to Plotly figure; check instrument data")
            
            # Create CSV
            all_data = []
            for instr in primary_y_instruments + secondary_y_instruments:
                instr_id = instr.instrument_id
                column_name = instr.column_name
                if instr_id not in time_series_data or not time_series_data[instr_id]:
                    logger.info(f"Skipping empty data for instrument {instr_id}")
                    continue
                for dt, val in time_series_data[instr_id]:
                    all_data.append({
                        'Instrument ID': instr_id,
                        'Column Name': column_name,
                        'Timestamp': format_datetime(dt),
                        'Value': val,
                        'Y-Axis': 'Primary' if instr in primary_y_instruments else 'Secondary'
                    })
            logger.info(f"all_data populated with {len(all_data)} entries")

            if not all_data:
                logger.error("No data for CSV generation")
                raise ValueError("No data available for CSV generation")
            
            df = pd.DataFrame(all_data)
            logger.info(f"DataFrame created with shape {df.shape}")
            csv_content = df.to_csv(index=False)
            logger.info(f"CSV content generated (length: {len(csv_content)})")
            
            # Prepare artefacts
            plotly_filename = f"time_series_{uuid.uuid4()}.json"
            try:
                plotly_json = fig.to_json()
                logger.info(f"Plotly JSON generated (length: {len(plotly_json)})")
            except Exception as e:
                logger.error(f"Failed to serialize Plotly figure: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to serialize Plotly figure: {str(e)}")

            artefacts = [
                {
                    'artifact_id': str(uuid.uuid4()),
                    'filename': plotly_filename,
                    'type': 'Plotly object',
                    'description': 'Time series plot JSON',
                    'content': plotly_json
                },
                {
                    'artifact_id': str(uuid.uuid4()),
                    'filename': f"time_series_{uuid.uuid4()}.csv",
                    'type': 'CSV',
                    'description': 'Time series data CSV',
                    'content': csv_content
                }
            ]
            logger.info(f"Artefacts list created with {len(artefacts)} items")
            
            content = f"Generated time series plot and CSV for instruments {', '.join(all_instrument_ids)} from {format_datetime(start_time)} to {format_datetime(end_time)}."
            logger.info(f"Returning content: {content}, artefacts count: {len(artefacts)}")
            return {"content": content, "artefacts": artefacts}
        
        except ValueError as e:
            return {"content": f"Error: Invalid input - {str(e)}. Correct and retry.", "artefacts": []}
        except Exception as e:
            return {"content": f"Error: Unexpected failure - {str(e)}. Verify database connection, instrument existence, or numeric data in columns.", "artefacts": []}

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
        logger.info(f"TimeSeriesPlotWrapperTool._run called")
        logger.info(f"Input JSON: {input_json}")
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
            logger.info(f"Parsed input dictionary: {input_dict}")
            
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
                highlight_zero=input_dict.get('highlight_zero', False)
            )
            logger.info(f"Plot tool result: {plot_tool_result}")

            content, artifacts = plot_tool_result
            result = {'content': content, 'artifacts': artifacts}
            return json.dumps(result)
        except Exception as e:
            content = f"Error processing input: {str(e)}"
            result = {'content': content, 'artifacts': []}
            return json.dumps(result)

hn72_crs = CRS.from_epsg(3405)  # VN-2000 / Vietnam TM-3 zone 105-45
wgs84_crs = CRS.from_epsg(4326)  # WGS84
transformer = Transformer.from_crs(hn72_crs, wgs84_crs, always_xy=True)

def easting_northing_to_lat_lon(easting: float, northing: float) -> Tuple[float, float]:
    """Convert easting and northing from VN-2000 to latitude and longitude using WGS84."""
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

def get_projection_from_db(sql_tool):
    """
    Fetch projection definition from the crs_configuration table.
    Returns a projection string (PROJ4 or WKT format) for coordinate transformation.
    
    Parameters:
    sql_tool: GeneralSQLQueryTool instance for database access
    
    Returns:
    str: Projection definition string from the database
    
    Raises:
    Exception: If no projection is found or database query fails
    """
    try:
        query = """
            SELECT projection_definition, projection_name, projection_type
            FROM crs_configuration
            WHERE is_deleted = 0
            AND projection_definition IS NOT NULL
            AND projection_definition != ''
            LIMIT 1;
        """
        result = sql_tool._run(query)
        logger.debug(f"Raw SQL result for CRS configuration: {result}")

        if result == "No data was found in the database matching the specified search criteria.":
            logger.warning("No CRS configuration found in database")
            raise ValueError("No CRS configuration found in crs_configuration table. Please add a projection definition.")

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                logger.debug("Parsed CRS configuration result using json.loads")
            except Exception:
                try:
                    parsed = eval(result, {"__builtins__": {}})
                    logger.debug("Parsed CRS configuration result using eval with restricted environment")
                except Exception as e:
                    logger.error(f"Failed to parse CRS configuration result: {e}")
                    raise ValueError(f"Unparseable result from crs_configuration query: {e}")
        else:
            parsed = result

        if parsed is None:
            raise ValueError("Empty result from crs_configuration table")

        if isinstance(parsed, (list, tuple)):
            rows = list(parsed)
        elif isinstance(parsed, dict):
            rows = [parsed]
        else:
            raise ValueError(
                f"Unexpected result type for crs_configuration query; got {type(parsed).__name__}, expected list/tuple/dict"
            )

        if not rows:
            raise ValueError("Empty result from crs_configuration table")

        first = rows[0]
        if isinstance(first, dict):
            projection_definition = first.get('projection_definition')
            projection_name = first.get('projection_name')
            projection_type = first.get('projection_type')
        elif isinstance(first, (list, tuple)):
            if len(first) < 3:
                raise ValueError("Result row has fewer than 3 columns: expected (projection_definition, projection_name, projection_type)")
            projection_definition, projection_name, projection_type = first[0], first[1], first[2]
        else:
            raise ValueError(f"Unsupported row type from crs_configuration: {type(first).__name__}")

        if not projection_definition or projection_definition.strip() == "":
            raise ValueError("Projection definition is empty in crs_configuration table")

        logger.info(f"Successfully fetched projection: {projection_name} ({projection_type})")
        logger.debug(f"Projection definition: {projection_definition}")

        return projection_definition
        
    except Exception as e:
        logger.error(f"Error fetching projection from database: {e}")
        raise Exception(f"Failed to fetch projection definition from database: {e}")

def create_coordinate_converter(sql_tool):
    """
    Create a coordinate conversion function using projection string from database.
    Uses pyproj to transform local coordinates to WGS84 lat/lon.

    Parameters:
    sql_tool: GeneralSQLQueryTool instance for database access

    Returns:
    function: A function that takes a list of (id, easting, northing) tuples and
              returns a list of (id, lat, lon) tuples in WGS84, transformed in one pass.

    Usage:
    converter = create_coordinate_converter(sql_tool)
    out = converter([("INST001", 123456.7, 234567.8), ("INST002", 123400.0, 234500.0)])
    # out -> [("INST001", lat1, lon1), ("INST002", lat2, lon2)]
    """
    try:
        from pyproj import Transformer
    except ImportError:
        logger.error("pyproj library not found. Please install it: pip install pyproj")
        raise ImportError("pyproj library is required for coordinate conversion. Install with: pip install pyproj")
    
    proj_string = get_projection_from_db(sql_tool)
    logger.info(f"Using projection string from database: {proj_string}")
    
    try:
        transformer = Transformer.from_proj(
            proj_string,
            "+proj=longlat +datum=WGS84 +no_defs",
            always_xy=True
        )
        logger.info("Successfully created coordinate transformer using projection from database")
    except Exception as e:
        logger.error(f"Failed to create transformer with projection string: {e}")
        raise Exception(f"Invalid projection definition in database: {e}")

    def convert_coordinates(points: List[Tuple[Any, Any, Any]]) -> List[Tuple[str, float, float]]:
        """
        Batch-convert a list of (id, easting, northing) to (id, lat, lon) in WGS84.

        - Accepts any iterable of 3-tuples; id is preserved as string
        - Invalid rows (non-numeric easting/northing) are skipped with a warning
        - Performed in one transformer call for performance
        """
        if not points:
            return []

        ids: List[str] = []
        eastings: List[float] = []
        northings: List[float] = []

        for row in points:
            try:
                rid, e, n = row
            except Exception:
                logger.warning(f"Skipping row with unexpected shape: {row!r}")
                continue
            try:
                fe = float(e)
                fn = float(n)
            except Exception:
                logger.warning(f"Skipping row with non-numeric coordinates for id={rid!r}: easting={e!r}, northing={n!r}")
                continue
            ids.append(str(rid))
            eastings.append(fe)
            northings.append(fn)

        if not ids:
            return []

        try:
            lons, lats = transformer.transform(eastings, northings)
        except Exception as e:
            logger.error(f"Batch coordinate transformation failed: {e}")
            raise Exception(f"Failed to transform coordinates batch: {e}")

        converted: List[Tuple[str, float, float]] = []
        for rid, lat, lon in zip(ids, lats, lons):
            try:
                converted.append((rid, float(lat), float(lon)))
            except Exception:
                logger.warning(f"Skipping converted row due to non-float outputs for id={rid!r}: lat={lat!r}, lon={lon!r}")
                continue

        return converted

    return convert_coordinates

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
    start_time: Optional[Union[str, datetime]] = Field(
        None,
        description="Start datetime for 'change_over_period'. Required for changes, ignored for 'value_at_time'."
    )
    end_time: Union[str, datetime] = Field(
        ...,
        description="End datetime (or single time for 'value_at_time'). Must be after start_time if provided."
    )
    buffer_period_hours: Optional[int] = Field(
        72,
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
    radius_meters: Optional[float] = Field(
        500,
        description="Radius for spatial filter (square bounding box of side 2*radius). >0."
    )
    exclude_instrument_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="List of instrument IDs to exclude."
    )

    @validator('start_time', 'end_time', pre=True)
    def _parse_datetime_fields(cls, v):
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return parse_datetime(v)
        return v

    @validator('buffer_period_hours', pre=True)
    def _coerce_buffer(cls, v):
        if v is None:
            return 72
        return v

    @validator('radius_meters', pre=True)
    def _coerce_radius(cls, v):
        if v is None:
            return 500
        return v

    @validator('exclude_instrument_ids', pre=True)
    def _coerce_exclude(cls, v):
        return [] if v is None else v

    @validator('buffer_period_hours')
    def validate_buffer(cls, v, values):
        if v < 0:
            raise ValueError("buffer_period_hours must be non-negative")
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

    @validator('radius_meters')
    def validate_radius(cls, v):
        if v <= 0:
            raise ValueError("radius_meters must be positive")
        return v

 

class MapPlotTool(BaseTool, BaseSQLQueryTool):
    """Tool for plotting instrument readings or review status on a map with Plotly."""
    name: str = "map_plot_tool"
    description: str = """
Creates an interactive Plotly map plot and CSV file from instrumentation data. Supports readings (instrument values) or review_levels (breach statuses), plotted as value at a specific time or change over a period. Returns a dictionary with 'content' and 'artefacts'. The artefacts include the Plotly JSON and CSV file.

Input schema:
- data_type: String, either 'readings' or 'review_levels'.
- plot_type: String, either 'value_at_time' or 'change_over_period'.
- start_time: Optional datetime string in 'D Month YYYY H:MM:SS AM/PM' (e.g., '1 August 2025 12:00:00 AM'), required for 'change_over_period'.
- end_time: Datetime string in same format (e.g., '31 August 2025 11:59:59 PM'). Required.
- buffer_period_hours: Optional integer, buffer period in hours (default 72).
- series: List of objects with 'instrument_type' (str), 'instrument_subtype' (str), 'database_field_name' (str, e.g., 'data1'), 'measured_quantity_name' (str, e.g., 'Pressure'), 'abbreviated_unit' (str, e.g., 'kPa').
 - series: List of objects with 'instrument_type' (str), 'instrument_subtype' (str), 'database_field_name' (str, e.g., 'data1'), 'measured_quantity_name' (str, e.g., 'Pressure'), 'abbreviated_unit' (str, e.g., 'kPa'). NOTE: When data_type='review_levels' ONLY the FIRST series element is used to derive and plot review statuses; additional series entries are ignored.
- center_instrument_id: Optional string, instrument ID to center the map.
- center_easting: Optional float, easting coordinate to center the map if center_instrument_id not provided.
- center_northing: Optional float, northing coordinate to center the map if center_instrument_id not provided.
- radius_meters: Optional float, radius in meters for map extent (default 500).
- exclude_instrument_ids: Optional list of strings, instrument IDs to exclude.

Example: {"data_type": "readings", "plot_type": "value_at_time", "end_time": "31 August 2025 11:59:59 PM", "buffer_period_hours": 72, "series": [{"instrument_type": "sensor", "instrument_subtype": "pressure", "database_field_name": "data1", "measured_quantity_name": "Pressure", "abbreviated_unit": "kPa"}], "radius_meters": 1000.0}
"""
    args_schema: Type[MapPlotInput] = MapPlotInput
    response_format: str = "content"

    def _get_center_coords(self, center_instrument_id: Optional[str], center_easting: Optional[float], center_northing: Optional[float]) -> Tuple[float, float]:
        logger.info(f"Fetching center coordinates for instrument_id={center_instrument_id}, easting={center_easting}, northing={center_northing}")
        if center_instrument_id:
            query = f"""
            SELECT l.easting, l.northing
            FROM instrum i
            JOIN location l ON i.location_id = l.id
            WHERE i.instr_id = '{center_instrument_id}';
            """
            logger.info(f"Executing SQL query for center: {query}")
            result = self.sql_tool._run(query)
            logger.info(f"Query result: {result}")
            if result == "No data was found in the database matching the specified search criteria.":
                raise ValueError(f"No location found for instrument ID {center_instrument_id}")
            try:
                # Attempt JSON load first, fallback to eval
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except Exception:
                        parsed = eval(result, {"__builtins__": {}}, {
                            "datetime": datetime_module,
                            "date": datetime_module.date,
                            "time": datetime_module.time,
                            "Decimal": decimal.Decimal,
                            "bytes": bytes,
                            "bytearray": bytearray
                        })
                else:
                    parsed = result
                if not isinstance(parsed, list) or not parsed:
                    raise ValueError(f"No location found for instrument ID {center_instrument_id}")
                first = parsed[0]
                if isinstance(first, dict):
                    easting_val = first.get('easting') or first.get('l.easting') or first.get('EASTING')
                    northing_val = first.get('northing') or first.get('l.northing') or first.get('NORTHING')
                elif isinstance(first, (list, tuple)) and len(first) >= 2:
                    easting_val, northing_val = first[0], first[1]
                else:
                    raise ValueError("Unrecognized row structure for location query")
                easting_clean = clean_numeric_string(easting_val)
                northing_clean = clean_numeric_string(northing_val)
                if None in (easting_clean, northing_clean):
                    raise ValueError(f"Invalid easting or northing for instrument ID {center_instrument_id}")
                e_f, n_f = float(easting_clean), float(northing_clean)
                logger.info(f"Center coords from instrument: easting={e_f}, northing={n_f}")
                return e_f, n_f
            except Exception as e:
                logger.error(f"Error processing location for instrument ID {center_instrument_id}: {str(e)}")
                raise ValueError(f"Error processing location for instrument ID {center_instrument_id}: {str(e)}")
        elif center_easting is not None and center_northing is not None:
            logger.info(f"Using provided center coords: easting={center_easting}, northing={center_northing}")
            return center_easting, center_northing
        else:
            raise ValueError("Either center_instrument_id or both center_easting and center_northing must be provided")

    def _get_bounds(self, center_e: float, center_n: float, radius_m: float) -> Tuple[float, float, float, float]:
        logger.info(f"Calculating bounds with center_e={center_e}, center_n={center_n}, radius_m={radius_m}")
        min_e = center_e - radius_m
        max_e = center_e + radius_m
        min_n = center_n - radius_m
        max_n = center_n + radius_m
        logger.info(f"Bounds: min_e={min_e}, max_e={max_e}, min_n={min_n}, max_n={max_n}")
        return min_e, max_e, min_n, max_n

    

    def _extract_numeric(self, v: Any) -> Optional[str]:
        """Return a cleaned numeric string from various DB-returned value forms.

        Handles:
        - None / empty -> None
        - Strings wrapped in extra quotes produced by JSON_EXTRACT (e.g. '"-6.67"')
        - Numeric types -> str(n)
        - Already clean numeric strings
        """
        if v is None:
            return None
        if isinstance(v, (int, float, decimal.Decimal)):
            return str(v)
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode('utf-8')  # type: ignore
            except Exception:
                return None
        if isinstance(v, str):
            s = v.strip()
            # Remove enclosing quotes from JSON_EXTRACT on MySQL / SQLite behavior
            if len(s) >= 2 and ((s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'"))):
                s = s[1:-1]
            # Empty after stripping
            if s == '':
                return None
            # Basic numeric validation (allow leading - and decimal point)
            if any(c.isdigit() for c in s):
                return s
            return None
        return None

    def _extract_row_reading(self, row: Any) -> Optional[Tuple[str, Any, Any, Any]]:
        """Normalize a single row from SQL result for readings queries.

        Supports rows as tuples (instr_id, easting, northing, value) or dicts with keys.
        Returns (instr_id, easting, northing, value_raw) or None if unusable.
        """
        if isinstance(row, dict):
            instr_id = row.get('instr_id') or row.get('id')
            e = row.get('easting') or row.get('east')
            n = row.get('northing') or row.get('north')
            v = row.get('value') or row.get('val') or row.get('value_end') or row.get('value_start')
            if instr_id is None:
                return None
            return instr_id, e, n, v
        if isinstance(row, (list, tuple)) and len(row) == 4:
            return row[0], row[1], row[2], row[3]
        return None

    def _extract_row_change(self, row: Any, value_key: str) -> Optional[Tuple[str, Any]]:
        """Normalize change_over_period rows expecting (instr_id, value_key)."""
        if isinstance(row, dict):
            instr_id = row.get('instr_id') or row.get('id')
            v = row.get(value_key)
            if instr_id is None:
                return None
            return instr_id, v
        if isinstance(row, (list, tuple)) and len(row) == 2:
            return row[0], row[1]
        return None

    def _render_review_levels(
        self,
        inputs: MapPlotInput,
        center_e: float,
        center_n: float,
        min_e: float,
        max_e: float,
        min_n: float,
        max_n: float,
        converter,
    ) -> Dict[str, Any]:
        s0 = inputs.series[0]
        if inputs.plot_type == 'value_at_time':
            snap_tool = GetMapSeriesReviewSnapshotTool(sql_tool=self.sql_tool)
            df = snap_tool._run(
                instrument_type=s0.instrument_type,
                instrument_subtype=s0.instrument_subtype,
                db_field_name=s0.database_field_name,
                timestamp=inputs.end_time.isoformat() if isinstance(inputs.end_time, datetime) else str(inputs.end_time),
                min_e=min_e,
                max_e=max_e,
                min_n=min_n,
                max_n=max_n,
                exclude_instrument_ids=inputs.exclude_instrument_ids or None,
            )
            if isinstance(df, str) and df.startswith("ERROR"):
                return {"content": f"Error: {df}", "artefacts": []}
        else:
            ch_tool = GetMapSeriesReviewChangeTool(sql_tool=self.sql_tool)
            df = ch_tool._run(
                instrument_type=s0.instrument_type,
                instrument_subtype=s0.instrument_subtype,
                db_field_name=s0.database_field_name,
                start_timestamp=inputs.start_time.isoformat() if isinstance(inputs.start_time, datetime) else str(inputs.start_time),
                end_timestamp=inputs.end_time.isoformat() if isinstance(inputs.end_time, datetime) else str(inputs.end_time),
                min_e=min_e,
                max_e=max_e,
                min_n=min_n,
                max_n=max_n,
                exclude_instrument_ids=inputs.exclude_instrument_ids or None,
            )
            if isinstance(df, str) and df.startswith("ERROR"):
                return {"content": f"Error: {df}", "artefacts": []}

        if df is None or (hasattr(df, 'empty') and df.empty):
            return {"content": "No data found for any series in the specified criteria. Check parameters or database.", "artefacts": []}

        chosen_instr_id: Optional[str] = None
        try:
            if inputs.center_instrument_id:
                chosen_instr_id = inputs.center_instrument_id
            else:
                if 'review_status' in df.columns and df['review_status'].notna().any():
                    chosen_instr_id = str(df.loc[df['review_status'].notna(), 'instrument_id'].iloc[0])
                elif 'end_review_status' in df.columns and df['end_review_status'].notna().any():
                    chosen_instr_id = str(df.loc[df['end_review_status'].notna(), 'instrument_id'].iloc[0])
                else:
                    chosen_instr_id = str(df['instrument_id'].iloc[0])
        except Exception:
            chosen_instr_id = inputs.center_instrument_id or None

        schema_map: Dict[str, Dict[str, Any]] = {}
        schema_order: List[str] = []
        if chosen_instr_id:
            schema_tool = GetReviewSchemaTool(sql_tool=self.sql_tool)
            schema_df = schema_tool._run(chosen_instr_id, s0.database_field_name)
            if isinstance(schema_df, str) and schema_df.startswith("ERROR"):
                schema_df = None
            if schema_df is not None and not schema_df.empty:
                values = [float(v) for v in schema_df['review_value'].tolist() if pd.notna(v)]
                if values:
                    dec_places = 0
                    for v in values:
                        s = f"{v:.6f}".rstrip('0').rstrip('.')
                        if '.' in s:
                            dec_places = max(dec_places, min(3, len(s.split('.')[-1])))
                    fmt = f"{{:.{dec_places}f}}"
                else:
                    fmt = "{:.0f}"
                for _, r in schema_df.iterrows():
                    name = str(r.get('review_name'))
                    direction = str(r.get('review_direction') or '').lower()
                    color = str(r.get('review_color') or '#FF0000')
                    val = float(r.get('review_value') or 0.0)
                    sign = '>' if direction == 'upper' else '<'
                    legend = f"{name} {sign} {fmt.format(val)}"
                    key = f"{name}::{direction}" if direction else name
                    schema_map[key] = {
                        'name': name,
                        'color': color,
                        'legend': legend,
                        'direction': direction,
                        'value': val,
                    }
                    if key not in schema_order:
                        schema_order.append(key)

        if not schema_map:
            fallback_colors = ['#f9c74f', '#f9844a', '#f94144', '#577590', '#43aa8b', '#90be6d']
            status_col = 'review_status' if inputs.plot_type == 'value_at_time' else 'end_review_status'
            value_col = 'db_field_value' if inputs.plot_type == 'value_at_time' else 'end_db_field_value'
            thresh_col = 'review_value' if inputs.plot_type == 'value_at_time' else 'end_review_value'
            fmt = "{:.0f}"
            try:
                vals_for_fmt = [float(v) for v in df.get(thresh_col, pd.Series(dtype=float)).dropna().tolist()]
                if vals_for_fmt:
                    dec_places = 0
                    for v in vals_for_fmt:
                        s = f"{v:.6f}".rstrip('0').rstrip('.')
                        if '.' in s:
                            dec_places = max(dec_places, min(3, len(s.split('.')[-1])))
                    fmt = f"{{:.{dec_places}f}}"
            except Exception:
                pass
            seen: set = set()
            for _, r in df.iterrows():
                name = r.get(status_col)
                if pd.isna(name):
                    continue
                try:
                    fv = float(r.get(value_col)) if r.get(value_col) is not None else None
                    tv = float(r.get(thresh_col)) if r.get(thresh_col) is not None else None
                except Exception:
                    fv, tv = None, None
                # Infer direction when possible
                direction = None
                if fv is not None and tv is not None:
                    if fv > tv:
                        direction = 'upper'
                    elif fv < tv:
                        direction = 'lower'
                name_s = str(name)
                key = f"{name_s}::{direction}" if direction else name_s
                if key in seen:
                    continue
                seen.add(key)
                idx = len(schema_map)
                color = fallback_colors[idx % len(fallback_colors)]
                sign = '>' if direction == 'upper' else '<' if direction == 'lower' else ''
                legend = f"{name_s} {sign} {fmt.format(tv)}" if (direction and tv is not None) else name_s
                schema_map[key] = {
                    'name': name_s,
                    'color': color,
                    'legend': legend,
                    'direction': direction or 'unknown',
                    'value': tv,
                }
                schema_order.append(key)

        no_breach_color = '#888888'
        groups: Dict[str, Dict[str, Any]] = {}
        for k in schema_order:
            groups[k] = {'rows': [], 'color': schema_map[k]['color'], 'legend': schema_map[k]['legend']}
        groups['__NO__'] = {'rows': [], 'color': no_breach_color, 'legend': 'No breach'}

        def _infer_direction_row(row: Any, at_time: bool) -> Optional[str]:
            try:
                if at_time:
                    fv = row.get('db_field_value')
                    tv = row.get('review_value')
                else:
                    fv = row.get('end_db_field_value')
                    tv = row.get('end_review_value')
                fv = float(fv) if fv is not None else None
                tv = float(tv) if tv is not None else None
            except Exception:
                return None
            if fv is None or tv is None:
                return None
            if fv > tv:
                return 'upper'
            if fv < tv:
                return 'lower'
            return None

        if inputs.plot_type == 'value_at_time':
            for _, r in df.iterrows():
                status = r.get('review_status')
                if pd.isna(status):
                    groups['__NO__']['rows'].append(r)
                    continue
                direction = _infer_direction_row(r, True)
                base = str(status)
                comp_key = f"{base}::{direction}" if direction else base
                key_to_use = comp_key if comp_key in groups else (base if base in groups else '__NO__')
                groups[key_to_use]['rows'].append(r)
        else:
            for _, r in df.iterrows():
                status = r.get('end_review_status')
                if pd.isna(status):
                    groups['__NO__']['rows'].append(r)
                    continue
                direction = _infer_direction_row(r, False)
                base = str(status)
                comp_key = f"{base}::{direction}" if direction else base
                key_to_use = comp_key if comp_key in groups else (base if base in groups else '__NO__')
                groups[key_to_use]['rows'].append(r)

        try:
            center_conv = converter([("__CENTER__", center_e, center_n)])
            if not center_conv:
                raise ValueError("Center coordinate conversion failed")
            _, center_lat, center_lon = center_conv[0]
        except Exception as e:
            return {"content": f"Error: Failed to convert center coordinate: {str(e)}", "artefacts": []}
        fig = go.Figure()
        all_data: List[Dict[str, Any]] = []


        def severity_key(name: str) -> float:
            meta = schema_map.get(name, {})
            val = meta.get('value')
            direction = meta.get('direction')
            if val is None:
                return -1e12
            if direction == 'upper':
                return val
            elif direction == 'lower':
                return -val
            return -1e12
        try:
            sortable = [k for k in schema_map.keys()]
            sortable.sort(key=severity_key)
            ordered_keys = (['__NO__'] if groups['__NO__']['rows'] else []) + sortable
        except Exception:
            ordered_keys = (['__NO__'] if groups['__NO__']['rows'] else []) + list(schema_map.keys())
        added_signatures: set = set()
        for key in ordered_keys:
            g = groups.get(key)
            if not g or not g['rows']:
                continue
            lats: List[float] = []
            lons: List[float] = []
            texts: List[str] = []
            ids: List[str] = []
            series_name = g['legend']
            color = g['color']
            df_grp = pd.DataFrame(g['rows'])
            for _, r in df_grp.iterrows():
                id_str = str(r.get('instrument_id'))
                try:
                    e_val = float(r.get('easting'))
                    n_val = float(r.get('northing'))
                except Exception:
                    continue
                try:
                    conv = converter([(id_str, e_val, n_val)])
                except Exception:
                    continue
                if not conv:
                    continue
                _, lat, lon = conv[0]
                if inputs.plot_type == 'value_at_time':
                    status_txt = str(r.get('review_status')) if pd.notna(r.get('review_status')) else 'No breach'
                    val = r.get('db_field_value')
                    ts = r.get('db_field_value_timestamp')
                    try:
                        val_str = f"{float(val):.3f} {s0.abbreviated_unit}"
                    except Exception:
                        val_str = f"{val} {s0.abbreviated_unit}"
                    txt = f"<b>{id_str}</b><br><b>{status_txt}</b><br>{val_str}<br>at {ts}"
                else:
                    end_status = str(r.get('end_review_status')) if pd.notna(r.get('end_review_status')) else 'No breach'
                    start_status = str(r.get('start_review_status')) if pd.notna(r.get('start_review_status')) else 'No breach'
                    ev = r.get('end_db_field_value'); sv = r.get('start_db_field_value')
                    et = r.get('end_db_field_value_timestamp'); st = r.get('start_db_field_value_timestamp')
                    try:
                        ev_str = f"{float(ev):.3f} {s0.abbreviated_unit}"
                    except Exception:
                        ev_str = f"{ev} {s0.abbreviated_unit}"
                    try:
                        sv_str = f"{float(sv):.3f} {s0.abbreviated_unit}"
                    except Exception:
                        sv_str = f"{sv} {s0.abbreviated_unit}"
                    txt = (
                        f"<b>{id_str}</b><br>Changed to:<br><b>{end_status}</b><br>{ev_str}<br>at {et}"
                        f"<br>From:<br><b>{start_status}</b><br>{sv_str}<br>at {st}"
                    )
                ids.append(id_str)
                lats.append(lat)
                lons.append(lon)
                texts.append(txt)
                all_data.append({
                    'Instrument ID': id_str,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Value/Color': color,
                    'Series': series_name,
                    'Unit': s0.abbreviated_unit
                })

            if not lats:
                continue
            signature = (series_name, tuple(sorted(ids)))
            if signature in added_signatures:
                continue
            added_signatures.add(signature)
            hoverlabel_cfg = {"font": {"color": "white"}, "bordercolor": "white"} if key == '__NO__' else None
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=go.scattermapbox.Marker(size=10, color=color, showscale=False),
                text=texts,
                hovertemplate="%{text}<extra></extra>",
                name=series_name,
                hoverlabel=hoverlabel_cfg
            ))

        if inputs.plot_type == 'value_at_time':
            title_txt = f"Review status at {format_datetime(inputs.end_time)}"
        else:
            title_txt = f"Review status changes between {format_datetime(inputs.start_time)} and {format_datetime(inputs.end_time)}"

        fig.update_layout(
            title=title_txt,
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            mapbox_zoom=15,
            showlegend=True,
        )

        if not all_data:
            return {"content": "No data found for any series in the specified criteria. Check parameters or database.", "artefacts": []}
        df_out = pd.DataFrame(all_data)
        csv_content = df_out.to_csv(index=False)

        plotly_filename = f"map_plot_{uuid.uuid4()}.json"
        try:
            plotly_json = fig.to_json()
        except Exception as e:
            return {"content": f"Error: Failed to serialize Plotly figure: {str(e)}", "artefacts": []}

        artefacts = [
            {
                'artifact_id': str(uuid.uuid4()),
                'filename': plotly_filename,
                'type': 'Plotly object',
                'description': 'Map plot JSON',
                'content': plotly_json
            },
            {
                'artifact_id': str(uuid.uuid4()),
                'filename': f"map_data_{uuid.uuid4()}.csv",
                'type': 'CSV',
                'description': 'Map data CSV',
                'content': csv_content
            }
        ]

        content = f"Generated map plot and CSV for review_levels."
        return {"content": content, "artefacts": artefacts}

    def _render_readings(
        self,
        inputs: MapPlotInput,
        center_e: float,
        center_n: float,
        min_e: float,
        max_e: float,
        min_n: float,
        max_n: float,
        exclude_clause: str,
        start_time_str: Optional[str],
        end_time_str: str,
        buffer_start: Optional[str],
        buffer_end: str,
        converter,
    ) -> Dict[str, Any]:
        data: Dict[str, List[Tuple[str, float, float, float]]] = {}
        suggestions: List[str] = []
        for s in inputs.series:
            key = f"{s.instrument_type}_{s.instrument_subtype}_{s.database_field_name}"
            logger.info(f"Processing series: {key}")
            points: List[Tuple[str, float, float, float]] = []
            if inputs.data_type == 'readings':
                points = self._get_readings(inputs, s, min_e, max_e, min_n, max_n, exclude_clause, start_time_str, end_time_str, buffer_start, buffer_end)
                if points:
                    vals = [v for _, _, _, v in points]
                    logger.info(f"Values for outlier detection: {vals}")
                    mean = np.mean(vals)
                    std = np.std(vals)
                    outliers = [id for id, _, _, v in points if abs(v - mean) > 3 * std]
                    logger.info(f"Outliers detected: {outliers}")
                    suggestions.extend(outliers)
            data[key] = []
            if points:
                for (id, e, n, v) in points:
                    try:
                        conv = converter([(id, e, n)])
                    except Exception:
                        continue
                    if not conv:
                        continue
                    _, lat, lon = conv[0]
                    data[key].append((id, lat, lon, v))

        if not any(data.values()):
            logger.warning("No data found for any series")
            return {"content": "No data found for any series in the specified criteria. Check parameters or database.", "artefacts": []}

        try:
            center_conv = converter([("__CENTER__", center_e, center_n)])
            if not center_conv:
                raise ValueError("Center coordinate conversion failed")
            _, center_lat, center_lon = center_conv[0]
        except Exception as e:
            logger.error(f"Failed to convert center coordinate: {e}")
            return {"content": f"Error: Failed to convert center coordinate: {str(e)}", "artefacts": []}
        logger.info(f"Center lat/lon: {center_lat}, {center_lon}")
        fig = go.Figure()
        logger.info("Plotly figure initialized")
        for i, s in enumerate(inputs.series):
            key = f"{s.instrument_type}_{s.instrument_subtype}_{s.database_field_name}"
            points2 = data.get(key, [])
            if not points2:
                logger.info(f"No points for series {key}")
                continue
            instr_ids, lats, lons, vals = zip(*points2)
            logger.info(f"Adding readings trace for series {key}")
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
            customdata = list(zip(instr_ids, vals))
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
                customdata=customdata,
                hovertemplate="%{customdata[0]}<br>%{customdata[1]:.3f} " + s.abbreviated_unit + "<extra></extra>",
                name=f"{s.measured_quantity_name} ({s.abbreviated_unit})"
            ))
            logger.info(f"Added readings trace")

        time_str = f"at {format_datetime(inputs.end_time)}" if inputs.plot_type == 'value_at_time' else f"change from {format_datetime(inputs.start_time)} to {format_datetime(inputs.end_time)}"
        logger.info(f"Updating layout with title: {inputs.data_type.capitalize()} {time_str}")
        fig.update_layout(
            title=f"{inputs.data_type.capitalize()} {time_str}",
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            mapbox_zoom=15,
            showlegend=False
        )
        logger.info("Figure layout updated")

        all_data: List[Dict[str, Any]] = []
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
        logger.info(f"all_data populated with {len(all_data)} entries")
        if not all_data:
            logger.error("No data for CSV generation")
            raise ValueError("No data available for CSV generation")
        df = pd.DataFrame(all_data)
        logger.info(f"DataFrame created with shape {df.shape}")
        csv_content = df.to_csv(index=False)
        logger.info(f"CSV content generated (length: {len(csv_content)})")

        plotly_filename = f"map_plot_{uuid.uuid4()}.json"
        try:
            plotly_json = fig.to_json()
            logger.info(f"Plotly JSON generated (length: {len(plotly_json)})")
        except Exception as e:
            logger.error(f"Failed to serialize Plotly figure: {str(e)}")
            raise ValueError(f"Failed to serialize Plotly figure: {str(e)}")

        artefacts = [
            {
                'artifact_id': str(uuid.uuid4()),
                'filename': plotly_filename,
                'type': 'Plotly object',
                'description': 'Map plot JSON',
                'content': plotly_json
            },
            {
                'artifact_id': str(uuid.uuid4()),
                'filename': f"map_data_{uuid.uuid4()}.csv",
                'type': 'CSV',
                'description': 'Map data CSV',
                'content': csv_content
            }
        ]
        logger.info(f"Artefacts list created with {len(artefacts)} items")

        content = f"Generated map plot and CSV for {inputs.data_type} {time_str}." + (f" Suggested outliers to exclude: {', '.join(set(suggestions))}" if suggestions and inputs.data_type == 'readings' else "")
        logger.info(f"Returning content: {content}, artefacts count: {len(artefacts)}")
        return {"content": content, "artefacts": artefacts}

    def _get_readings(self, inputs: MapPlotInput, s: SeriesDict, min_e: float, max_e: float, min_n: float, max_n: float, exclude_clause: str, 
                      start_time_str: Optional[str], end_time_str: str, buffer_start: Optional[str], buffer_end: str) -> List[Tuple]:
        field = s.database_field_name
        logger.info(f"Getting readings for field={field}")
        is_data = field.startswith('data')
        
        if is_data:
            value_extract = field
            extra_conditions = f"AND {field} != ''"
        else:
            field_path = f'$.{field}'
            value_extract = f"CASE WHEN m.custom_fields IS NOT NULL AND LENGTH(m.custom_fields) > 0 AND JSON_VALID(m.custom_fields) THEN JSON_EXTRACT(m.custom_fields, '{field_path}') ELSE NULL END"
            extra_conditions = f"AND m.custom_fields IS NOT NULL AND LENGTH(m.custom_fields) > 0 AND JSON_VALID(m.custom_fields) " \
                               f"AND {value_extract} IS NOT NULL " \
                               f"AND {value_extract} != 'null' " \
                               f"AND {value_extract} != ''"
        
        safe_eval = lambda s: eval(s, {"__builtins__": {}}, {
            "datetime": datetime_module,
            "date": datetime_module.date,
            "time": datetime_module.time,
            "Decimal": decimal.Decimal,
            "bytes": bytes,
            "bytearray": bytearray
        }) if s != "No data was found in the database matching the specified search criteria." else []

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
            logger.info(f"Executing readings query for value_at_time: {query}")
            raw_result = self.sql_tool._run(query)
            logger.info(f"Raw SQL result: {raw_result!r}")
            raw_data = safe_eval(raw_result)
            logger.info(f"Parsed eval result: {raw_data}")
            if not raw_data:
                logger.warning(f"No data returned for series {s.instrument_type}_{s.instrument_subtype}_{s.database_field_name}")
                return []
            data = []
            for row in raw_data:
                logger.info(f"Processing row: {row}")
                norm = self._extract_row_reading(row)
                if not norm:
                    logger.warning(f"Skipping unrecognized row format: {row}")
                    continue
                id_, e, n, v = norm
                logger.info(f"Raw extracted values: id={id_!r}, easting={e!r}, northing={n!r}, value={v!r}")
                e_clean = self._extract_numeric(e)
                n_clean = self._extract_numeric(n)
                v_clean = self._extract_numeric(v)
                logger.info(f"Cleaned values: easting={e_clean!r}, northing={n_clean!r}, value={v_clean!r}")
                if None in (e_clean, n_clean, v_clean):
                    logger.warning(f"Invalid cleaned values after normalization: easting={e!r}, northing={n!r}, value={v!r}")
                    continue
                try:
                    fe = float(e_clean); fn = float(n_clean); fv = float(v_clean)
                    data.append((id_, fe, fn, fv))
                    logger.info(f"Appended valid data point: {id_}, {fe}, {fn}, {fv}")
                except ValueError as conv_err:
                    logger.warning(f"ValueError converting numeric strings: {conv_err}, row: {row}")
                    continue
            logger.info(f"Collected {len(data)} valid data points")
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
            logger.info(f"Executing end readings query for change_over_period: {query_end}")
            raw_end = safe_eval(self.sql_tool._run(query_end))
            result_end = []
            for row in raw_end:
                norm = self._extract_row_change(row, 'value_end')
                if not norm:
                    continue
                instr_id, val_end = norm
                val_end_clean = self._extract_numeric(val_end)
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
            logger.info(f"Executing start readings query for change_over_period: {query_start}")
            raw_start = safe_eval(self.sql_tool._run(query_start))
            result_start = []
            for row in raw_start:
                norm = self._extract_row_change(row, 'value_start')
                if not norm:
                    continue
                instr_id, val_start = norm
                val_start_clean = self._extract_numeric(val_start)
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
                logger.info("No changes found")
                return []
            instr_ids_str = ','.join(f"'{id_}'" for id_ in changes)
            loc_query = f"""
            SELECT i.instr_id, l.easting, l.northing
            FROM instrum i JOIN location l ON i.location_id = l.id
            WHERE i.instr_id IN ({instr_ids_str});
            """
            logger.info(f"Executing location query: {loc_query}")
            raw_locs = safe_eval(self.sql_tool._run(loc_query))
            loc_dict = {}
            for row in raw_locs:
                if isinstance(row, dict):
                    id_ = row.get('instr_id') or row.get('id')
                    e = row.get('easting') or row.get('east')
                    n = row.get('northing') or row.get('north')
                elif isinstance(row, (list, tuple)) and len(row) == 3:
                    id_, e, n = row
                else:
                    continue
                e_clean = self._extract_numeric(e)
                n_clean = self._extract_numeric(n)
                if None in (e_clean, n_clean) or id_ is None:
                    continue
                try:
                    loc_dict[id_] = (float(e_clean), float(n_clean))
                except ValueError:
                    continue
            data = [(id_, loc_dict[id_][0], loc_dict[id_][1], changes[id_]) for id_ in changes if id_ in loc_dict]
            logger.info(f"Collected {len(data)} change data points")
            return data

    def _run(
        self,
        data_type: str,
        plot_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        buffer_period_hours: Optional[int] = None,
        series: Optional[List[SeriesDict]] = None,
        center_instrument_id: Optional[str] = None,
        center_easting: Optional[float] = None,
        center_northing: Optional[float] = None,
        radius_meters: Optional[float] = None,
        exclude_instrument_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        logger.info(f"MapPlotTool._run called with data_type={data_type}, plot_type={plot_type}, start_time={start_time}, end_time={end_time}, buffer_period_hours={buffer_period_hours}, series_count={len(series) if series else 0}, center_instrument_id={center_instrument_id}, center_easting={center_easting}, center_northing={center_northing}, radius_meters={radius_meters}, exclude_instrument_ids={exclude_instrument_ids}")
        try:
            if buffer_period_hours is None:
                buffer_period_hours = 72
            if radius_meters is None:
                radius_meters = 500.0
            if exclude_instrument_ids is None:
                exclude_instrument_ids = []
            if end_time is None:
                raise ValueError("end_time is required")
            if series is None:
                raise ValueError("series is required and must be a non-empty list")

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
            logger.info(f"Validated inputs: {inputs}")

            if inputs.plot_type == 'change_over_period' and inputs.start_time is None:
                raise ValueError("start_time is required for 'change_over_period'")
            if inputs.start_time is not None and inputs.end_time <= inputs.start_time:
                raise ValueError("end_time must be later than start_time")
            if not inputs.series:
                raise ValueError("At least one series is required")

            center_e, center_n = self._get_center_coords(inputs.center_instrument_id, inputs.center_easting, inputs.center_northing)
            min_e, max_e, min_n, max_n = self._get_bounds(center_e, center_n, inputs.radius_meters)
            exclude_str = ','.join(f"'{id}'" for id in inputs.exclude_instrument_ids)
            exclude_clause = f"AND i.instr_id NOT IN ({exclude_str})" if exclude_str else ""
            logger.info(f"Exclude clause: {exclude_clause}")
            start_time_str = inputs.start_time.strftime("%Y-%m-%d %H:%M:%S") if inputs.start_time else None
            end_time_str = inputs.end_time.strftime("%Y-%m-%d %H:%M:%S")
            buffer_end = (inputs.end_time - timedelta(hours=inputs.buffer_period_hours)).strftime("%Y-%m-%d %H:%M:%S")
            buffer_start = (inputs.start_time - timedelta(hours=inputs.buffer_period_hours)).strftime("%Y-%m-%d %H:%M:%S") if inputs.start_time else None
            logger.info(f"Time strings: start_time_str={start_time_str}, end_time_str={end_time_str}, buffer_start={buffer_start}, buffer_end={buffer_end}")

            try:
                converter = create_coordinate_converter(self.sql_tool)
            except Exception as e:
                logger.error(f"Failed to instantiate coordinate converter: {e}")
                return {"content": f"Error: Failed to set up coordinate conversion: {str(e)}", "artefacts": []}

            if inputs.data_type == 'review_levels':
                return self._render_review_levels(inputs, center_e, center_n, min_e, max_e, min_n, max_n, converter)
            return self._render_readings(inputs, center_e, center_n, min_e, max_e, min_n, max_n, exclude_clause, start_time_str, end_time_str, buffer_start, buffer_end, converter)
        
        except ValueError as e:
            logger.error(f"ValueError in MapPlotTool._run: {str(e)}")
            return {"content": f"Error: Invalid input - {str(e)}. Correct and retry.", "artefacts": []}
        except Exception as e:
            logger.error(f"Unexpected error in MapPlotTool._run: {str(e)}", exc_info=True)
            return {"content": f"Error: Unexpected failure - {str(e)}. Verify database connection, instrument existence, or numeric data in columns.", "artefacts": []}

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
        logger.info(f"MapPlotWrapperTool._run called")
        logger.info(f"Input JSON: {input_json}")
        try:
            # Remove JSON code block markers if present
            input_json = re.sub(r'^```json\s*\n|\s*```$', '', input_json, flags=re.MULTILINE)
            # Replace single quotes with double quotes for valid JSON
            input_json = input_json.replace("'", '"')
            # Replace "None" with "null" for JSON compatibility
            input_json = re.sub(r':\s*None\b', ': null', input_json)
            input_dict = json.loads(input_json)
            logger.info(f"Parsed input dictionary: {input_dict}")

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
            logger.info(f"Plot tool result: {content}, {artefacts}")

            return json.dumps({'content': content, 'artifacts': artefacts})
        except KeyError as e:
            return json.dumps({'content': f"Error processing input: Missing required field {str(e)}", 'artifacts': []})
        except ValueError as e:
            return json.dumps({'content': f"Error processing input: {str(e)}", 'artifacts': []})
        except Exception as e:
            return json.dumps({'content': f"Error: {str(e)}", 'artifacts': []})