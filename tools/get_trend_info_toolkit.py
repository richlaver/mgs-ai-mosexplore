from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, Tuple, Type, Union, Optional, Dict, Any
from pydantic import Field
import json
import re
import logging
import numpy as np
import ast
from datetime import datetime, timedelta
import datetime as datetime_module
from .sql_security_toolkit import GeneralSQLQueryTool
from pydantic import validator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
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


class TrendPoint(BaseModel):
    """Model for a single time series data point."""
    timestamp: datetime
    value: float

class TrendExtractorInput(BaseModel):
    """Input model for extracting time series trend information from database."""
    instrument_id: str = Field(
        ..., 
        description="ID of the instrument to get data for"
    )
    start_time: datetime = Field(
        ...,
        description="Start time for time series data"
    )
    end_time: datetime = Field(
        ...,
        description="End time for time series data"
    )
    column_name: str = Field(
        ...,
        description="Name of the database column containing values e.g. 'data1' or 'calculation1'"
    )
    window_width_days: float = Field(
        ...,
        description="Width of smoothing window in days"
    )

    @validator('column_name')
    def validate_column_name(cls, v):
        import re
        if not re.match(r'^(data|calculation)\d+$', v):
            raise ValueError("Field name must be in format 'data<n>' or 'calculation<n>' where <n> is an integer")
        return v

class TrendExtractorOutput(BaseModel):
    """Output model for time series trend information."""
    error: Union[str, None] = Field(
        None,
        description="Error message if processing failed"
    )
    unsmoothed: Union[List[Tuple[datetime, float]], None] = Field(
        None,
        description="List of tuples containing (timestamp, unsmoothed value)"
    )
    smoothed: Union[List[Tuple[datetime, float]], None] = Field(
        None,
        description="List of tuples containing (timestamp, smoothed value)"
    )
    first_derivative: Union[List[Tuple[datetime, float]], None] = Field(
        None,
        description="List of tuples containing (timestamp, first derivative)"
    )
    second_derivative: Union[List[Tuple[datetime, float]], None] = Field(
        None,
        description="List of tuples containing (timestamp, second derivative)"
    )


class SmoothedDataAndRatesToolInput(BaseModel):
    """Input model for the smoothed data and rates calculation tool."""
    time_series: List[Tuple[datetime, float]] = Field(
        ...,
        description="List of tuples containing (timestamp, value) pairs"
    )
    window_width_days: float = Field(
        ...,
        description="Width of the smoothing window in days"
    )


class SmoothedDataAndRatesToolOutput(BaseModel):
    """Output model for the smoothed data and rates calculation tool."""
    error: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )
    unsmoothed: Optional[List[Tuple[datetime,float]]] = Field(
        None,
        description="List of tuples containing (timestamp, unsmoothed value)"
    )
    smoothed: Optional[List[Tuple[datetime,float]]] = Field(
        None,
        description="List of tuples containing (timestamp, smoothed value)"
    )
    first_derivative: Optional[List[Tuple[datetime,float]]] = Field(
        None,
        description="List of tuples containing (timestamp, first derivative)"
    )
    second_derivative: Optional[List[Tuple[datetime,float]]] = Field(
        None,
        description="List of tuples containing (timestamp, second derivative)"
    )


class SmoothedDataAndRatesTool(BaseTool):
    """
Tool for calculating smoothed rates and derivatives from time series data."""

    name: str = "get_smoothed_data_and_rates"
    description: str = """
Process time series data using a smoothing window to calculate either:
- Smoothed values
- First derivative with respect to time
- Second derivative with respect to time
    """
    args_schema: Type[SmoothedDataAndRatesToolInput] = SmoothedDataAndRatesToolInput

    def _get_local_window_width(
        self,
        current_time: float,
        time_range: Tuple[float, float],
        full_window_width: float
    ) -> float:
        """Calculate local window width based on distance from edges."""
        half_width = full_window_width / 2.0
        start_time, end_time = time_range
        
        # Distance to nearest edge
        dist_to_start = current_time - start_time
        dist_to_end = end_time - current_time
        min_dist_to_edge = min(dist_to_start, dist_to_end)
        
        if min_dist_to_edge >= half_width:
            # Not near an edge, use full window
            return full_window_width
        else:
            # Near an edge, use reduced window
            return 2.0 * min_dist_to_edge


    def _run(
        self,
        time_series: List[Tuple[datetime, float]],
        window_width_days: float
    ) -> SmoothedDataAndRatesToolOutput:
        """Process time series data, computing smoothed values and derivatives."""
        logger.debug(f"SmoothedDataAndRatesTool._run called with {len(time_series)} points and window_width={window_width_days}")
        try:
            # Input validation
            if not time_series:
                return SmoothedDataAndRatesToolOutput(error="Input time series is empty")
            if window_width_days <= 0:
                return SmoothedDataAndRatesToolOutput(
                    error=f"Invalid window width: {window_width_days}. Must be greater than 0"
                )

            # Filter and validate data points
            filtered_series = [
                (t, float(v)) for t, v in time_series 
                if v is not None and str(v).strip().lower() != "null"
            ]
            logger.debug(f"After filtering null values: {len(filtered_series)} points remaining")
            
            if not filtered_series:
                return SmoothedDataAndRatesToolOutput(
                    error="No valid data points after filtering null values"
                )

            # Convert to numpy arrays and sort by time
            sorted_series = sorted(filtered_series)
            timestamps = [t for t, _ in sorted_series]
            base_time = sorted_series[0][0]
            times = np.array([(t - base_time).total_seconds() / 86400 for t in timestamps])
            values = np.array([v for _, v in sorted_series])

            # Handle single point case
            if len(times) == 1:
                return SmoothedDataAndRatesToolOutput(
                    timestamps=timestamps,
                    smoothed=[values[0]],
                    first_derivative=[0.0],
                    second_derivative=[0.0]
                )

            # Initialize output arrays
            time_range = (times[0], times[-1])
            n_points = len(times)
            smoothed = np.zeros(n_points)
            first_deriv = np.zeros(n_points)
            second_deriv = np.zeros(n_points)

            # Process each point once, computing everything we need
            for i in range(n_points):
                # Get window data
                t = times[i]
                local_width = self._get_local_window_width(t, time_range, window_width_days)
                mask = np.abs(times - t) <= local_width/2
                points_in_window = np.sum(mask)
                logger.debug(f"Processing point {i}/{n_points}: t={t:.2f}, local_width={local_width:.2f}, points_in_window={points_in_window}")

                if points_in_window >= 3:
                    # Enough points for quadratic fit
                    window_times = times[mask]
                    window_values = values[mask]
                else:
                    # Get nearest points for minimum viable fit
                    distances = np.abs(times - t)
                    k = min(3, len(times))  # Avoid requesting more points than we have
                    idx = np.argpartition(distances, k)[:k]
                    window_times = times[idx]
                    window_values = values[idx]

                # Fit quadratic: f(t) = at² + bt + c
                A = np.vstack([window_times**2, window_times, np.ones_like(window_times)]).T
                try:
                    (a, b, c), *_ = np.linalg.lstsq(A, window_values, rcond=None)
                    # f(t) = at² + bt + c
                    smoothed[i] = a*t**2 + b*t + c
                    # f'(t) = 2at + b
                    first_deriv[i] = 2*a*t + b
                    # f''(t) = 2a
                    second_deriv[i] = 2*a
                except np.linalg.LinAlgError:
                    # Fallback to linear fit if quadratic fails
                    A = np.vstack([window_times, np.ones_like(window_times)]).T
                    try:
                        (m, c), *_ = np.linalg.lstsq(A, window_values, rcond=None)
                        smoothed[i] = m*t + c
                        first_deriv[i] = m
                        second_deriv[i] = 0.0
                    except np.linalg.LinAlgError:
                        # Last resort: use point value
                        smoothed[i] = values[i]
                        first_deriv[i] = 0.0
                        second_deriv[i] = 0.0

            unsmoothed_output = [(timestamps[i], values[i]) for i in range(n_points)]
            smoothed_output = [(timestamps[i], smoothed[i]) for i in range(n_points)]
            first_deriv_output = [(timestamps[i], first_deriv[i]) for i in range(n_points)]
            second_deriv_output = [(timestamps[i], second_deriv[i]) for i in range(n_points)]

            logger.debug(f'unsmoothed_output: {unsmoothed_output}')
            logger.debug(f'smoothed_output: {smoothed_output}')
            logger.debug(f'first_deriv_output: {first_deriv_output}')
            logger.debug(f'second_deriv_output: {second_deriv_output}')

            return SmoothedDataAndRatesToolOutput(
                unsmoothed=unsmoothed_output,
                smoothed=smoothed_output,
                first_derivative=first_deriv_output,
                second_derivative=second_deriv_output
            )

        except Exception as e:
            return SmoothedDataAndRatesToolOutput(
                error=f"Error processing time series: {str(e)}"
            )


class SmoothedDataAndRatesWrapperInput(BaseModel):
    """Input model for the smoothed data and rates wrapper tool."""
    input_json: str = Field(
        description="""
A JSON string containing time_series, window_width_days and output_type, e.g.,
'{"time_series": [["2 August 2025 12:00:00 PM", 1.2], ["15 August 2025 2:30:00 PM", 1.3]], 
  "window_width_days": 7.0, 
  "output_type": "smoothed"}'
"""
    )


class SmoothedDataAndRatesWrapperTool(BaseTool):
    """A wrapper tool to parse JSON input and call the SmoothedDataAndRatesTool."""
    name: str = "smoothed_data_and_rates_wrapper"
    description: str = """
A wrapper tool that accepts a JSON string to calculate smoothed rates.
Input:
- input_json: A JSON string with fields:
    - time_series: List of [timestamp_string, value] pairs
    - window_width_days: Width of smoothing window in days
    - output_type: One of 'smoothed', 'first_derivative', 'second_derivative' 
                  (defaults to 'smoothed')
Returns:
- smoothed_series: List of [timestamp_string, value] pairs or error message string
"""
    args_schema: Type[SmoothedDataAndRatesWrapperInput] = SmoothedDataAndRatesWrapperInput

    def _run(self, input_json: str, **kwargs) -> SmoothedDataAndRatesToolOutput:
        """Parse JSON input and call the SmoothedRatesTool."""
        logger.debug(f"SmoothedDataAndRatesWrapperTool._run called with input JSON length: {len(input_json)}")
        try:
            input_json = re.sub(r'^```json\s*\n|\s*```$', '', input_json, 
                              flags=re.MULTILINE)
            input_json = input_json.replace("'", "\"")
            input_dict = json.loads(input_json)
            logger.debug(f"Parsed input dict keys: {list(input_dict.keys())}")
            
            if "time_series" not in input_dict:
                return SmoothedDataAndRatesToolOutput(
                    smoothed_series="Missing required field: time_series"
                )
            if "window_width_days" not in input_dict:
                return SmoothedDataAndRatesToolOutput(
                    smoothed_series="Missing required field: window_width_days"
                )
            
            # Convert timestamp strings to datetime objects, preserving null/None values
            try:
                time_series = [
                    (parse_datetime(t), v) 
                    for t, v in input_dict["time_series"]
                ]
            except ValueError as e:
                return SmoothedDataAndRatesToolOutput(
                    smoothed_series=str(e)
                )
            
            tool = SmoothedDataAndRatesTool()
            return tool._run(
                time_series=time_series,
                window_width_days=float(input_dict["window_width_days"]),
                output_type=input_dict.get("output_type", "smoothed")
            )
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON input: {str(e)}"
            logging.error(error_message)
            return SmoothedDataAndRatesToolOutput(smoothed_series=error_message)
        except Exception as e:
            error_message = f"Error processing wrapper input: {str(e)}"
            logging.error(error_message)
            return SmoothedDataAndRatesToolOutput(smoothed_series=error_message)


class BaseSQLQueryTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    sql_tool: GeneralSQLQueryTool = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class TrendExtractorTool(BaseTool, BaseSQLQueryTool):
    """Tool for extracting time series trend information from the database."""
    name: str = "trend_extractor"
    description: str = """
Extracts time series data from database, smoothing the data and finding first and second derivatives.
Use this tool to identify trends and changes in trends.
Handles data padding around time range to avoid window width reduction at edges.
    """
    args_schema: Type[TrendExtractorInput] = TrendExtractorInput
        
    def _run(
        self,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        column_name: str,
        window_width_days: float
    ) -> TrendExtractorOutput:
        """Extract and process time series data."""
        logger.debug(f"TrendExtractorTool._run called with params: instrument_id={instrument_id}, "
                    f"start_time={start_time}, end_time={end_time}, column_name={column_name}, "
                    f"window_width_days={window_width_days}")
        try:
            if window_width_days <= 0:
                return TrendExtractorOutput(
                    data=f"Window width must be positive, got {window_width_days}"
                )
                
            if start_time >= end_time:
                return TrendExtractorOutput(
                    data="Start time must be before end time"
                )
            
            # Calculate extended time range to get padding data
            padding = timedelta(days=window_width_days)
            extended_start = start_time - padding
            extended_end = end_time + padding
            
            # SQL templates for data extraction
            data_query_template = """
            SELECT 
                date1 as timestamp,
                {column_name} as value
            FROM mydata m
            WHERE m.instr_id = '{instrument_id}'
            AND m.date1 BETWEEN '{start_time}' AND '{end_time}'
            AND {column_name} IS NOT NULL
            AND {column_name} != ''
            ORDER BY date1;
            """

            calculation_query_template = """
            SELECT 
                date1 as timestamp,
                JSON_EXTRACT(custom_fields, '$.{column_name}') as value
            FROM mydata m
            WHERE m.instr_id = '{instrument_id}'
            AND m.date1 BETWEEN '{start_time}' AND '{end_time}'
            AND custom_fields IS NOT NULL
            AND JSON_VALID(custom_fields)
            AND JSON_EXTRACT(custom_fields, '$.{column_name}') IS NOT NULL
            AND JSON_EXTRACT(custom_fields, '$.{column_name}') != 'null'
            AND JSON_EXTRACT(custom_fields, '$.{column_name}') != ''
            ORDER BY date1;
            """

            # Choose the appropriate template based on column_name
            if not (column_name.startswith('data') or column_name.startswith('calculation')):
                return TrendExtractorOutput(
                    data=f"Invalid field name: {column_name}. Must start with 'data' or 'calculation'"
                )

            # Select and format query template
            query_template = data_query_template if column_name.startswith('data') else calculation_query_template
            query = query_template.format(
                column_name=column_name,
                instrument_id=instrument_id,
                start_time=extended_start.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=extended_end.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            logger.debug(f"Executing SQL query:\n{query}")
            results = self.sql_tool._run(query)
            logger.debug(f"Query results: {results}")
            
            # Handle case where results is the specific no-data message
            if results == "No data was found in the database matching the specified search criteria.":
                logger.debug("No data found in database")
                return TrendExtractorOutput(data="No data returned from database")
                
            def clean_numeric_string(val_str):
                """Remove all surrounding single/double quotes and whitespace from a string."""
                cleaned = str(val_str).strip()
                # Iteratively remove surrounding single or double quotes
                while cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
                    cleaned = cleaned[1:-1].strip()
                # Validate that the cleaned string is a valid float
                if re.match(r'^-?\d*\.?\d+$', cleaned):
                    return cleaned
                return None

            try:
                # Use eval with restricted globals (only 'datetime' module allowed)
                parsed_data = eval(
                    results,
                    {"__builtins__": {}, "datetime": datetime_module},
                    {}
                )
                # Convert second element to float, skip invalid conversions
                valid_time_series = []
                for dt, val_str in parsed_data:
                    cleaned_val = clean_numeric_string(val_str)
                    if cleaned_val is None:
                        logger.debug(f"Skipping tuple with non-numeric value: '{val_str}'")
                        continue
                    try:
                        val = float(cleaned_val)
                        valid_time_series.append((dt, val))
                    except ValueError as e:
                        logger.debug(f"Skipping tuple with value '{cleaned_val}' (original: '{val_str}'): {str(e)}")
                        continue
                
                if not valid_time_series:
                    raise ValueError("No valid data points with numeric values found")
                
                logger.debug(f"Processed {len(valid_time_series)} data points")
            except (ValueError, SyntaxError, AttributeError, NameError) as e:
                logger.error(f"Error processing time series: {str(e)}")
                return TrendExtractorOutput(data="Error processing database results")
            
            logger.debug(f"Sample data point: {valid_time_series[0] if valid_time_series else 'No data'}")
            
            # Process data using SmoothedDataAndRatesTool
            smoother = SmoothedDataAndRatesTool()
            result = smoother._run(
                time_series=valid_time_series,
                window_width_days=window_width_days
            )
            logger.debug(f"SmoothedDataAndRatesTool result: {result}")
            
            if result.error:
                return TrendExtractorOutput(data=result.error)
            
            # Format output with 7 significant figures
            formatted_unsmoothed = [(t, float(f"{v:.7g}")) for t, v in result.unsmoothed]
            formatted_smoothed = [(t, float(f"{v:.7g}")) for t, v in result.smoothed]
            formatted_first_deriv = [(t, float(f"{v:.7g}")) for t, v in result.first_derivative]
            formatted_second_deriv = [(t, float(f"{v:.7g}")) for t, v in result.second_derivative]

            logger.debug(f'formatted_unsmoothed: {formatted_unsmoothed}')
            logger.debug(f'formatted_smoothed: {formatted_smoothed}')
            logger.debug(f'formatted_first_deriv: {formatted_first_deriv}')
            logger.debug(f'formatted_second_deriv: {formatted_second_deriv}')

            return TrendExtractorOutput(
                unsmoothed=formatted_unsmoothed,
                smoothed=formatted_smoothed,
                first_derivative=formatted_first_deriv,
                second_derivative=formatted_second_deriv
            )
            
        except Exception as e:
            return TrendExtractorOutput(
                data=f"Error processing time series: {str(e)}"
            )

class TrendExtractorWrapperInput(BaseModel):
    """Input model for the time series extractor wrapper tool."""
    input_json: str = Field(
        description="""
A JSON string containing:
- instrument_id: string ID of instrument
- start_time: datetime string in format "D Month YYYY H:MM:SS AM/PM"
- end_time: datetime string in format "D Month YYYY H:MM:SS AM/PM"
- column_name: database column name e.g. 'data1', 'calculation1'
- window_width_days: positive float
"""
    )

class TrendExtractorWrapperTool(BaseTool, BaseSQLQueryTool):
    """Wrapper tool to parse JSON input for TrendExtractorTool."""
    name: str = "time_series_extractor_wrapper"
    description: str = """
Wrapper tool that accepts a JSON string to extract and process time series data.
Returns processed time series with smoothed values and derivatives.
"""
    args_schema: Type[TrendExtractorWrapperInput] = TrendExtractorWrapperInput
    
    def _run(self, input_json: str, **kwargs) -> TrendExtractorOutput:
        """Parse JSON input and call TrendExtractorTool."""
        logger.debug(f"TrendExtractorWrapperTool._run called with input JSON length: {len(input_json)}")
        try:
            input_json = re.sub(r'^```json\s*\n|\s*```$', '', input_json, 
                              flags=re.MULTILINE)
            input_json = input_json.replace("'", "\"")
            input_dict = json.loads(input_json)
            logger.debug(f"Parsed input dict: {json.dumps(input_dict, indent=2)}")
            
            required_fields = [
                "instrument_id", "start_time", "end_time",
                "column_name", "window_width_days"
            ]
            
            missing = [f for f in required_fields if f not in input_dict]
            if missing:
                return TrendExtractorOutput(
                    data=f"Missing required fields: {', '.join(missing)}"
                )
            
            try:
                start_time = parse_datetime(input_dict["start_time"])
                end_time = parse_datetime(input_dict["end_time"])
            except ValueError as e:
                return TrendExtractorOutput(
                    data=str(e)
                )
            
            tool = TrendExtractorTool(sql_tool=self.sql_tool)
            return tool._run(
                instrument_id=str(input_dict["instrument_id"]),
                start_time=start_time,
                end_time=end_time,
                column_name=str(input_dict["column_name"]),
                window_width_days=float(input_dict["window_width_days"])
            )
            
        except json.JSONDecodeError as e:
            return TrendExtractorOutput(
                data=f"Invalid JSON input: {str(e)}"
            )
        except Exception as e:
            return TrendExtractorOutput(
                data=f"Error processing input: {str(e)}"
            )
