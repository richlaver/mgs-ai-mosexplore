from langchain_core.tools import BaseTool, StructuredTool
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Type, Any
from enum import Enum
from prompts import prompts
import re
import json
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TimeUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"

class DatetimeShiftToolInput(BaseModel):
    """Input model for the datetime calculator tool."""
    input_datetime: str = Field(
        description="""
The datetime string in format 'D[D] MonthName YYYY H[H]:MM:SS AM/PM', e.g., 
'20 July 2025 05:53:22 PM'
"""
    )
    operation: str = Field(
        default="add",
        description="""
The operation to perform: 'add' or 'subtract'. Defaults to 'add'.
"""
    )
    value: float = Field(
        description="""
The numerical value of the time period to add or subtract. 
Negative values imply subtraction.
"""
    )
    unit: TimeUnit = Field(
        description="""
The unit of the time period: 'seconds', 'minutes', 'hours', 'days', 'weeks', 
'months', or 'years'
"""
    )

class DatetimeShiftToolOutput(BaseModel):
    """Output model for the datetime calculator tool."""
    result_datetime: str = Field(
        description="The resulting datetime string in format 'DD MonthName YYYY HH:MM:SS AM/PM'"
    )

class DatetimeShiftTool(BaseTool):
    """A tool to add or subtract a time period from a specified datetime."""
    name: str = "datetime_shift_tool"
    description: str = """
Calculates a new datetime by adding or subtracting a time period from a 
given datetime.
Inputs:
- input_datetime: String in format 'D[D] MonthName YYYY H[H]:MM:SS AM/PM' 
(e.g., '20 July 2025 05:53:22 PM').
- operation: String, either 'add' or 'subtract' 
(defaults to 'add'; negative value implies subtraction).
- value: Float, the time period to add/subtract (e.g., 5.5 for 5.5 hours).
- unit: String, one of 'seconds', 'minutes', 'hours', 'days', 'weeks', 
'months', 'years'.
Returns:
- result_datetime: String in format 'DD MonthName YYYY HH:MM:SS AM/PM'.
Example: {'input_datetime': '07 August 2025 06:46:21 PM', 'operation': 
'subtract', 'value': 1, 'unit': 'days'} returns '06 August 2025 06:46:21 PM'.
"""
    args_schema: Type[DatetimeShiftToolInput] = DatetimeShiftToolInput

    def _run(
        self,
        input_datetime: str,
        operation: str = "add",
        value: float = None,
        unit: TimeUnit = None,
        **kwargs
    ) -> DatetimeShiftToolOutput:
        """Add or subtract the time period from the datetime."""
        try:
            if ' 00:' in input_datetime:
                input_datetime = input_datetime.replace(' 00:', ' 12:')
            dt = datetime.strptime(input_datetime, "%d %B %Y %I:%M:%S %p")
            effective_operation = "subtract" if value < 0 else operation
            abs_value = abs(value)
            value_int = int(abs_value) if unit in [
                TimeUnit.DAYS,
                TimeUnit.WEEKS,
                TimeUnit.MONTHS,
                TimeUnit.YEARS
            ] else abs_value
            delta_args = {unit.value: value_int}
            
            if unit in [
                TimeUnit.SECONDS,
                TimeUnit.MINUTES,
                TimeUnit.HOURS,
                TimeUnit.DAYS
            ]:
                delta = relativedelta(**delta_args)
            elif unit == TimeUnit.WEEKS:
                delta = relativedelta(days=value_int * 7)
            else:
                delta = relativedelta(**delta_args)
            
            result_dt = dt + delta if effective_operation == "add" else dt - delta
            result_str = result_dt.strftime("%d %B %Y %I:%M:%S %p")
            
            return DatetimeShiftToolOutput(result_datetime=result_str)
        
        except Exception as e:
            error_message = f"Error processing datetime calculation: {str(e)}"
            logging.error(error_message)
            raise ValueError(error_message)

class DatetimeShiftWrapperInput(BaseModel):
    """Input model for the datetime shift wrapper tool."""
    input_json: str = Field(
        description="""
A JSON string containing input_datetime, operation, value, and unit, e.g., 
'{"input_datetime": "07 August 2025 06:46:21 PM", "operation": "subtract", 
"value": 1, "unit": "days"}'
"""
    )

class DatetimeShiftWrapperTool(BaseTool):
    """A wrapper tool to parse JSON input and call the DatetimeShiftTool."""
    name: str = "datetime_shift_wrapper"
    description: str = """
A wrapper tool that accepts a JSON string to add or subtract a time period from 
a datetime.
Input:
- input_json: A JSON string with fields:
    - input_datetime: String in format 'D[D] MonthName YYYY H[H]:MM:SS AM/PM' 
    (e.g., '20 July 2025 05:53:22 PM').
    - operation: String, either 'add' or 'subtract' (defaults to 'add').
    - value: Float, the time period to add/subtract (e.g., 5.5 for 5.5 hours).
    - unit: String, one of 'seconds', 'minutes', 'hours', 'days', 'weeks', 
    'months', 'years'.
Returns:
- result_datetime: String in format 'DD MonthName YYYY HH:MM:SS AM/PM'.
Example: '{"input_datetime": "07 August 2025 06:46:21 PM", "operation": 
"subtract", "value": 1, "unit": "days"}' returns '06 August 2025 06:46:21 PM'.
"""
    args_schema: Type[DatetimeShiftWrapperInput] = DatetimeShiftWrapperInput

    def _run(self, input_json: str, **kwargs) -> DatetimeShiftToolOutput:
        """Parse JSON input and call the DatetimeShiftTool."""
        try:
            # Replace single quotes with double quotes for JSON compatibility
            input_json = re.sub(r'^```json\s*\n|\s*```$', '', input_json, 
                                flags=re.MULTILINE)
            input_json = input_json.replace("'", "\"")
            input_dict = json.loads(input_json)
            tool = DatetimeShiftTool()
            return tool._run(
                input_datetime=input_dict.get("input_datetime"),
                operation=input_dict.get("operation", "add"),
                value=input_dict.get("value"),
                unit=TimeUnit(input_dict.get("unit"))
            )
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON input: {str(e)}"
            logging.error(error_message)
            raise ValueError(error_message)
        except ValueError as e:
            error_message = f"Invalid unit value: {str(e)}"
            logging.error(error_message)
            raise ValueError(error_message)
        except Exception as e:
            error_message = f"Error processing wrapper input: {str(e)}"
            logging.error(error_message)
            raise ValueError(error_message)


class GetDatetimeNowTool(BaseTool):
    """
    Tool to get date and time now.
    """

    name: str = "get_datetime_now"
    description: str = """
    Use this tool to get the date and time now.
    Use this tool to interpret a date range when the user's query mentions:
    - words pertaining to the current time e.g. "now", "today", "at the moment",
      "currently", "right now" etc.
    - words pertaining to a time period or moment in relation to the current 
    time e.g. "last week", "this month", "next year", "current week", "recent 
    month", "latest week", "yesterday", "tomorrow", "day after tomorrow", "day 
    before yesterday", "a year from now", 
    "a month ago", "within a week from now", "in two days"
    The tool returns the current date and time as a string in the format:
    20 August 2023 01:23:45 PM
    """


    def _run(self, *args, **kwargs) -> str:
        """
        Run the tool to get the date and time now.
        Returns a string with the current date and time in the following format:
        20 August 2023 01:23:45 PM
        """

        return datetime.now().strftime("%d %B %Y %I:%M:%S %p")