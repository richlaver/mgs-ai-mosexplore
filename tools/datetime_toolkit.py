from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Literal, Type, List
from enum import Enum
from prompts import prompts
import re
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
        description="""The datetime from which to add or subtract, as a string 
        in the format 'D[D] MonthName YYYY H[H]:MM:SS AM/PM', e.g., '20 July 
        2025 05:53:22 PM' or '2 July 2025 5:53:22 PM'"""
    )
    operation: str = Field(
        default="add",
        description="The operation to perform: 'add' or 'subtract'. Defaults to 'add' if omitted. If value is negative, operation is inferred as 'subtract'."
    )
    value: float = Field(
        description="The numerical value of the time period to add or subtract. A negative value implies subtraction."
    )
    unit: TimeUnit = Field(
        description="""The unit of the time period: seconds, minutes, hours, 
        days, weeks, months, or years"""
    )

    @field_validator("input_datetime")
    @classmethod
    def validate_datetime(cls, v):
        """Validate the datetime string format, allowing flexible day and hour 
        digits."""
        try:
            # Try strict parsing with expected format
            datetime.strptime(v, "%d %B %Y %I:%M:%S %p")
        except ValueError:
            try:
                # Fallback to dateutil.parser for flexible parsing
                parsed_dt = parser.parse(v, dayfirst=True)
                # Verify the parsed datetime matches the expected format 
                # structure
                formatted_dt = parsed_dt.strftime("%d %B %Y %I:%M:%S %p")
                # Ensure the input string roughly matches the expected pattern
                if not re.match(r"""^\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2}
                                :\d{2}\s+(AM|PM)$""", v, re.IGNORECASE):
                    raise ValueError("Invalid datetime format")
                return formatted_dt  # Normalize to standard format
            except ValueError:
                raise ValueError(
                    """input_datetime must be in the format 'D[D] MonthName YYYY
                      H[H]:MM:SS AM/PM', e.g., '20 July 2025 05:53:22 PM' or '2 
                      July 2025 5:53:22 PM'"""
                )
        return v

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v):
        """Ensure operation is either 'add' or 'subtract'."""
        if v not in ["add", "subtract"]:
            raise ValueError("Operation must be 'add' or 'subtract'")
        return v

class DatetimeShiftToolOutput(BaseModel):
    """Output model for the datetime calculator tool."""
    result_datetime: str = Field(
        description="""The resulting datetime string after calculation in the 
        format 'DD MonthName YYYY HH:MM:SS AM/PM'"""
    )

class DatetimeShiftTool(BaseTool):
    """A tool to add or subtract a time period from a specified datetime."""
    
    name: str = "add_or_subtract_datetime"
    description: str = """
    Calculates a new datetime by adding or subtracting a time period from a given datetime.
    Required arguments:
    - input_datetime: String in the format 'D[D] MonthName YYYY H[H]:MM:SS AM/PM' (e.g., '20 July 2025 05:53:22 PM').
    - operation: String, either 'add' or 'subtract' (defaults to 'add' if omitted; inferred as 'subtract' if value is negative).
    - value: Float, the time period to add/subtract (e.g., 5.5 for 5.5 hours; negative values imply subtraction).
    - unit: String, one of 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'.
    Returns the resulting datetime in the format 'DD MonthName YYYY HH:MM:SS AM/PM'.
    Example: {'input_datetime': '20 July 2025 05:53:22 PM', 'operation': 'add', 'value': 2, 'unit': 'days'} returns '22 July 2025 05:53:22 PM'.
    Do not use incorrect field names like 'datetime' or 'days'.
    """
    args_schema: Type[BaseModel] = DatetimeShiftToolInput
    return_schema: Type[BaseModel] = DatetimeShiftToolOutput

    def _run(
        self,
        input_datetime: str,
        operation: str,
        value: float,
        unit: TimeUnit
    ) -> DatetimeShiftToolOutput:
        """Add or subtract the time period from the datetime."""
        try:
            # Parse the input datetime string
            dt = datetime.strptime(input_datetime, "%d %B %Y %I:%M:%S %p")
            
            # Determine operation based on value's sign
            effective_operation = "subtract" if value < 0 else operation
            
            # Use absolute value for calculations
            abs_value = abs(value)
            
            # Convert value to appropriate type (int for days, weeks, months, years; float for seconds, minutes, hours)
            value_int = int(abs_value) if unit in [TimeUnit.DAYS, TimeUnit.WEEKS, TimeUnit.MONTHS, TimeUnit.YEARS] else abs_value
            
            # Define the time delta based on unit
            delta_args = {unit.value: value_int}
            
            # Create the time delta
            if unit in [TimeUnit.SECONDS, TimeUnit.MINUTES, TimeUnit.HOURS, TimeUnit.DAYS]:
                delta = relativedelta(**delta_args)
            elif unit == TimeUnit.WEEKS:
                delta = relativedelta(days=value_int * 7)
            else:  # months or years
                delta = relativedelta(**delta_args)
            
            # Perform the operation
            if effective_operation == "add":
                result_dt = dt + delta
            else:  # subtract
                result_dt = dt - delta
            
            # Format the result
            result_str = result_dt.strftime("%d %B %Y %I:%M:%S %p")
            
            return DatetimeShiftToolOutput(result_datetime=result_str)
        
        except Exception as e:
            raise ValueError(f"Error processing datetime calculation: {str(e)}")


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