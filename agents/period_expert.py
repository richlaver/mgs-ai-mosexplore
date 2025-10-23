import datetime
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langgraph.types import Command
from pydantic import BaseModel, Field

from classes import RelevantDateRange, ContextState


class PeriodExpertOutput(BaseModel):
    """Structured output containing relevant date ranges."""
    relevant_date_ranges: List[RelevantDateRange] = Field(
        description="List of all relevant date ranges mentioned or implied in the query"
    )


def period_expert(state: ContextState, llm: BaseLanguageModel, db: any) -> dict:
    """
    Deduce relevant date ranges for data extraction from the user query.
    """
    if isinstance(state.context, dict):
        query = state.context.get("retrospective_query", "")
    else:
        query = getattr(state.context, "retrospective_query", "")

    system_prompt = """
    You are an expert in deducing relevant date ranges from user queries for extracting data from a database in a way that caters for missing readings. 
    Your goal is to identify all relevant date ranges mentioned or implied in the query, following strict rules to avoid common pitfalls.
    
    Current datetime: {current_datetime}
    
    Rules:
    1. Always output date ranges in ISO 8601 format (e.g., 2025-03-21T00:00:00Z).
    2. Always output a date range within which to query data, never a single datetime, because data is highly unlikely to be recorded on the exact single datetime you specify.
    3. Always ensure ranges are before the current datetime; never include future dates.
    4. If a date part is missing (e.g., year), infer it intelligently based on the current datetime, ensuring the range is in the past, because this is most likely to accord with the user's intention.
    5. Account for potential missing readings due to finite frequency, holidays, weather, etc., by including buffers.
    6. For approximate date ranges:
       - Example if current year is 2025: "End of March" -> 2025-03-21T00:00:00Z to 2025-03-31T23:59:59Z (assumes current year if not specified).
       - Example if current year is 2023: "Beginning of the year" -> 2023-01-01T00:00:00Z to 2023-03-31T23:59:59Z.
       - Use symmetric buffers around approximate points (e.g., mid-January: 7 days either side of January 15).
    7. For a specific day (e.g., "on 17 October"):
       - Include a buffer of up to one week before: e.g., 2022-10-10T00:00:00Z to 2022-10-17T23:59:59Z (if the current year is 2022) to account for missing readings.
       - Your explanation should instruct to take the most recent reading within the range as this will be most relevant to the query.
    8. For queries requesting all data within an explicit date range (e.g., "all readings between 10 and 12 October"):
       - Use the exact date range because the user specifically asked for it: 2024-10-10T00:00:00Z to 2024-10-12T23:59:59Z (if the current year is 2024).
       - Your explanation should instruct to take all available readings within the range, as requested by the user.
    9. For queries requesting the change over a period (e.g., "change from 5 to 10 October"):
       - Output two date ranges: one for the start date and one for the end date, each with a buffer of approximately half the period length before the date. This accounts for missing readings around both dates but keeps the ranges distinct.
       - E.g., for "change from 5 to 10 October" (5-day period), use:
         - Start range: 2024-10-02T00:00:00Z to 2024-10-05T23:59:59Z
         - End range: 2024-10-07T00:00:00Z to 2024-10-10T23:59:59Z
       - Your explanation should instruct to take the most recent reading within each range and compute the difference.
    10. For most recent readings (explicit or implicit, e.g., no date mentioned):
       - Use a large historical range: 1900-01-01T00:00:00Z to {current_datetime} to account for the most recent reading being far in the past.
       - Your explanation should instruct to take the latest reading as the most recent.
    11. If multiple date ranges are relevant, output a list with each.
    12. For each range, provide a detailed explanation: what it refers to, why chosen (citing rules), and how to apply (e.g., filter data within range, select latest/closest readings) so that downstream agents fully understand how to use the date range e.g.:
       - "The date range refers to the query implicitly requesting the most recent reading. The extended period of the date range from a very early date to now allows for the most recent reading being at any time. Take the latest reading returned as the most recent reading."
       - "The date range refers to the query requesting reading on 20 Oct. The date range spanning the seven days before the requested date assumes this refers to the current year and that readings could be missing on that date. Take the most recent reading from the requested date."
       - "The date range refers to the query requesting readings in mid January. The date range spans symmetrically across 15 January with 7 days either side to account for missing readings on 15 January itself. Take the three readings closest to 15 January as the readings in mid January."
    13. The user will never ask for dates before year 2000. Check that no date range covers before 2000.

    If no date is mentioned or implied, default to the most recent reading range.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    current_datetime = datetime.datetime.now().isoformat() + "Z"

    chain = prompt | llm.with_structured_output(PeriodExpertOutput)

    result: PeriodExpertOutput = chain.invoke({
        "query": query,
        "current_datetime": current_datetime,
    })

    date_ranges: List[RelevantDateRange] = result.relevant_date_ranges

    context_update = {"relevant_date_ranges": date_ranges}

    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "period_deduced": True,
        },
    )