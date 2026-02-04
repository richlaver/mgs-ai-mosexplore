import datetime
import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langgraph.types import Command
from pydantic import BaseModel, Field

from classes import RelevantDateRange, ContextState
from utils.json_utils import strip_to_json_payload


class PeriodExpertOutput(BaseModel):
    """Structured output containing relevant date ranges."""
    relevant_date_ranges: List[RelevantDateRange] = Field(
        description="List of all relevant date ranges mentioned or implied in the query"
    )
    timezone_interpretation: str = Field(
        description="Explanation of whether times in the query are in user or project timezone and the assumption applied"
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
    6. Determine whether the query times are explicitly in the user timezone or the project timezone. If the query does not specify, assume times are in the project timezone. Record this interpretation in the returned context so downstream agents know which timezone assumption was applied.
    7. For approximate date ranges:
       - Example if current year is 2025: "End of March" -> 2025-03-21T00:00:00Z to 2025-03-31T23:59:59Z (assumes current year if not specified).
       - Example if current year is 2023: "Beginning of the year" -> 2023-01-01T00:00:00Z to 2023-03-31T23:59:59Z.
       - Use symmetric buffers around approximate points (e.g., mid-January: 7 days either side of January 15).
    8. For a specific day (e.g., "on 17 October"):
       - Include a buffer of up to one week before: e.g., 2022-10-10T00:00:00Z to 2022-10-17T23:59:59Z (if the current year is 2022) to account for missing readings.
       - Your explanation should instruct to take the most recent reading within the range as this will be most relevant to the query.
    9. For queries requesting all data within an explicit date range (e.g., "all readings between 10 and 12 October"):
       - Use the exact date range because the user specifically asked for it: 2024-10-10T00:00:00Z to 2024-10-12T23:59:59Z (if the current year is 2024).
       - Your explanation should instruct to take all available readings within the range, as requested by the user.
    10. For queries requesting the change over a period (explicit or implicit, e.g., "change from 5 to 10 October" or "Where has settlement changed by more than 3mm over the past week?"):
            - Output two date ranges: one for the start date and one for the end date. Apply a buffer before each date that is no more than half of the period length; shorten the buffer if needed to avoid overlap or future dates. This caters for missing readings while keeping the ranges distinct and bounded.
            - E.g., for "change from 5 to 10 October" (5-day period), use:
                - Start range: 2024-10-02T00:00:00Z to 2024-10-05T23:59:59Z
                - End range: 2024-10-07T00:00:00Z to 2024-10-10T23:59:59Z
            - For relative periods (e.g., "over the past week"), infer the start date as the end date minus the period length, then apply the same capped-buffer rule.
            - Your explanation should instruct to take the most recent reading within each range and compute the difference.
            - If you are applying a buffer to the current date allow for the fact that data typically takes at least two days from when it is taken to appear in the database.
    11. For most recent readings (explicit or implicit, e.g., no date mentioned):
       - Use a large historical range: 1900-01-01T00:00:00Z to {current_datetime} to account for the most recent reading being far in the past.
       - Your explanation should instruct to take the latest reading as the most recent.
    12. If multiple date ranges are relevant, output a list with each.
    13. For each range, provide a detailed explanation: what it refers to, why chosen (citing rules), and how to apply (e.g., filter data within range, select latest/closest readings) so that downstream agents fully understand how to use the date range e.g.:
       - "The date range refers to the query implicitly requesting the most recent reading. The extended period of the date range from a very early date to now allows for the most recent reading being at any time. Take the latest reading returned as the most recent reading."
       - "The date range refers to the query requesting reading on 20 Oct. The date range spanning the seven days before the requested date assumes this refers to the current year and that readings could be missing on that date. Take the most recent reading from the requested date."
       - "The date range refers to the query requesting readings in mid January. The date range spans symmetrically across 15 January with 7 days either side to account for missing readings on 15 January itself. Take the three readings closest to 15 January as the readings in mid January."
    14. The user will never ask for dates before year 2000. Check that no date range covers before 2000.

    If no date is mentioned or implied, default to the most recent reading range.

    Output format (strict): Return a SINGLE JSON object with exactly these keys:
    {{
        "relevant_date_ranges": [
            {{
                "start_datetime": "<ISO-8601 start>",
                "end_datetime": "<ISO-8601 end>",
                "explanation": "<how to use this range>"
            }}
        ],
        "timezone_interpretation": "<short explanation of timezone assumption>"
    }}
    Do not return a bare list. Do not include any other top-level keys.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    current_datetime = datetime.datetime.now().isoformat() + "Z"

    result: PeriodExpertOutput | None = None
    try:
        chain = prompt | llm.with_structured_output(PeriodExpertOutput)
        result = chain.invoke({
            "query": query,
            "current_datetime": current_datetime,
        })
    except Exception:
        result = None

    if result is None:
        response = llm.invoke(
            prompt.format_prompt(query=query, current_datetime=current_datetime).to_messages()
        )
        raw_text = getattr(response, "content", "") or str(response)
        raw_text = strip_to_json_payload(
            raw_text,
            [
                '"relevant_date_ranges"',
                '"timezone_interpretation"',
            ],
        )
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            parsed = {
                "relevant_date_ranges": parsed,
                "timezone_interpretation": "",
            }
        elif isinstance(parsed, dict):
            parsed.setdefault("relevant_date_ranges", [])
            parsed.setdefault("timezone_interpretation", "")

        if isinstance(parsed, dict) and isinstance(parsed.get("relevant_date_ranges"), list):
            normalized_ranges = []
            for item in parsed["relevant_date_ranges"]:
                if isinstance(item, dict) and "date_range" in item:
                    raw_range = str(item.get("date_range") or "")
                    if " to " in raw_range:
                        start_dt, end_dt = raw_range.split(" to ", 1)
                    else:
                        start_dt, end_dt = raw_range, ""
                    normalized = {
                        "start_datetime": start_dt.strip(),
                        "end_datetime": end_dt.strip(),
                        "explanation": item.get("explanation") or "",
                    }
                    normalized_ranges.append(normalized)
                else:
                    normalized_ranges.append(item)
            parsed["relevant_date_ranges"] = normalized_ranges

        result = PeriodExpertOutput.model_validate(parsed)

    if result is None:
        raise ValueError("Period expert returned no result")

    date_ranges: List[RelevantDateRange] = result.relevant_date_ranges

    existing_tz_context = {}
    if isinstance(state.context, dict):
        existing_tz_context = state.context.get("timezone_context") or {}
    else:
        existing_tz_context = getattr(state.context, "timezone_context", {}) or {}

    merged_tz_context = dict(existing_tz_context)
    if result.timezone_interpretation:
        merged_tz_context["query_timezone_interpretation"] = result.timezone_interpretation

    context_update = {
        "relevant_date_ranges": date_ranges,
    }
    if merged_tz_context:
        context_update["timezone_context"] = merged_tz_context

    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "period_deduced": True,
        },
    )