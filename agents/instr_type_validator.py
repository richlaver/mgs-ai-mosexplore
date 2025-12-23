import json
import logging
import ast
import re

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from classes import ContextState

logger = logging.getLogger(__name__)

def extract_instruments_with_llm(llm: BaseLanguageModel, query: str) -> dict:
    try:
        analysis_prompt = f"""
# Role
You are an expert geotechnical instrumentation engineer with rich experience working on a wide variety of construction sites. You are adept at differentiating words which are instrument names from those which are not.

# Task
Identify instrument names in a query requesting construction site instrumentation data from a database.

# Tips on Identifying Instrument Names
- They are always proper nouns
- Likely to have a concise or abbreviated prefix identifying the instrument type
- Likely to be suffixed by some alphanumeric characters to differentiate between instruments of the same type
- May contain concise or abbreviated alphanumeric characters differentiating other metadata of the instrument e.g. zone, contract
- Can contain hyphens or slashes

# Steps
1. Analyse this query: {query}
2. Identify any words which could be proper nouns
3. For each word, assess the probability that it is an instrument name based on your own experience and above tips

# Output
Respond in this exact JSON format:
{{
    "instrument_names": [
        {{
            "text": "identified name",
            "probability": 0.0-1.0
        }}
    ]
}}
        """
        response = llm.invoke(analysis_prompt)
        logger.info(f"instrument validator LLM response: {response}")
        response_text = response.content if hasattr(response, 'content') else str(response)
        try:
            analysis_result = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group(1))
            else:
                raise ValueError("Could not parse LLM response as JSON")
        candidates = analysis_result.get("instrument_names", [])
        final_instruments = [candidate for candidate in candidates if candidate.get("probability", 0.0) > 0.3]
        return {
            "instruments": sorted(final_instruments, key=lambda x: x["probability"], reverse=True),
            "error": None
        }
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        return {"instruments": [], "error": str(e)}

def validate_instruments_in_database(db: any, instrument_ids: list[str]) -> dict[str, dict[str, str]]:
    valid_instruments = {}
    for instrument_id in instrument_ids:
        query = f"""
        SELECT type1, subtype1
        FROM instrum 
        WHERE instr_id = '{instrument_id}' 
        LIMIT 1
        """
        try:
            result = db.run_no_throw(query)
            logger.info(f"Validation query result for {instrument_id}: {result}")
            if result and isinstance(result, str) and result.strip():
                parsed_result = ast.literal_eval(result)
                if parsed_result and len(parsed_result) > 0:
                    type1, subtype1 = parsed_result[0]
                    if type1 and type1.strip():
                        valid_instruments[instrument_id] = {
                            "type": type1.strip(),
                            "subtype": subtype1.strip() if subtype1 else "DEFAULT"
                        }
        except Exception as e:
            logger.error(f"Error validating {instrument_id}: {e}")
    return valid_instruments

def instr_type_validator(state: ContextState, llm: BaseLanguageModel, db: any) -> dict:
    """Instrument validator agent: Excludes and identifies/validates instrument IDs, updates context.
    Returns a Command to update state in the graph.
    """

    retro_query = ""
    if isinstance(state.context, dict):
        retro_query = state.context.get("retrospective_query", "")
    elif hasattr(state.context, "retrospective_query"):
        retro_query = getattr(state.context, "retrospective_query") or ""

    # Exclusion of instrument IDs has been disabled to reduce latency.
    # To include again, uncomment the code below and remove the line setting ignore_IDs to [].
#     exclude_prompt_messages = [
#         ("system", """You are a precise instrument ID analyzer.
# Your task is to identify terms that look like instrument IDs but have been explicitly mentioned as NOT being instrument IDs.

# Analyze the query and return your findings in the following JSON format:
# {{
#     "excluded_ids": [
#         {{
#             "id": "string",            # The ID to exclude
#             "reason": "string",        # Why this ID should be excluded
#             "confidence": float,       # Confidence score 0-1
#             "source_text": "string"    # The text that indicates this should be excluded
#         }}
#     ],
#     "analysis": {{
#         "total_found": int,           # Total number of exclusions found
#         "reasoning": "string"         # Brief explanation of the analysis
#     }}
# }}

# Only include IDs that are explicitly mentioned as NOT being instrument IDs.
# If no exclusions are found, return an empty list for excluded_ids.
# """),
#     ("human", f"""
# Analyze this query for terms that should NOT be treated as instrument IDs:
# {retro_query}
# """)
#     ]
#     exclude_prompt = ChatPromptTemplate.from_messages(exclude_prompt_messages)
#     exclude_chain = exclude_prompt | llm
#     exclude_response = exclude_chain.invoke({})
#     try:
#         response_data = json.loads(exclude_response.content)
#         CONFIDENCE_THRESHOLD = 0.7
#         ignore_IDs = [
#             item["id"] for item in response_data.get("excluded_ids", [])
#             if item.get("confidence", 0) >= CONFIDENCE_THRESHOLD
#         ]
#     except json.JSONDecodeError:
#         ignore_IDs = []
    ignore_IDs = []

    result = extract_instruments_with_llm(llm, retro_query)
    instruments = result.get("instruments", [])
    ids = [inst["text"] for inst in instruments if inst["probability"] > 0.7]
    ids = [id_ for id_ in ids if id_ not in ignore_IDs]
    valid_instruments = validate_instruments_in_database(db, ids)
    unverif_IDs = [id_ for id_ in ids if id_ not in valid_instruments]

    clarification_request = ""
    # Skip clarification request to allow types and subtypes that are incorrectly identified as instrument IDs to pass downstream.
    # if unverif_IDs:
    #     clarification_request = (
    #         f"The following instrument IDs could not be found in the database: "
    #         f"{', '.join(unverif_IDs)}. Please clarify or provide correct IDs."
    #     )
    context_update = {
        "ignore_IDs": ignore_IDs,
        "verif_ID_info": valid_instruments,
        "unverif_IDs": unverif_IDs,
    }
    
    if clarification_request:
        context_update["clarification_requests"] = [clarification_request]

    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "instruments_validated": True,
        }
    )