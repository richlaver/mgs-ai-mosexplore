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
You are an expert in identifying technical instrument IDs from natural language queries about monitoring equipment, sensors, and measurement devices.

Analyze this query: "{query}"

Your task is to:
1. Determine if this query is asking about technical instruments/sensors/monitoring equipment
2. Identify any proper nouns or technical identifiers that could be instrument IDs
3. Pay special attention to compound queries with "and", "or", "&" etc. where multiple instruments are mentioned
4. Provide confidence scores and reasoning

IMPORTANT: Handle compound queries intelligently:
- If a query mentions multiple instruments in list format (e.g., "A, B, and C"), all items are likely instruments
- Numbers, letters, or short codes in instrument context should be considered potential IDs

Examples of compound instrument queries:
- "settlement at 0003-L-1 and 10" → both "0003-L-1" and "10" are instruments
- "reading from PZ001 and S5" → both "PZ001" and "S5" are instruments
- "status of A, B, and C" → all three are likely instruments

Consider these characteristics of instrument IDs:
- They are often proper nouns or technical identifiers
- They may contain alphanumeric combinations (PZ001, ABC-123, Sn, Dn) 
- They can be simple numbers/letters when in instrument context (10, A, B1)
- They appear in contexts related to measurements, readings, monitoring
- They are typically the object of prepositions like "on", "of", "from"
- They are NOT common English words like "reading", "settlement", "latest"

Chain-of-thought:
1. What is the context of the query
2. Are multiple instruments mentioned? If no, skip to step 5.
3. How are conjunctions used to connect instruments?
4. Analyse query semantically and summarise
5. List instrument candidates: where, how confident, why, relationship between

Please respond in this exact JSON format:
{{
    "instrument_candidates": [
        {{
            "text": "identified text",
            "confidence": 0.0-1.0,
            "reasoning": "why you think this is/isn't an instrument ID",
            "context_type": "measurement/monitoring/settlement/status/etc",
            "compound_context": "how this relates to other instruments in the query"
        }}
    ]
}}
"""
        response = llm.invoke(analysis_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        try:
            analysis_result = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group(1))
            else:
                raise ValueError("Could not parse LLM response as JSON")
        candidates = analysis_result.get("instrument_candidates", [])
        final_instruments = [candidate for candidate in candidates if candidate.get("confidence", 0.0) > 0.3]
        return {
            "instruments": sorted(final_instruments, key=lambda x: x["confidence"], reverse=True),
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

def instrument_validator(state: ContextState, llm: BaseLanguageModel, db: any) -> dict:
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
    ids = [inst["text"] for inst in instruments if inst["confidence"] > 0.7]
    ids = [id_ for id_ in ids if id_ not in ignore_IDs]
    valid_instruments = validate_instruments_in_database(db, ids)
    unverif_IDs = [id_ for id_ in ids if id_ not in valid_instruments]

    clarification_request = ""
    if unverif_IDs:
        clarification_request = (
            f"The following instrument IDs could not be found in the database: "
            f"{', '.join(unverif_IDs)}. Please clarify or provide correct IDs."
        )
    context_update = {
        "ignore_IDs": ignore_IDs,
        "verif_ID_info": valid_instruments,
        "unverif_IDs": unverif_IDs,
    }
    
    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "clarification_request": clarification_request,
            "instruments_validated": True,
        }
    )