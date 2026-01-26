import json
import logging
import re
from typing import Any, Dict, List

import setup
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from classes import InstrInfo, DbField, DbSource, QueryWords, ContextState
from langgraph.types import Command
from pydantic import BaseModel, Field
from utils.context_data import get_instrument_context

logger = logging.getLogger(__name__)


def _log_instrument_cache_summary(instrument_data: dict[str, Any]) -> None:
    """Log a concise summary of the instrument context cache contents."""
    if not instrument_data:
        logger.info("Instrument context cache empty")
        return
    types = {}
    for key, entry in instrument_data.items():
        if not isinstance(entry, dict):
            continue
        types.setdefault(entry.get("type", "UNKNOWN"), 0)
        types[entry.get("type", "UNKNOWN")] += 1
    logger.info(
        "Instrument context loaded: %d instruments across %d types (%s)",
        len(instrument_data),
        len(types),
        ", ".join(f"{k}:{v}" for k, v in sorted(types.items())),
    )



class InstrumentSelectionItem(BaseModel):
    """Structured representation of an instrument key and its selected fields."""

    key: str = Field(..., description="Instrument key formatted as TYPE_SUBTYPE")
    database_field_names: List[str] = Field(
        default_factory=list,
        description="Database field names relevant to the query context",
    )


class InstrumentSelectionList(BaseModel):
    """Object-wrapped list for structured LLM output (required by Vertex AI)."""

    items: List[InstrumentSelectionItem] = Field(
        default_factory=list,
        description="Collection of instrument selections relevant to the query",
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)

        def _strip_additional_properties(obj: Any) -> None:
            if isinstance(obj, dict):
                obj.pop("additionalProperties", None)
                for value in obj.values():
                    _strip_additional_properties(value)
            elif isinstance(obj, list):
                for value in obj:
                    _strip_additional_properties(value)

        _strip_additional_properties(schema)
        return schema

    def to_dict_list(self) -> List[dict[str, Any]]:
        return [item.model_dump() for item in self.items]


def identify_and_filter_instruments(llm: BaseLanguageModel, query: str, instrument_data: dict[str, any], verified_type_subtype: list[str]) -> list[dict[str, any]]:
    logger.info(
        "Identify instruments start | query len=%d | verified=%d | instruments=%d",
        len(query),
        len(verified_type_subtype),
        len(instrument_data),
    )
    context = setup.build_instrument_search_context(instrument_data)
    verified_str = ", ".join(verified_type_subtype) if verified_type_subtype else "None"

    system_template = """You are an expert in geotechnical monitoring instruments.
Based on the available instruments (with their detailed field information), identify relevant instrument keys and select ONLY the specific fields that match the query context.

Field Selection Guidelines:
- Analyze the query intent and match it with field metadata (common names, descriptions, units)
- "reading", "value", "measurement" + specific context → select fields matching that context
- "latest", "current" → select the most relevant measurement fields for the instrument type
- For settlement instruments: prioritize settlement calculation fields for settlement queries
- For vibration instruments: include all relevant velocity components (X, Y, Z axes) for vibration queries  
- For groundwater instruments: select level calculation fields for level queries
- For load instruments: select appropriate load measurement fields
- Use field descriptions and common names to make intelligent selections
- Be selective but comprehensive - include all relevant fields for the specific query context

IMPORTANT: Match query semantics with field semantics using the rich metadata provided in the instrument context.

Output MUST be valid JSON list format: [[{{"key": "LP_MOVEMENT", "database_field_names": ["calculation1"]}}, ...]
If no relevant fields for a key, use empty list (will skip). If no additional semantics, only explicit.
If no matches at all, return empty list []."""

    human_template = """Query: {query}

Always include these explicit keys if any: {verified_str}, and for each, select ONLY the field names that are relevant to the specific query context based on field metadata analysis.
Additionally, identify other relevant keys from query semantics, and select their relevant fields.

Available instruments:
{context}

Analyze the query "{query}" and return JSON with instrument keys and their relevant fields only."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

    call_inputs = {
        "query": query,
        "context": context,
        "verified_str": verified_str
    }

    result: list[dict[str, Any]] | None = None
    try:
        cached_context = setup.build_instrument_selection_cached_context(context)
        cached_content_id = setup.ensure_cached_content(
            cache_key="instrument_selection",
            content_text=cached_context,
            llm=llm,
            display_prefix="instrument-selection-cache",
            legacy_hash_keys=["instrument_selection_cached_context_hash"],
        )
        structured_llm = llm.with_structured_output(
            InstrumentSelectionList,
            method="json_mode",
        )

        prompt_text = (
            "Query: {query}\n\n"
            "Always include these explicit keys if any: {verified_str}, and for each, select ONLY the field names "
            "that are relevant to the specific query context based on field metadata analysis. "
            "Additionally, identify other relevant keys from query semantics, and select their relevant fields.\n\n"
            "Return JSON with instrument keys and their relevant fields only."
        ).format(query=query, verified_str=verified_str)

        message = HumanMessage(content=prompt_text)
        if cached_content_id:
            structured_response = structured_llm.invoke([message], cached_content=cached_content_id)
        else:
            structured_response = structured_llm.invoke([message])

        if structured_response is None:
            raise ValueError("Structured output returned no result.")
        result = structured_response.to_dict_list()
        logger.info("Structured instrument selection succeeded with %d items", len(result))
    except Exception as structured_error:
        logger.warning("Structured output failed; falling back to raw parsing: %s", structured_error)

    if result is None:
        chain = prompt | llm
        response = chain.invoke(call_inputs)
        try:
            logger.debug(f"Raw LLM response: {response.content}")
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response.content, re.DOTALL | re.IGNORECASE)
            if json_match:
                result = json.loads(json_match.group(1))
                logger.debug("Successfully extracted JSON from markdown code block")
            else:
                result = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)
            logger.error("Raw response content: %r", response.content)
            if verified_type_subtype:
                logger.info("JSON parsing failed but verified instruments exist, using intelligent field selection fallback")
                default_items = []
                for type_subtype in verified_type_subtype:
                    if type_subtype in instrument_data:
                        relevant_fields = _select_relevant_fields_with_llm(llm, query, type_subtype, instrument_data)
                        default_items.append({
                            "key": type_subtype,
                            "database_field_names": relevant_fields
                        })
                return default_items
            return []
        except Exception as e:
            logger.error("Unexpected error parsing identified instruments: %s", e)
            if verified_type_subtype:
                logger.info("Error occurred but verified instruments exist, using intelligent field selection")
                fallback_items = []
                for type_subtype in verified_type_subtype:
                    if type_subtype in instrument_data:
                        relevant_fields = _select_relevant_fields_with_llm(llm, query, type_subtype, instrument_data)
                        fallback_items.append({
                            "key": type_subtype,
                            "database_field_names": relevant_fields
                        })
                return fallback_items
            return []

    if not isinstance(result, list):
        logger.warning(f"Expected list but got {type(result)}: {result}")
        return []

    valid_items = []
    for item in result:
        if isinstance(item, dict) and 'key' in item and 'database_field_names' in item:
            valid_items.append(item)
        else:
            logger.warning(f"Invalid item structure: {item}")

    if verified_type_subtype and not valid_items:
        logger.info("No LLM results but verified instruments exist, creating default entries with intelligent field selection")
        for type_subtype in verified_type_subtype:
            if type_subtype in instrument_data:
                relevant_fields = _select_relevant_fields_with_llm(llm, query, type_subtype, instrument_data)
                valid_items.append({
                    "key": type_subtype,
                    "database_field_names": relevant_fields
                })

    logger.info("Instrument identification complete with %d valid items", len(valid_items))
    return valid_items

def get_instrument_metadata_from_json(instrument_data: dict[str, any], instr_type: str, instr_subtype: str) -> dict[str, any]:
    key = f"{instr_type}_{instr_subtype}"
    entry = instrument_data.get(key)
    if not entry:
        for k, v in instrument_data.items():
            if isinstance(v, dict) and v.get('type') == instr_type and v.get('subtype') == instr_subtype:
                entry = v
                break
    return entry or {}

def get_fields_from_json(instrument_data: dict[str, any], instr_type: str, instr_subtype: str) -> list[DbField]:
    key = f"{instr_type}_{instr_subtype}"
    entry = instrument_data.get(key)
    if not entry:
        for k, v in instrument_data.items():
            if isinstance(v, dict) and v.get('type') == instr_type:
                entry = v
                break
        if not entry:
            return []
    fields = entry.get('fields', [])
    db_fields = []
    for field in fields:
        if isinstance(field, dict):
            db_fields.append(DbField(
                db_name=field.get('database_field_name', ''),
                db_type=field.get('database_field_type', 'data'),
                labels=field.get('common_names', ''),
                description=field.get('description', ''),
                units=field.get('unit', field.get('units', ''))
            ))
    return db_fields

def extract_matching_words(query: str, target_text: str) -> str:
    query_words = set(query.lower().split())
    target_words = set(target_text.lower().split())
    matching = query_words.intersection(target_words)
    return ' '.join(sorted(matching)) if matching else ' '.join(query.split()[:3])

def _select_relevant_fields_with_llm(llm: BaseLanguageModel, query: str, type_subtype: str, instrument_data: Dict[str, Any]) -> List[str]:
    """
    Use LLM to intelligently select relevant fields based on query context and field metadata
    """
    logger.debug("Selecting fields via LLM | type_subtype=%s | query=%s", type_subtype, query)
    try:
        instrument_info = instrument_data.get(type_subtype, {})
        fields = instrument_info.get('fields', [])
        
        if not fields:
            logger.info("No fields available for %s; returning empty selection", type_subtype)
            return []
        
        field_descriptions = []
        for field in fields:
            field_desc = f"""
Field: {field.get('database_field_name', 'unknown')}
Type: {field.get('database_field_type', 'unknown')}
Common Names: {', '.join(field.get('common_names', []))}
Description: {field.get('description', '')}
Unit: {field.get('unit', 'none')}
"""
            field_descriptions.append(field_desc.strip())
        
        system_prompt = f"""You are an expert in geotechnical monitoring data analysis.

Instrument Type: {type_subtype}
Instrument Purpose: {', '.join(instrument_info.get('purpose', []))}
Data Interpretation: {instrument_info.get('data_interpretation_guidelines', '')}

Available Fields:
{chr(10).join(field_descriptions)}

Your task: Analyze the query and select ONLY the most relevant field names that directly answer the query.

Guidelines:
- For "reading", "value", "latest", "current" queries: Select fields that contain the primary measurement data
- For "settlement", "movement" queries: Select calculation fields for settlement values
- For "level", "depth" queries: Select fields measuring levels or depths  
- For "vibration", "velocity" queries: Select velocity/acceleration fields
- For "load", "force" queries: Select force/load measurement fields
- Prioritize calculation fields over raw data fields when both are relevant
- Be selective - typically 1-3 fields maximum unless query specifically asks for multiple measurements

Return ONLY a JSON array of field names: ["field1", "field2"]
If no fields are relevant, return: []"""

        human_prompt = f'Query: "{query}"\n\nSelect the most relevant field name(s) for this query.'
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        chain = prompt | llm
        response = chain.invoke({})
        
        # Parse LLM response
        try:
            # First, try to extract JSON from markdown code blocks (most common format)
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response.content, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                # Found JSON in markdown code block, try to parse it
                try:
                    selected_fields = json.loads(json_match.group(1))
                    logger.debug("Successfully extracted JSON from markdown code block in field selection")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from markdown block: {e}")
                    # Fall back to direct parsing
                    selected_fields = json.loads(response.content)
            else:
                # No markdown code block found, try direct JSON parsing
                selected_fields = json.loads(response.content)
            
            if isinstance(selected_fields, list):
                # Validate that selected fields exist in the instrument data
                valid_field_names = [f.get('database_field_name') for f in fields]
                chosen = [field for field in selected_fields if field in valid_field_names]
                logger.debug("LLM field selection for %s produced %d valid entries", type_subtype, len(chosen))
                return chosen
            else:
                logger.warning(f"LLM returned non-list response: {selected_fields}")
                return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM field selection response: {response.content}")
            # Try to extract field names from response text as secondary fallback
            response_text = response.content.lower()
            extracted_fields = []
            valid_field_names = [f.get('database_field_name') for f in fields]
            for field_name in valid_field_names:
                if field_name and field_name.lower() in response_text:
                    extracted_fields.append(field_name)
            
            if extracted_fields:
                logger.info(f"Extracted fields from LLM response text: {extracted_fields}")
                return extracted_fields[:3]  # Limit to 3 fields
            
            # If text extraction fails, use fallback
            return _fallback_field_selection(query, fields)
            
    except Exception as e:
        logger.error("Error in LLM field selection for %s: %s", type_subtype, e)
        # Fallback to basic heuristic
        return _fallback_field_selection(query, fields)

def _fallback_field_selection(query: str, fields: List[Dict[str, Any]]) -> List[str]:
    """
    Fallback field selection using simplified LLM approach when main LLM fails
    Uses a simpler prompt that's more resilient to LLM connection issues
    """
    logger.info("Fallback field selection triggered | fields=%d", len(fields))
    try:
        # Try a simplified LLM approach first
        return _simple_llm_field_selection(query, fields)
    except Exception as e:
        logger.warning(f"Simple LLM fallback also failed: {e}, using metadata matching")
        # Ultimate fallback: pure metadata matching
        return _metadata_based_selection(query, fields)

def _simple_llm_field_selection(query: str, fields: List[Dict[str, Any]]) -> List[str]:
    """
    Simplified LLM-based field selection for fallback scenarios
    Uses basic string matching with LLM reasoning instead of complex prompts
    """
    if not fields:
        return []
    
    # Create simplified field summaries
    field_summaries = []
    for field in fields:
        summary = f"{field.get('database_field_name', '')}: {', '.join(field.get('common_names', []))}"
        if field.get('description'):
            summary += f" - {field.get('description')}"
        field_summaries.append(summary)
    
    # Super simple prompt for resilience
    simple_prompt = f"""Query: "{query}"
    
Available fields:
{chr(10).join(field_summaries)}

Select the most relevant field name(s) that answer the query. 
Return only field names separated by commas, no explanation needed.
Example: calculation1, data2"""
    
    # This would use a simpler LLM call or local reasoning
    # For now, fall back to metadata matching
    raise Exception("Simple LLM not available")

def _metadata_based_selection(query: str, fields: List[Dict[str, Any]]) -> List[str]:
    """
    Pure metadata-based field selection as ultimate fallback
    Uses semantic matching of query words with field metadata
    """
    logger.debug("Metadata-based selection | query='%s' | fields=%d", query, len(fields))
    query_words = set(query.lower().split())
    
    # Score each field based on metadata matches
    field_scores = []
    for field in fields:
        score = 0
        field_name = field.get('database_field_name', '')
        
        # Check common names for semantic matches
        common_names = field.get('common_names', [])
        for name in common_names:
            name_words = set(name.lower().split())
            # Score based on word overlap
            overlap = len(query_words.intersection(name_words))
            if overlap > 0:
                score += overlap * 5  # High weight for common name matches
        
        # Check description for semantic matches
        description = field.get('description', '').lower()
        desc_words = set(description.split())
        overlap = len(query_words.intersection(desc_words))
        if overlap > 0:
            score += overlap * 2  # Medium weight for description matches
        
        # Check for exact or partial word matches in field name
        field_name_words = set(field_name.lower().split('_'))
        overlap = len(query_words.intersection(field_name_words))
        if overlap > 0:
            score += overlap * 1  # Lower weight for field name matches
        
        # Prefer calculation fields for general queries only if no specific matches
        if field.get('database_field_type') == 'calc':
            score += 1
        
        # Check for general query indicators - prefer calc fields
        general_indicators = ['reading', 'latest', 'current', 'value', 'measurement']
        if any(indicator in query.lower() for indicator in general_indicators):
            if field.get('database_field_type') == 'calc':
                score += 2  # Prefer calc fields for general queries
            # But reduce score if it's just a general query without specific semantic match
            if score <= 3:  # Only general + calc bonus, no semantic matches
                score = max(score - 1, 1)  # Reduce but keep some score
        
        field_scores.append((field_name, score))
    
    # Sort by score and return top field(s)
    field_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return fields with highest scores (at least score > 0)
    # Be more selective - only return fields with significant scores
    if field_scores and field_scores[0][1] > 2:  # At least 3 points (semantic match)
        # Return only the top-scoring field(s) with similar scores
        top_score = field_scores[0][1]
        selected_fields = [field_name for field_name, score in field_scores 
                          if score >= top_score and score > 2]
        logger.debug("Metadata selection strong match; returning %s", selected_fields[:2])
        return selected_fields[:2]  # Limit to top 2 fields
    
    # If no strong semantic matches, be more conservative
    selected_fields = [field_name for field_name, score in field_scores if score > 0]
    
    # For general queries with weak matches, prefer single calc field
    general_indicators = ['reading', 'latest', 'current', 'value', 'measurement', 'data']
    if any(indicator in query.lower() for indicator in general_indicators) and not selected_fields:
        calc_fields = [f.get('database_field_name') for f in fields 
                       if f.get('database_field_type') == 'calc' and f.get('database_field_name')]
        return calc_fields[:1] if calc_fields else []
    
    # If no semantic matches, return first calc field or first field
    if not selected_fields:
        calc_fields = [f.get('database_field_name') for f in fields 
                       if f.get('database_field_type') == 'calc' and f.get('database_field_name')]
        if calc_fields:
            return calc_fields[:1]
        else:
            all_fields = [f.get('database_field_name') for f in fields if f.get('database_field_name')]
            return all_fields[:1] if all_fields else []
    
    # For specific queries with good matches, return top field only
    if field_scores and field_scores[0][1] >= 6:  # Strong semantic match
        return [field_scores[0][0]]
    
    # Return top 1-2 fields based on score
    return selected_fields[:2]

def database_expert(state: ContextState, llm: BaseLanguageModel, db: any, selected_project_key: str | None = None) -> dict:
    """Database expert agent: Retrieves and updates word_context if no clarification needed.
    Returns a Command to update state in the graph.
    """
    logger.info("Database expert invoked")
    try:
        clar = []
        if isinstance(state.context, dict):
            clar = state.context.get("clarification_requests") or []
        else:
            clar = getattr(state.context, "clarification_requests", []) or []
        if clar:
            logger.info("Clarifications present; deferring to supervisor")
            return Command(
                goto="supervisor",
                update={
                    "db_context_provided": False,
                }
            )
    except Exception:
        logger.exception("Clarification check failed; continuing")
        pass

    logger.debug("Database expert using project key: %s", selected_project_key or "<session_default>")
    raw_instrument_data = get_instrument_context(selected_project_key)
    instrument_data = setup.normalize_instrument_context(raw_instrument_data)
    _log_instrument_cache_summary(instrument_data)
    if isinstance(state.context, dict):
        query = state.context.get("retrospective_query", "")
        verif_id_info = state.context.get("verif_ID_info") or {}
        verif_type_info = state.context.get("verif_type_info") or []
    else:
        query = getattr(state.context, "retrospective_query", "")
        verif_id_info = getattr(state.context, "verif_ID_info", {}) or {}
        verif_type_info = getattr(state.context, "verif_type_info", []) or []
    logger.debug("Query extracted: %s", query)
    query_words_list: list[QueryWords] = []
    type_subtype_groups: dict[tuple, list[str]] = {}
    verified_type_subtype: list[str] = []
    verified_type_subtype_keys: set[str] = set()

    def _add_verified_type(instr_type: str | None, instr_subtype: str | None) -> None:
        if not instr_type or not instr_subtype:
            return
        key = f"{instr_type}_{instr_subtype}"
        if key not in verified_type_subtype_keys:
            verified_type_subtype_keys.add(key)
            verified_type_subtype.append(key)

    if verif_id_info:
        for instr_id, info in verif_id_info.items():
            db_type = info["type"]
            db_subtype = info["subtype"]
            key_tuple = (db_type, db_subtype)
            if key_tuple not in type_subtype_groups:
                type_subtype_groups[key_tuple] = []
            type_subtype_groups[key_tuple].append(instr_id)
            _add_verified_type(db_type, db_subtype)
    logger.info("Verified IDs processed: %d groups", len(type_subtype_groups))

    if verif_type_info:
        for type_entry in verif_type_info:
            if not isinstance(type_entry, dict):
                continue
            instr_type = type_entry.get("type")
            subtypes = type_entry.get("subtypes") or []
            for subtype in subtypes:
                _add_verified_type(instr_type, subtype)
    logger.info("Verified type/subtype pairs collected: %d", len(verified_type_subtype))
    semantic_filtered = identify_and_filter_instruments(llm, query, instrument_data, verified_type_subtype)
    verified_type_subtype_set = verified_type_subtype_keys
    for item in semantic_filtered:
        key = item.get('key')
        selected_fields = item.get('database_field_names', [])
        if key:
            try:
                instr_type, instr_subtype = key.split('_')
            except ValueError:
                continue
            entry = get_instrument_metadata_from_json(instrument_data, instr_type, instr_subtype)
            if entry:
                instr_info = InstrInfo(
                    labels=entry.get('name', []) if isinstance(entry.get('name'), list) else [entry.get('name', '')],
                    physical_form=entry.get('form', ''),
                    raw_instr_output=entry.get('raw_instrument_output', []) if isinstance(entry.get('raw_instrument_output'), list) else entry.get('raw_instrument_output', '').split(', '),
                    derived_measurands=entry.get('derived_measurands', []) if isinstance(entry.get('derived_measurands'), list) else entry.get('derived_measurands', '').split(', '),
                    purposes=entry.get('purpose', []) if isinstance(entry.get('purpose'), list) else [entry.get('purpose', '')],
                    data_interpretation=entry.get('data_interpretation_guidelines', ''),
                    typical_plots=entry.get('typical_plots', []) if isinstance(entry.get('typical_plots'), list) else entry.get('typical_plots', '').split(', ')
                )
                db_fields_full = get_fields_from_json(instrument_data, instr_type, instr_subtype)
                db_fields = [f for f in db_fields_full if f.db_name in selected_fields] if selected_fields else []
                if db_fields:
                    db_sources = DbSource(
                        instr_type=instr_type,
                        instr_subtype=instr_subtype,
                        instr_info=instr_info,
                        db_fields=db_fields
                    )
                    ids = type_subtype_groups.get((instr_type, instr_subtype), [])
                    query_words_str = f"Instruments: {', '.join(ids)}" if key in verified_type_subtype_set else extract_matching_words(query,f"""
{entry.get('name', '')} 
{entry.get('raw_instrument_output', '')} 
{entry.get('derived_measurands', '')} 
{entry.get('purpose', '')}""")
                    query_words_list.append(QueryWords(
                        query_words=query_words_str,
                        data_sources=[db_sources]
                    ))
                else:
                    logger.info("No db_fields selected for %s; skipping QueryWords", key)
            else:
                logger.info("No instrument metadata found for key %s", key)
    context_update = {
        "word_context": query_words_list
    }
    logger.info("Database expert completed with %d word_context entries", len(query_words_list))

    return Command(
        goto="supervisor",
        update={
            "context": context_update,
            "db_context_provided": True,
        }
    )