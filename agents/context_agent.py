import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

from classes import Context, InstrInfo, DbField, DbSource, QueryWords
import re
import logging
import os
import ast

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_instruments_with_llm(llm: BaseLanguageModel, query: str) -> Dict:
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
- Numbers, letters, or short codes in instrument contexts should be considered potential IDs

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

def load_instrument_context(file_path: str = "instrument_context.json") -> Dict[str, Any]:
    """
    Load instrument context
    
    Args:
        file_path: Path to instrument context JSON file
        
    Returns:
        Dict containing instrument context data
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Instrument context file {file_path} not found")
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
            
        logger.info(f"Loaded instrument context: {len([k for k in context_data.keys() if not k.startswith('_')])} instruments")
        return context_data
        
    except Exception as e:
        logger.error(f"Error loading instrument context: {e}")
        return {}

def create_instrument_search_context(instrument_data: Dict[str, Any]) -> str:
    context_parts = []
    if isinstance(instrument_data, dict):
        context_parts.append("AVAILABLE INSTRUMENTS:")
        type_groups = {}
        for instrument_key, instrument_info in instrument_data.items():
            if isinstance(instrument_info, dict):
                instrument_type = instrument_info.get('type', 'UNKNOWN')
                if instrument_type not in type_groups:
                    type_groups[instrument_type] = []
                type_groups[instrument_type].append((instrument_key, instrument_info))
        for instrument_type, instruments in sorted(type_groups.items()):
            context_parts.append(f"\n{instrument_type} INSTRUMENTS:")
            for instrument_key, instrument_info in instruments:
                name = instrument_info.get('name', 'Unknown')
                subtype = instrument_info.get('subtype', 'DEFAULT')
                purpose_snippet = instrument_info.get('purpose', '')[:250] + "..." if len(instrument_info.get('purpose', '')) > 250 else instrument_info.get('purpose', '')
                fields = instrument_info.get('fields', [])
                fields_to_show = fields if len(fields) < 10 else fields[:10]
                fields_snippet = ", ".join(
                    f"{f.get('database_field_name', '')} ({f.get('unit', f.get('units', ''))}): "
                    f"{', '.join(f.get('common_names', []))} - "
                    f"{f.get('description', '') + ('...' if len(f.get('description', '')) > 200 else '')}"
                    for f in fields_to_show
                )
                context_parts.append(f"  - {instrument_key} ({name} - {subtype}): {purpose_snippet}\n    Fields: {fields_snippet}")
    return "\n".join(context_parts)

def identify_and_filter_instruments(llm: BaseLanguageModel, query: str, instrument_data: Dict[str, Any], verified_type_subtype: List[str]) -> List[Dict[str, Any]]:
    context = create_instrument_search_context(instrument_data)
    verified_str = ", ".join(verified_type_subtype) if verified_type_subtype else "None"

    system_template = """You are an expert in geotechnical monitoring instruments.
Based on the query and available instruments (with their detailed field information), identify relevant instrument keys and select ONLY the specific fields that match the query context.

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

Always include these explicit keys if any: {verified_str}, and for each, select ONLY the field names that are relevant to the specific query context based on field metadata analysis.
Additionally, identify other relevant keys from query semantics, and select their relevant fields.

Output MUST be valid JSON list format: [[{{"key": "LP_MOVEMENT", "fields": ["calculation1"]}}, ...]
If no relevant fields for a key, use empty list (will skip). If no additional semantics, only explicit.
If no matches at all, return empty list []."""

    human_template = """Query: {query}

Available instruments:
{context}

Analyze the query "{query}" and return JSON with instrument keys and their relevant fields only."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

    chain = prompt | llm
    response = chain.invoke({
        "query": query,
        "context": context,
        "verified_str": verified_str
    })

    try:
        # Log the raw response for debugging
        logger.debug(f"Raw LLM response: {response.content}")
        
        # First, try to extract JSON from markdown code blocks (most common format)
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response.content, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            # Found JSON in markdown code block, try to parse it
            try:
                result = json.loads(json_match.group(1))
                logger.debug("Successfully extracted JSON from markdown code block")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from markdown block: {e}")
                # Fall back to direct parsing
                result = json.loads(response.content)
        else:
            # No markdown code block found, try direct JSON parsing
            result = json.loads(response.content)
        
        # Validate the result structure
        if not isinstance(result, list):
            logger.warning(f"Expected list but got {type(result)}: {result}")
            return []
        
        # Validate the structure of each item
        valid_items = []
        for item in result:
            if isinstance(item, dict) and 'key' in item and 'fields' in item:
                valid_items.append(item)
            else:
                logger.warning(f"Invalid item structure: {item}")
        
        # If we have verified type/subtype but no LLM results, create default entries
        if verified_type_subtype and not valid_items:
            logger.info("No LLM results but verified instruments exist, creating default entries with intelligent field selection")
            for type_subtype in verified_type_subtype:
                if type_subtype in instrument_data:
                    # Use LLM to intelligently select fields based on query context and field metadata
                    relevant_fields = _select_relevant_fields_with_llm(llm, query, type_subtype, instrument_data)
                    valid_items.append({
                        "key": type_subtype,
                        "fields": relevant_fields
                    })
        
        logger.debug(f"Successfully parsed {len(valid_items)} instrument items")
        return valid_items
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Raw response content: {repr(response.content)}")
        
        # If JSON parsing fails and we have verified instruments, create default entries
        if verified_type_subtype:
            logger.info("JSON parsing failed but verified instruments exist, using intelligent field selection fallback")
            default_items = []
            for type_subtype in verified_type_subtype:
                if type_subtype in instrument_data:
                    relevant_fields = _select_relevant_fields_with_llm(llm, query, type_subtype, instrument_data)
                    default_items.append({
                        "key": type_subtype,
                        "fields": relevant_fields
                    })
            return default_items
        
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing identified instruments: {e}")
        
        # Fallback for verified instruments
        if verified_type_subtype:
            logger.info("Error occurred but verified instruments exist, using intelligent field selection")
            fallback_items = []
            for type_subtype in verified_type_subtype:
                if type_subtype in instrument_data:
                    relevant_fields = _select_relevant_fields_with_llm(llm, query, type_subtype, instrument_data)
                    fallback_items.append({
                        "key": type_subtype,
                        "fields": relevant_fields
                    })
            return fallback_items
        
        return []

def validate_instruments_in_database(db: Any, instrument_ids: List[str]) -> Dict[str, Dict[str, str]]:
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

def get_instrument_metadata_from_json(instrument_data: Dict[str, Any], instr_type: str, instr_subtype: str) -> Dict[str, Any]:
    key = f"{instr_type}_{instr_subtype}"
    entry = instrument_data.get(key, {})
    if not entry:
        for k, v in instrument_data.items():
            if isinstance(v, dict) and v.get('type') == instr_type and v.get('subtype') == instr_subtype:
                entry = v
                break
    return entry

def get_fields_from_json(instrument_data: Dict[str, Any], instr_type: str, instr_subtype: str) -> List[DbField]:
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
                labels=[field.get('database_field_name', '')],
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
    try:
        instrument_info = instrument_data.get(type_subtype, {})
        fields = instrument_info.get('fields', [])
        
        if not fields:
            return []
        
        # Create detailed field descriptions for LLM analysis
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
        
        # LLM prompt for intelligent field selection
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
                return [field for field in selected_fields if field in valid_field_names]
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
        logger.error(f"Error in LLM field selection: {e}")
        # Fallback to basic heuristic
        return _fallback_field_selection(query, fields)

def _fallback_field_selection(query: str, fields: List[Dict[str, Any]]) -> List[str]:
    """
    Fallback field selection using simplified LLM approach when main LLM fails
    Uses a simpler prompt that's more resilient to LLM connection issues
    """
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

def _create_enhanced_fallback_query(original_query: str, word_context_str: str, verified_ids: List[str]) -> str:
    """
    Create an enhanced contextual query when LLM JSON parsing fails
    This ensures we still get database context even with LLM response issues
    """
    enhanced_parts = [original_query]
    
    # Add database table context
    enhanced_parts.append("Query the instrum table to find relevant instruments")
    
    # Add verified instruments context
    if verified_ids:
        enhanced_parts.append(f"specifically focusing on instruments: {', '.join(verified_ids)}")
    
    # Add field context from word context
    if "calculation1" in word_context_str:
        enhanced_parts.append("retrieve calculation1 field values from the data table")
    elif "settlement" in original_query.lower():
        enhanced_parts.append("retrieve settlement calculation values from the data table")
    elif "vibration" in original_query.lower():
        enhanced_parts.append("retrieve vibration measurement values from the data table")
    else:
        enhanced_parts.append("retrieve relevant measurement values from the data table")
    
    # Add temporal context if present
    if any(word in original_query.lower() for word in ['march', '2025', 'latest', 'recent']):
        if 'march 2025' in original_query.lower():
            enhanced_parts.append("for the March 2025 time period (2025-03-01 to 2025-03-31)")
        elif 'latest' in original_query.lower() or 'recent' in original_query.lower():
            enhanced_parts.append("for the most recent available data")
    
    # Add spatial context if present
    if any(word in original_query.lower() for word in ['around', 'near', 'nearby']):
        enhanced_parts.append("including nearby instruments in the spatial area")
    
    return ". ".join(enhanced_parts) + "."

@dataclass
class ContextAgentState:
    chat_history: List[BaseMessage]
    context: Context
    clarification_request: str
    messages: List[BaseMessage]

def create_context_subgraph(llm: BaseLanguageModel, db: Any) -> StateGraph:
    def query_history_combiner(state: ContextAgentState) -> ContextAgentState:
        state.messages.append(AIMessage(
            name="QueryHistoryCombiner",
            content="Reviewing your chat history to better understand your query...",
            additional_kwargs={
                "stage": "node",
                "process": "query_enricher"
            }
        ))
        current_query = next((msg.content for msg in reversed(state.chat_history) 
                              if isinstance(msg, HumanMessage)), None)
        if not current_query:
            raise ValueError("No query found in chat history")
        
        # Check if there's meaningful chat history (more than just the current query)
        previous_messages = [msg for msg in state.chat_history[:-1] if isinstance(msg, (HumanMessage, AIMessage))]
        
        if len(previous_messages) == 0:
            # No previous context, just use the current query
            state.context.retrospective_query = current_query
            return state
        
        messages = [
            ("system", """
You are a context-aware assistant. 
Review the chat history and current query, 
then rewrite the current query to include relevant context from previous messages. 
Focus on maintaining critical details and relationships from the conversation that would help answer the current query accurately.
If there is no relevant context from previous messages, just return the current query as-is.
             """),
            ("human", f"""
Chat history (newest last):
{chr(10).join(f"{'Human: ' if isinstance(msg, HumanMessage) else 'Assistant: ' if isinstance(msg, AIMessage) else ''}{msg.content}" 
            for msg in previous_messages)}

Current query: {current_query}

Rewrite the query incorporating relevant context from the chat history, or return the query as-is if no relevant context exists.
            """)
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        try:
            retrospective_query = chain.invoke({})
            state.context.retrospective_query = retrospective_query.content
        except Exception as e:
            logger.error(f"Error generating retrospective query: {e}")
            # Fallback to using the original query if LLM fails
            state.context.retrospective_query = current_query
        return state

    def instr_id_excluder(state: ContextAgentState) -> ContextAgentState:
        messages = [
            ("system", """You are a precise instrument ID analyzer.
Your task is to identify terms that look like instrument IDs but have been explicitly mentioned as NOT being instrument IDs.

Analyze the query and return your findings in the following JSON format:
{{
    "excluded_ids": [
        {{
            "id": "string",            # The ID to exclude
            "reason": "string",        # Why this ID should be excluded
            "confidence": float,       # Confidence score 0-1
            "source_text": "string"    # The text that indicates this should be excluded
        }}
    ],
    "analysis": {{
        "total_found": int,           # Total number of exclusions found
        "reasoning": "string"         # Brief explanation of the analysis
    }}
}}

Only include IDs that are explicitly mentioned as NOT being instrument IDs.
If no exclusions are found, return an empty list for excluded_ids.
"""),
            ("human", f"""
Analyze this query for terms that should NOT be treated as instrument IDs:
{state.context.retrospective_query}
""")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        exclude_response = chain.invoke({})
        try:
            response_data = json.loads(exclude_response.content)
            CONFIDENCE_THRESHOLD = 0.7
            state.context.ignore_IDs = [
                item["id"] for item in response_data.get("excluded_ids", [])
                if item.get("confidence", 0) >= CONFIDENCE_THRESHOLD
            ]
        except json.JSONDecodeError:
            state.context.ignore_IDs = []
        return state

    def instr_id_identifier(state: ContextAgentState) -> ContextAgentState:
        state.messages.append(AIMessage(
            name="InstrumentIDIdentifier",
            content="Identifying instrument names in your query...",
            additional_kwargs={
                "stage": "node",
                "process": "query_enricher"
            }
        ))
        result = extract_instruments_with_llm(llm, state.context.retrospective_query)
        instruments = result.get("instruments", [])
        ids = [inst["text"] for inst in instruments if inst["confidence"] > 0.7]
        ids = [id_ for id_ in ids if id_ not in state.context.ignore_IDs]
        valid_instruments = validate_instruments_in_database(db, ids)
        state.context.verif_ID_info = valid_instruments
        state.context.unverif_IDs = [id_ for id_ in ids if id_ not in valid_instruments]
        if state.context.unverif_IDs:
            state.clarification_request = (
                f"The following instrument IDs could not be found in the database: "
                f"{', '.join(state.context.unverif_IDs)}. Please clarify or provide correct IDs."
            )
        return state

    def instr_context_retriever(state: ContextAgentState) -> ContextAgentState:
        state.messages.append(AIMessage(
            name="InstrumentContextRetriever",
            content="Retrieving domain and project knowledge relevant to your query...",
            additional_kwargs={
                "stage": "node",
                "process": "query_enricher"
            }
        ))
        instrument_data = load_instrument_context()
            
        query = state.context.retrospective_query
        query_words_list: List[QueryWords] = []
        type_subtype_groups: Dict[tuple, List[str]] = {}
        verified_type_subtype = []
        if state.context.verif_ID_info:
            for instr_id, info in state.context.verif_ID_info.items():
                db_type = info["type"]
                db_subtype = info["subtype"]
                key_tuple = (db_type, db_subtype)
                if key_tuple not in type_subtype_groups:
                    type_subtype_groups[key_tuple] = []
                type_subtype_groups[key_tuple].append(instr_id)
            verified_type_subtype = [f"{t}_{s}" for t, s in type_subtype_groups.keys()]
        semantic_filtered = identify_and_filter_instruments(llm, query, instrument_data, verified_type_subtype)
        verified_type_subtype_set = set(verified_type_subtype)
        for item in semantic_filtered:
            key = item.get('key')
            selected_fields = item.get('fields', [])
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
        state.context.word_context = query_words_list
        return state

    def router(state: ContextAgentState) -> str:
        if state.clarification_request:
            return "END"
        return "instr_context_retriever"

    workflow = StateGraph(ContextAgentState)
    workflow.add_node("query_history_combiner", query_history_combiner)
    workflow.add_node("instr_id_excluder", instr_id_excluder)
    workflow.add_node("instr_id_identifier", instr_id_identifier)
    workflow.add_node("instr_context_retriever", instr_context_retriever)
    workflow.set_entry_point("query_history_combiner")
    workflow.add_edge('query_history_combiner', 'instr_id_excluder')
    workflow.add_edge('instr_id_excluder', 'instr_id_identifier')
    workflow.add_conditional_edges(
        'instr_id_identifier',
        router,
        {
            "instr_context_retriever": "instr_context_retriever",
            "END": END
        }
    )
    workflow.add_edge('instr_context_retriever', END)
    return workflow.compile()

def context_agent(
        llm: BaseLanguageModel,
        chat_history: List[BaseMessage],
        db: Any
    ) -> dict:
    context = Context(
        retrospective_query="",
        ignore_IDs=[],
        verif_ID_info={},
        unverif_IDs=[],
        word_context=[]
    )
    context_graph = create_context_subgraph(llm, db)
    initial_state = ContextAgentState(
        chat_history=chat_history,
        context=context,
        clarification_request="",
        messages=[]
    )
    final_state = context_graph.invoke(initial_state)
    clarification_request = final_state.get("clarification_request", "")
    messages = final_state.get("messages") or []
    if clarification_request:
        messages.append(AIMessage(
            name="QueryClarification",
            content=clarification_request,
            additional_kwargs={
                "stage": "final",
                "process": "response"
            }
        ))

    logger.info(f"Retrospective query: {context.retrospective_query}")
    logger.info(f"Ignored IDs: {context.ignore_IDs}")
    logger.info(f"Verified IDs: {context.verif_ID_info}")
    logger.info(f"Unverified IDs: {context.unverif_IDs}")
    logger.info(f"Word context: {context.word_context}")

    return {
        "context": final_state['context'],
        "messages": messages
    }