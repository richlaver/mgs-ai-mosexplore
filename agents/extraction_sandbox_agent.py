import datetime as datetime_module
from datetime import datetime
import decimal
import time
import json
import logging
import re
from typing import Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel, Field
import sqlparse
from sqlglot import parse_one, exp
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from tools.sql_security_toolkit import GeneralSQLQueryTool

logger = logging.getLogger(__name__)


def _sanitize_column_name(column: str) -> str:
    if not column:
        return column
    column = column.strip()
    if len(column) >= 2 and column[0] == column[-1] and column[0] in {"'", '"', '`'}:
        column = column[1:-1]
    return column


def _extract_expected_columns(prompt: str) -> Optional[List[str]]:
    if not prompt:
        return None
    match = re.search(r'Output columns:\s*\[([^\]]*)\]', prompt)
    if not match:
        return None
    columns_str = match.group(1)
    raw_columns = [col.strip() for col in columns_str.split(',') if col.strip()]
    cleaned_columns = [_sanitize_column_name(col) for col in raw_columns]
    logger.debug("[expected_columns] Extracted columns: %s", cleaned_columns)
    return cleaned_columns if cleaned_columns else None

class ExtractionSandboxAgentState(TypedDict):
    prompt: str
    sql_query: Optional[str]
    query_error: Optional[str]
    execution_result: Optional[str]
    final_output: Optional[pd.DataFrame]
    previous_errors: List[str]
    attempt_count: int
    messages: List[AIMessage]
    next_path: Optional[str]  # Added for conditional routing after decide_next

def create_extraction_sandbox_subgraph(llm, db, table_info, table_relationship_graph, user_id, global_hierarchy_access):
    logger.debug(
        "Creating extraction sandbox subgraph | tables=%d, rel_nodes=%d, user_id=%s, global_access=%s",
        len(table_info) if isinstance(table_info, list) else -1,
        len(table_relationship_graph) if isinstance(table_relationship_graph, dict) else -1,
        str(user_id),
        str(global_hierarchy_access)
    )
    general_sql_tool = GeneralSQLQueryTool(
        db=db,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access
    )

    checker_tool = QuerySQLCheckerTool(db=db, llm=llm)

    generate_prompt = ChatPromptTemplate.from_template("""
    You are an expert SQL query writer for a MySQL database.

    Current date: {current_date}

    Database schema (tables, columns, relationships):
    {schema}

    User prompt for data to extract:
    {prompt}

    Previous errors (if any; avoid repeating the same mistakes and fix accordingly):
    {errors}

    Task:
    1. Consider 3 to 5 possible SQL queries to achieve the desired outcome.
    2. If the prompt requires extraction from a JSON string, ALWAYS use CASE WHEN JSON_VALID(column) THEN JSON_EXTRACT(column, '$.path') ELSE NULL END to avoid errors with empty strings or invalid JSON.
    3. If the prompt requests time series data, extract with timestamp as first column.
    4. Evaluate each for:
       - Correctness: Does it accurately retrieve the requested data?
       - Performance: Is it efficient (e.g., uses indexes, avoids unnecessary joins)?
       - Reliability: Handles edge cases (e.g., nulls, empty results)?
    5. Check that you are not repeating the same mistakes as in previous errors.
    6. Select the best query.
    7. Use meaningful AS aliases for all selected columns that give full meaning to extracted results (e.g., SELECT value AS settlement_mm).
    8. Ensure the query is read-only (no writes).
    9. Always qualify column names with table aliases in joins to avoid ambiguity.
    10. If the prompt specifies 'Output columns: [col1, col2, ...]', use these exact names as aliases in the SELECT clause. Enclose all aliases in backticks, e.g., AS `settlement_(mm)`. Do not add extra columns.
    11. Fields named 'calculation1', 'calculation2'... etc. are accessible from the JSON stored in the 'custom_fields' column of the 'mydata' table.
    12. When extracting on a specific date, NEVER filter using a single date. Instead use a date range to ensure data is found. For example, for '2023-10-15', use "timestamp >= '2023-10-15 00:00:00' AND timestamp < '2023-10-16 00:00:00'".

    Think step-by-step in your reasoning.
    Finally, output ONLY the selected SQL query (no explanations or extra text).
    """)

    generate_chain = generate_prompt | llm | StrOutputParser()

    def generate_sql(state: ExtractionSandboxAgentState) -> ExtractionSandboxAgentState:
        t0 = time.perf_counter()
        logger.debug(
            "[generate_sql] start | attempt=%s, prev_errors=%d",
            state.get('attempt_count', 0),
            len(state.get('previous_errors', []))
        )
        messages = state.get('messages', [])
        messages.append(AIMessage(
            name="ExtractionSandboxAgent",
            content="Generating SQL query...",
            additional_kwargs={"stage": "intermediate", "process": "action"}
        ))

        prompt = state['prompt']
        errors = "\n\n".join(state.get('previous_errors', [])) or "None"
        inputs = {
            "current_date": datetime.now().strftime('%B %d, %Y'),
            "schema": json.dumps(table_info, indent=2),
            "prompt": prompt,
            "errors": errors
        }
        previous_errors = state.get('previous_errors', [])
        try:
            sql_query = generate_chain.invoke(inputs).strip()
            sql_query = re.sub(r'^```sql\n|```$', '', sql_query).strip()  # Clean any markdown
            logger.debug("[generate_sql] produced SQL (len=%d): %s", len(sql_query), sql_query[:500])
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=f"Generated SQL query: {sql_query}",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[generate_sql] end | duration=%.3fs", time.perf_counter() - t0)
            return {"sql_query": sql_query, "query_error": None, "messages": messages}
        except Exception as e:
            error_msg = f"Error: Generation failed: {str(e)}"
            previous_errors.append(error_msg)
            logger.error("[generate_sql] %s", error_msg)
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=error_msg,
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[generate_sql] end (error) | duration=%.3fs", time.perf_counter() - t0)
            return {
                "sql_query": None,
                "query_error": error_msg,
                "previous_errors": previous_errors,
                "messages": messages
            }

    def check_sql(state: ExtractionSandboxAgentState) -> ExtractionSandboxAgentState:
        t0 = time.perf_counter()
        logger.debug(
            "[check_sql] start | attempt=%s, prev_errors=%d, has_sql=%s",
            state.get('attempt_count', 0),
            len(state.get('previous_errors', [])),
            bool(state.get('sql_query'))
        )
        messages = state.get('messages', [])
        messages.append(AIMessage(
            name="ExtractionSandboxAgent",
            content="Checking SQL query...",
            additional_kwargs={"stage": "intermediate", "process": "action"}
        ))

        sql_query = state.get('sql_query')
        previous_errors = state.get('previous_errors', [])
        
        if not sql_query:
            if state.get('query_error'):
                # Already handled in generation
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=f"Generation error present: {state['query_error']}. Skipping check.",
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))
                logger.debug("[check_sql] skipping due to generation error")
                logger.debug("[check_sql] end | duration=%.3fs", time.perf_counter() - t0)
                return {"messages": messages}
            else:
                error_msg = "Error: No SQL query generated."
                previous_errors.append(error_msg)
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=error_msg,
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))
                logger.error("[check_sql] %s", error_msg)
                logger.debug("[check_sql] end | duration=%.3fs", time.perf_counter() - t0)
                return {
                    "query_error": error_msg,
                    "previous_errors": previous_errors,
                    "sql_query": None,
                    "messages": messages
                }

        # Extract expected columns from prompt
        prompt = state.get('prompt', '')
        expected_columns = _extract_expected_columns(prompt)
        if expected_columns:
            logger.debug("[check_sql] Expected columns from prompt: %s", expected_columns)

        # Parse SQL query to extract aliases using sqlglot
        try:
            ast = parse_one(sql_query, dialect='mysql')  # Use 'mysql' dialect for JSON_EXTRACT support
            select = ast.find(exp.Select)
            if not select:
                raise ValueError("No SELECT clause found in query")
            
            aliases = []
            for expression in select.expressions:
                alias = expression.alias
                if not alias:
                    # Fallback to the expression's name or unaliased column name
                    alias = expression.alias_or_name or str(expression.this or '').strip('`')
                    if not alias:
                        logger.warning("[check_sql] No alias or name for expression: %s", str(expression))
                        continue
                aliases.append(alias.strip('`'))
            
            logger.debug("[check_sql] Extracted aliases: %s", aliases)

            # Compare aliases with expected columns
            if expected_columns and sorted(aliases) != sorted(expected_columns):  # Sort to ignore order
                error_msg = f"Error: SQL query aliases {aliases} do not match expected columns {expected_columns}."
                previous_errors.append(f"SQL: {sql_query}\n{error_msg}")
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=error_msg,
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))
                logger.error("[check_sql] %s", error_msg)
                logger.debug("[check_sql] end (alias mismatch) | duration=%.3fs", time.perf_counter() - t0)
                return {
                    "query_error": error_msg,
                    "previous_errors": previous_errors,
                    "sql_query": None,
                    "messages": messages
                }
        except Exception as e:
            error_msg = f"Error: Failed to parse SQL for alias check: {str(e)}."
            previous_errors.append(f"SQL: {sql_query}\n{error_msg}")
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=error_msg,
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.warning("[check_sql] %s", error_msg)
            logger.debug("[check_sql] end (parse error) | duration=%.3fs", time.perf_counter() - t0)
            return {
                "query_error": error_msg,
                "previous_errors": previous_errors,
                "sql_query": None,
                "messages": messages
            }

        # Proceed with existing query validation only if alias check passed
        try:
            check_result = checker_tool.invoke({"query": sql_query})
            logger.debug("[check_sql] raw check result (len=%d): %s", len(str(check_result)), str(check_result)[:500])
            corrected_query = re.sub(r'^```sql\n|```$', '', check_result).strip()
            if corrected_query != sql_query:
                correction_msg = f"Query was corrected to: {corrected_query}"
                previous_errors.append(f"SQL: {sql_query}\nCorrection: {correction_msg}")
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=f"SQL query corrected: {correction_msg}",
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))
                logger.debug("[check_sql] query corrected")
                sql_query = corrected_query
            else:
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content="SQL query validation passed.",
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))
            logger.debug("[check_sql] end | duration=%.3fs", time.perf_counter() - t0)
            return {"sql_query": sql_query, "query_error": None, "messages": messages}
        except Exception as e:
            error_msg = f"Error: Check failed: {str(e)}"
            previous_errors.append(f"SQL: {sql_query}\n{error_msg}")
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=f"{error_msg}",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.error("[check_sql] %s", error_msg)
            logger.debug("[check_sql] end (check error) | duration=%.3fs", time.perf_counter() - t0)
            return {
                "query_error": error_msg,
                "previous_errors": previous_errors,
                "sql_query": None,
                "messages": messages
            }

    def execute_sql(state: ExtractionSandboxAgentState) -> ExtractionSandboxAgentState:
        t0 = time.perf_counter()
        logger.debug(
            "[execute_sql] start | attempt=%s, prev_errors=%d, has_sql=%s",
            state.get('attempt_count', 0),
            len(state.get('previous_errors', [])),
            bool(state.get('sql_query'))
        )
        messages = state.get('messages', [])
        messages.append(AIMessage(
            name="ExtractionSandboxAgent",
            content="Executing SQL query...",
            additional_kwargs={"stage": "intermediate", "process": "action"}
        ))

        sql_query = state.get('sql_query')
        previous_errors = state.get('previous_errors', [])
        
        if not sql_query:
            error_msg = f"Error: {state.get('query_error', 'No SQL query to execute.')}"
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=error_msg,
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[execute_sql] skipping execution due to no query")
            logger.debug("[execute_sql] end | duration=%.3fs", time.perf_counter() - t0)
            return {"execution_result": error_msg, "messages": messages}

        try:
            result = general_sql_tool._run(sql_query)
            # logger.debug("[execute_sql] result (len=%d): %s", len(str(result)), str(result)[:500])
            logger.debug("[execute_sql] result (len=%d): %s", len(str(result)), str(result))
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content="SQL query executed successfully.",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[execute_sql] end | duration=%.3fs", time.perf_counter() - t0)
            return {"execution_result": result, "messages": messages}
        except Exception as e:
            error_str = str(e)
            
            # Handle specific MySQL connection errors
            if "MySQL Connection not available" in error_str or "OperationalError" in error_str:
                error_msg = f"Error: Database connection error: {error_str}"
                logger.error(f"MySQL connection issue: {error_msg}")
            elif "Table" in error_str and "doesn't exist" in error_str:
                error_msg = f"Error: Table access error: {error_str}"
                logger.error(f"Table not found: {error_msg}")
            else:
                error_msg = f"Error: Error executing query: {error_str}"
                logger.error(f"General execution error: {error_msg}")
            
            previous_errors.append(f"SQL: {sql_query}\n{error_msg}")
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=f"{error_msg}",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[execute_sql] end (error) | duration=%.3fs", time.perf_counter() - t0)
            return {
                "execution_result": error_msg,
                "previous_errors": previous_errors,
                "messages": messages
            }

    def decide_next_node(state: ExtractionSandboxAgentState) -> ExtractionSandboxAgentState:
        logger.debug(
            "[decide_next_node] evaluating | attempt=%s, has_query_error=%s, has_execution_result=%s",
            state.get('attempt_count', 0),
            bool(state.get('query_error')),
            bool(state.get('execution_result'))
        )
        messages = state.get('messages', [])
        attempt_count = state.get('attempt_count', 0)
        previous_errors = state.get('previous_errors', [])
        query_error = state.get('query_error')
        execution_result = state.get('execution_result')
        no_data_msg = "No data was found in the database matching the specified search criteria."

        should_retry = False
        retry_reason = ""

        if query_error:
            should_retry = True
            retry_reason = query_error
        elif execution_result and isinstance(execution_result, str):
            retry_reason = execution_result
            if execution_result.startswith("Error:"):
                should_retry = True
            elif no_data_msg in execution_result:
                should_retry = False

        if should_retry:
            new_attempt_count = attempt_count + 1
            if new_attempt_count >= 5:
                logger.debug("[decide_next_node] max attempts reached -> parse_results")
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=f"Max attempts (5) reached. Proceeding to parse results despite errors.",
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))
                return {
                    **state,
                    "attempt_count": new_attempt_count,
                    "messages": messages,
                    "next_path": "parse_results"
                }
            else:
                previous_errors.append(f"Retry attempt {new_attempt_count}: {retry_reason}")
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=f"Retrying generation due to: {retry_reason[:100]}... Attempt {new_attempt_count}/5.",
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))
                logger.debug("[decide_next_node] retrying | new_attempt=%d", new_attempt_count)
                return {
                    **state,
                    "attempt_count": new_attempt_count,
                    "previous_errors": previous_errors,
                    "messages": messages,
                    "next_path": "generate_sql"
                }
        else:
            logger.debug("[decide_next_node] success -> parse_results")
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content="Proceeding to parse results.",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            return {**state, "messages": messages, "next_path": "parse_results"}

    def parse_results(state: ExtractionSandboxAgentState) -> ExtractionSandboxAgentState:
        t0 = time.perf_counter()
        logger.debug(
            "[parse_results] start | has_execution_result=%s",
            bool(state.get('execution_result'))
        )
        messages = state.get('messages', [])
        messages.append(AIMessage(
            name="ExtractionSandboxAgent",
            content="Parsing execution results...",
            additional_kwargs={"stage": "intermediate", "process": "action"}
        ))

        result = state.get('execution_result', '')
        no_data_msg = "No data was found in the database matching the specified search criteria."
        
        if isinstance(result, str) and (
            result.startswith("Error:") or 
            no_data_msg in result
        ):
            logger.warning(f"[parse_results] Execution failed or no data: {result}")
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=f"Results parsed with issue: {result}",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[parse_results] end (no data) | duration=%.3fs", time.perf_counter() - t0)
            return {"final_output": None, "messages": messages}

        # Extract expected output columns from prompt
        prompt = state.get('prompt', '')
        expected_columns = _extract_expected_columns(prompt)
        if expected_columns:
            logger.debug("[parse_results] Expected columns from prompt: %s", expected_columns)

        try:
            rows: List[dict] = eval(
                result,
                {"__builtins__": {}},
                {
                    "datetime": datetime_module,
                    "date": datetime_module.date,
                    "time": datetime_module.time,
                    "Decimal": decimal.Decimal,
                    "bytes": bytes,
                    "bytearray": bytearray
                }
            ) if result else []
        except Exception as e:
            logger.error(f"[parse_results] Failed to eval result: {str(e)}")
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content=f"Failed to parse results: {str(e)}",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[parse_results] end (parse error) | duration=%.3fs", time.perf_counter() - t0)
            return {"final_output": None, "messages": messages}

        if not rows:
            messages.append(AIMessage(
                name="ExtractionSandboxAgent",
                content="No data found.",
                additional_kwargs={"stage": "intermediate", "process": "observation"}
            ))
            logger.debug("[parse_results] end (empty rows) | duration=%.3fs", time.perf_counter() - t0)
            return {"final_output": None, "messages": messages}

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(rows)

        # Enforce expected columns if specified
        if expected_columns:
            logger.debug("[parse_results] Enforcing expected columns: %s", expected_columns)
            # Check for missing columns
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                logger.warning("[parse_results] Missing columns: %s. Filling with NaN.", missing_cols)
                for col in missing_cols:
                    df[col] = pd.NA
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=f"Missing columns {missing_cols} filled with NaN.",
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))

            # Check for extra columns
            extra_cols = [col for col in df.columns if col not in expected_columns]
            if extra_cols:
                logger.debug("[parse_results] Dropping extra columns: %s", extra_cols)
                df = df.drop(columns=extra_cols)
                messages.append(AIMessage(
                    name="ExtractionSandboxAgent",
                    content=f"Dropped extra columns: {extra_cols}.",
                    additional_kwargs={"stage": "intermediate", "process": "observation"}
                ))

            # Reorder columns to match expected_columns
            df = df[expected_columns]
            logger.debug("[parse_results] Columns reordered to: %s", list(df.columns))

        # Parse quoted numeric strings (e.g., '"-7.37"') to floats, handling scientific notation
        def safe_float(val):
            if isinstance(val, str) and val.startswith('"') and val.endswith('"'):
                try:
                    return float(val.strip('"'))
                except ValueError:
                    return val
            return val

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(safe_float)

        # Robustly convert string numerics to float/int where possible (skip datetimes)
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['reading_timestamp', 'date1']:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().all():
                    df[col] = converted
                    if df[col].dtype == 'float64' and df[col].eq(df[col].astype('int64')).all():
                        df[col] = df[col].astype('int64')
                    logger.debug(f"[parse_results] Converted column '{col}' to numeric (dtype: {df[col].dtype})")

        messages.append(AIMessage(
            name="ExtractionSandboxAgent",
            content="Results parsed successfully.",
            additional_kwargs={"stage": "intermediate", "process": "observation"}
        ))
        logger.debug("[parse_results] DataFrame created | shape=%s, columns=%s", df.shape, list(df.columns))
        logger.debug("[parse_results] end | duration=%.3fs", time.perf_counter() - t0)
        return {"final_output": df, "messages": messages}

    graph = StateGraph(ExtractionSandboxAgentState)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("check_sql", check_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("decide_next", decide_next_node)
    graph.add_node("parse_results", parse_results)

    graph.set_entry_point("generate_sql")
    graph.add_edge("generate_sql", "check_sql")
    graph.add_edge("check_sql", "execute_sql")
    graph.add_edge("execute_sql", "decide_next")
    graph.add_conditional_edges(
        "decide_next",
        lambda state: state.get("next_path", "parse_results"),
        {"generate_sql": "generate_sql", "parse_results": "parse_results"}
    )
    graph.add_edge("parse_results", END)

    compiled = graph.compile()
    logger.debug("Extraction sandbox subgraph compiled successfully.")
    return compiled


class ExtractionSandboxAgentInput(BaseModel):
    """Structured input schema for the ExtractionSandboxAgentTool."""
    prompt: str = Field(..., description="User prompt describing the data to extract.")


class ExtractionSandboxAgentTool(BaseTool):
    name: str = "extraction_sandbox_agent"
    description: str = (
        "Tool to run the data extraction agent in a sandboxed environment. Extracts data from SQL database.\n\n"
        "Input: A string prompt describing the data to extract."
        "The prompt should include 'Output columns: [col1, col2, ...]' to specify the desired DataFrame column names.\n\n"
        "Returns: Pandas DataFrame containing the extracted data with specified columns, or None if an error occurred or no data was found."
    )
    llm: BaseLanguageModel = Field(...)
    db: SQLDatabase = Field(...)
    table_info: List[Dict] = Field(...)
    table_relationship_graph: Dict[str, List[tuple]] = Field(...)
    user_id: int = Field(...)
    global_hierarchy_access: bool = Field(...)
    args_schema: Type[BaseModel] = ExtractionSandboxAgentInput

    def __init__(
        self,
        llm,
        db,
        table_info,
        table_relationship_graph,
        user_id,
        global_hierarchy_access,
    ):
        logger.debug(
            "Initializing ExtractionSandboxAgentTool | user_id=%s, global_access=%s, tables=%d, rel_nodes=%d",
            str(user_id),
            str(global_hierarchy_access),
            len(table_info) if isinstance(table_info, list) else -1,
            len(table_relationship_graph) if isinstance(table_relationship_graph, dict) else -1,
        )
        super().__init__(
            llm=llm,
            db=db,
            table_info=table_info,
            table_relationship_graph=table_relationship_graph,
            user_id=user_id,
            global_hierarchy_access=global_hierarchy_access
        )

    def _run(self, prompt: str) -> Optional[pd.DataFrame]:
        t0 = time.perf_counter()
        logger.debug("Tool run started | prompt(len=%d): %s", len(prompt), prompt[:500])
        extraction_graph = create_extraction_sandbox_subgraph(
            llm=self.llm,
            db=self.db,
            table_info=self.table_info,
            table_relationship_graph=self.table_relationship_graph,
            user_id=self.user_id,
            global_hierarchy_access=self.global_hierarchy_access
        )
        initial_state = {
            "prompt": prompt,
            "previous_errors": [],
            "attempt_count": 0,
            "messages": [AIMessage(
                name="ExtractionSandboxAgent",
                content="Starting data extraction process.",
                additional_kwargs={"stage": "intermediate", "process": "action"}
            )]
        }
        logger.debug("Invoking extraction graph...")
        final_state = extraction_graph.invoke(initial_state)
        logger.debug("Graph invocation completed")
        for msg in final_state.get('messages', []):
            logger.info(msg.content)
        final_output = final_state.get('final_output', None)
        if isinstance(final_output, pd.DataFrame):
            logger.debug("Tool run completed | result shape=%s", final_output.shape)
        else:
            logger.debug("Tool run completed | result is None or non-DataFrame: %s", type(final_output).__name__)
        logger.debug("Total run duration=%.3fs", time.perf_counter() - t0)
        return final_output

    def invoke(self, input=None, **kwargs):
        """Normalize various call styles into the expected single input.

        Supported forms:
        - invoke(prompt=...)  -> treated as the tool input string
        - invoke(input=...)   -> standard Runnable interface
        - invoke({"prompt": ...}) -> dict input using args_schema
        - invoke("...")      -> plain string input
        """
        if input is None and "prompt" in kwargs:
            input = kwargs.pop("prompt")
        if isinstance(input, dict) and set(input.keys()) == {"prompt"}:
            input = input["prompt"]
        return super().invoke(input, **kwargs)

def extraction_sandbox_agent(
    llm,
    db,
    table_info,
    table_relationship_graph,
    user_id,
    global_hierarchy_access,
) -> BaseTool:
    """
    Initializes and returns the ExtractionSandboxAgentTool instance with the provided dependencies.

    Args:
        llm: An instance of a LangChain LLM (e.g., Gemini).
        db: An instance of SQLDatabase from langchain_community.utilities.sql_database.
        table_info: List[Dict] containing database schema information.
        table_relationship_graph: Dict[str, List[Tuple]] for table relationships.
        user_id: int, current user ID for permissions.
        global_hierarchy_access: bool, whether user has global access.

    Returns:
        BaseTool: An instance of ExtractionSandboxAgentTool configured with the provided dependencies.
    """
    logger.debug("Factory: creating ExtractionSandboxAgentTool")
    return ExtractionSandboxAgentTool(
        llm=llm,
        db=db,
        table_info=table_info,
        table_relationship_graph=table_relationship_graph,
        user_id=user_id,
        global_hierarchy_access=global_hierarchy_access,
    )