from typing import Any, Dict, Optional, Sequence, Union, List, Tuple

from sqlalchemy.engine import Result

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from utils.run_cancellation import get_active_run_controller, RunCancelledError
import sqlparse
from collections import deque
import logging
import re

logger = logging.getLogger(__name__)


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class BaseUserPermissionsTool(BaseModel):
    """Base tool for applying with user permissions."""

    table_relationship_graph: Dict[str, List[Tuple]] = Field(exclude=True)
    user_id: int = Field(exclude=True)
    global_hierarchy_access: bool = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class GeneralSQLQueryTool(BaseSQLDatabaseTool, BaseUserPermissionsTool, BaseTool):
    """Tool for executing general-purpose SQL queries with security controls.
    This tool handles ad-hoc queries and data exploration, applying user permissions
    to ensure data access is properly restricted. It is not intended for generating
    plots or formatted tables, which have their own specialized query tools.
    """

    name: str = "general_sql_query"
    description: str = """
    Execute a general-purpose SQL query for data exploration and analysis.
    Use this tool for ad-hoc queries and general data retrieval.
    Do not use this tool for generating plots or formatted tables - use specialized tools instead.

    Input:
    - A string containing a detailed and correct SQL query for general data retrieval
    Returns:
    - Raw query results with user's hierarchy permissions automatically applied
    
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    
    Example: "SELECT * FROM users LIMIT 10"
    """


    def strip_markdown(self, query: str) -> str:
        """
Remove markdown code fences and backticks from the query, preserving valid SQL 
syntax. Also handles multiple SQL statements by keeping only the first one.
"""
        logging.debug(f"Original query before markdown stripping: {query}")
        
        query = query.strip()
        if query.startswith("```sql") or query.startswith("```"):
            lines = query.splitlines()
            if lines and lines[-1].strip() == "```":
                query = "\n".join(lines[1:-1]).strip()
            else:
                query = "\n".join(lines[1:]).strip()
        
        parsed = sqlparse.parse(query)
        if len(parsed) > 1:
            logging.warning(f"""Multiple SQL statements detected. 
Only the first statement will be executed. Discarded statements: 
{len(parsed) - 1}""")
            query = str(parsed[0])
        
        logging.debug(f"Query after markdown stripping and statement isolation: {query}")
        return query

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        controller = get_active_run_controller()
        if controller:
            controller.raise_if_cancelled("sql-tool:start")
        original_query = query

        def _truncate(text: Any, max_len: int = 500) -> str:
            if text is None:
                return ""
            text = str(text)
            return text if len(text) <= max_len else text[:max_len] + "...[truncated]"

        logger.info(
            "[GeneralSQLQueryTool] Received query len=%d user_id=%s gha=%s | %s",
            len(original_query),
            self.user_id,
            self.global_hierarchy_access,
                original_query
        )
        # First strip any markdown formatting that might wrap a JSON string
        if query.startswith("```"):
            lines = query.splitlines()
            if len(lines) > 1:
                # Remove first line (```json or just ```) and last line (```)
                if lines[-1].strip() == "```":
                    query = "\n".join(lines[1:-1])
                else:
                    query = "\n".join(lines[1:])
            query = query.strip()
        
        # Handle case where input is a JSON string
        if '"query":' in query or "'query':" in query:
            try:
                import json
                # Replace single quotes with double quotes if 'query': is found
                if "'query':" in query:
                    query = query.replace("'", '"')
                query_dict = json.loads(query)
                if isinstance(query_dict, dict) and 'query' in query_dict:
                    query = query_dict['query']
            except json.JSONDecodeError:
                # If JSON parsing fails, treat it as a regular query string
                pass
                
        query = self.strip_markdown(query)
        if query != original_query:
            logger.info(
                "[GeneralSQLQueryTool] Query normalized (len=%d): %s",
                len(query),
                _truncate(query, 300)
            )

        def check_for_write_statements(query: str) -> Optional[str]:
            """
            Check if the query contains write statements, returning an error 
            message if found."""
            write_keywords = {
                'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
                'TRUNCATE', 'MERGE', 'GRANT', 'REVOKE', 'SET', 'REPLACE'
            }
            parsed = sqlparse.parse(query)
            for statement in parsed:
                if statement.get_type().upper() in write_keywords:
                    error_message = f"""Query contains forbidden write 
                    operation: {statement.get_type().upper()}. Only read-only 
                    queries are allowed."""
                    logging.error(error_message)
                    return error_message
            logging.debug("No write operations detected in query.")
            return None

        def get_all_tables(query):
            """Extract all table names and their aliases from FROM and JOIN 
            clauses."""
            parsed = sqlparse.parse(query)[0]
            logging.debug(f'Parsed SQL query: {parsed}')
            tables = []
            for i, token in enumerate(parsed.tokens):
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    for sub_token in parsed.tokens[i + 1:]:
                        if isinstance(sub_token, sqlparse.sql.Identifier):
                            table_name = sub_token.get_real_name()
                            alias = sub_token.get_alias() or table_name
                            if table_name:
                                tables.append((table_name.lower(), alias))
                        elif isinstance(sub_token, sqlparse.sql.IdentifierList):
                            for identifier in sub_token.get_identifiers():
                                table_name = identifier.get_real_name()
                                alias = identifier.get_alias() or table_name
                                if table_name:
                                    tables.append((table_name.lower(), alias))
                        elif sub_token.ttype is sqlparse.tokens.Keyword and sub_token.value.upper().startswith('JOIN'):
                            break
                elif token.ttype is sqlparse.tokens.Keyword and token.value.upper().startswith('JOIN'):
                    for sub_token in parsed.tokens[i + 1:]:
                        if isinstance(sub_token, sqlparse.sql.Identifier):
                            table_name = sub_token.get_real_name()
                            alias = sub_token.get_alias() or table_name
                            if table_name:
                                tables.append((table_name.lower(), alias))
                            break
            logging.debug(f'Getting tables from query: {tables}')
            return tables

        def find_shortest_path(start_table, target_table="location"):
            """Find shortest join path to target_table using BFS."""
            if start_table not in self.table_relationship_graph:
                logging.warning(f"Start table {start_table} not found in table_relationship_graph")
                return None
            if target_table not in self.table_relationship_graph:
                logging.warning(f"Target table {target_table} not found in table_relationship_graph")
                return None
            queue = deque([(start_table, [(start_table, None, None)])])
            visited = set([start_table])
            
            while queue:
                table, path = queue.popleft()
                if table == target_table:
                    logging.debug(f'Found shortest path: {path}')
                    return path
                if table not in self.table_relationship_graph:
                    logging.warning(f"Table {table} not found in table_relationship_graph")
                    continue
                for next_table, src_col, tgt_col in self.table_relationship_graph[table]:
                    if next_table not in visited:
                        visited.add(next_table)
                        queue.append((next_table, path + [(next_table, src_col, tgt_col)]))
            return None

        def extend_query(query: str) -> str:
            """Apply deterministic permissions filtering without rewriting joins."""
            logging.debug(f"Original query: {query}")
            query = query.rstrip().rstrip(";")
            logger.info("[GeneralSQLQueryTool] Applying permissions filter: %s", _truncate(query, 300))

            parsed = sqlparse.parse(query)
            if not parsed:
                return query
            statement = parsed[0]

            def _select_clause(stmt: sqlparse.sql.Statement) -> str:
                in_select = False
                parts: List[str] = []
                for token in stmt.tokens:
                    if token.ttype is sqlparse.tokens.DML and token.value.upper() == "SELECT":
                        in_select = True
                        continue
                    if in_select:
                        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
                            break
                        parts.append(str(token))
                return "".join(parts)

            def _permission_clause_for_location(location_ref: str) -> str:
                return (
                    "EXISTS ("
                    "SELECT 1 "
                    "FROM user_access_groups_permissions uagp "
                    "JOIN user_access_groups uag ON uagp.user_group_id = uag.id "
                    "JOIN user_access_groups_users uagu ON uag.id = uagu.group_id "
                    "JOIN geo_12_users u ON uagu.user_id = u.id "
                    f"WHERE u.id = {self.user_id} "
                    "AND uagu.user_deleted = 0 "
                    "AND u.prohibit_portal_access NOT IN (1, 2, 3) "
                    "AND ("
                    f"(uagp.project = 0 OR uagp.project = {location_ref}.project_id) "
                    f"AND (uagp.contract = 0 OR uagp.contract = {location_ref}.contract_id OR uagp.project = 0) "
                    f"AND (uagp.site = 0 OR uagp.site = {location_ref}.site_id OR uagp.contract = 0 OR uagp.project = 0)"
                    ")"
                    ")"
                )

            def _permission_clause_for_instr_id(instr_id_ref: str) -> str:
                return (
                    "EXISTS ("
                    "SELECT 1 "
                    "FROM instrum i_perm "
                    "JOIN location l_perm ON i_perm.location_id = l_perm.id "
                    "JOIN user_access_groups_permissions uagp ON 1=1 "
                    "JOIN user_access_groups uag ON uagp.user_group_id = uag.id "
                    "JOIN user_access_groups_users uagu ON uag.id = uagu.group_id "
                    "JOIN geo_12_users u ON uagu.user_id = u.id "
                    f"WHERE i_perm.instr_id = {instr_id_ref} "
                    f"AND u.id = {self.user_id} "
                    "AND uagu.user_deleted = 0 "
                    "AND u.prohibit_portal_access NOT IN (1, 2, 3) "
                    "AND ("
                    "(uagp.project = 0 OR uagp.project = l_perm.project_id) "
                    "AND (uagp.contract = 0 OR uagp.contract = l_perm.contract_id OR uagp.project = 0) "
                    "AND (uagp.site = 0 OR uagp.site = l_perm.site_id OR uagp.contract = 0 OR uagp.project = 0)"
                    ")"
                    ")"
                )

            def _allowed_instr_subquery() -> str:
                return (
                    "SELECT DISTINCT i_perm.instr_id "
                    "FROM instrum i_perm "
                    "JOIN location l_perm ON i_perm.location_id = l_perm.id "
                    "JOIN user_access_groups_permissions uagp ON 1=1 "
                    "JOIN user_access_groups uag ON uagp.user_group_id = uag.id "
                    "JOIN user_access_groups_users uagu ON uag.id = uagu.group_id "
                    "JOIN geo_12_users u ON uagu.user_id = u.id "
                    f"WHERE u.id = {self.user_id} "
                    "AND uagu.user_deleted = 0 "
                    "AND u.prohibit_portal_access NOT IN (1, 2, 3) "
                    "AND ("
                    "(uagp.project = 0 OR uagp.project = l_perm.project_id) "
                    "AND (uagp.contract = 0 OR uagp.contract = l_perm.contract_id OR uagp.project = 0) "
                    "AND (uagp.site = 0 OR uagp.site = l_perm.site_id OR uagp.contract = 0 OR uagp.project = 0)"
                    ")"
                )

            def _append_where_predicate(stmt: sqlparse.sql.Statement, predicate: str) -> str:
                where_token = None
                for token in stmt.tokens:
                    if isinstance(token, sqlparse.sql.Where):
                        where_token = token
                        break
                if where_token:
                    new_where = str(where_token).rstrip() + " AND " + predicate
                    rendered = []
                    for token in stmt.tokens:
                        if token is where_token:
                            rendered.append(new_where)
                        else:
                            rendered.append(str(token))
                    return "".join(rendered)

                insert_at = len(stmt.tokens)
                for i, token in enumerate(stmt.tokens):
                    if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in {"GROUP", "ORDER", "LIMIT", "HAVING"}:
                        insert_at = i
                        break
                rendered = []
                for i, token in enumerate(stmt.tokens):
                    if i == insert_at:
                        rendered.append(" WHERE " + predicate + " ")
                    rendered.append(str(token))
                if insert_at == len(stmt.tokens):
                    rendered.append(" WHERE " + predicate + " ")
                return "".join(rendered)

            select_clause = _select_clause(statement)
            select_lower = select_clause.lower()
            output_instr_col = None
            if re.search(r"\binstrument_id\b", select_lower):
                output_instr_col = "instrument_id"
            elif re.search(r"\binstr_id\b", select_lower):
                output_instr_col = "instr_id"

            raw = str(statement)

            if output_instr_col:
                allowed_instr = _allowed_instr_subquery()
                wrapped = (
                    "SELECT __perm_q.* FROM (" + query + ") AS __perm_q "
                    "JOIN (" + allowed_instr + ") AS __perm_allow "
                    f"ON __perm_allow.instr_id = __perm_q.{output_instr_col}"
                )
                logger.info("[GeneralSQLQueryTool] Applied outer permissions join via %s", output_instr_col)
                return wrapped

            if re.search(r"\bas\s+instrument_id\b", raw, re.IGNORECASE):
                allowed_instr = _allowed_instr_subquery()
                wrapped = (
                    "SELECT __perm_q.* FROM (" + query + ") AS __perm_q "
                    "JOIN (" + allowed_instr + ") AS __perm_allow "
                    "ON __perm_allow.instr_id = __perm_q.instrument_id"
                )
                logger.info("[GeneralSQLQueryTool] Applied outer permissions join via instrument_id (projected)")
                return wrapped
            if re.search(r"\bas\s+instr_id\b", raw, re.IGNORECASE):
                allowed_instr = _allowed_instr_subquery()
                wrapped = (
                    "SELECT __perm_q.* FROM (" + query + ") AS __perm_q "
                    "JOIN (" + allowed_instr + ") AS __perm_allow "
                    "ON __perm_allow.instr_id = __perm_q.instr_id"
                )
                logger.info("[GeneralSQLQueryTool] Applied outer permissions join via instr_id (projected)")
                return wrapped

            # No instrument identifier in output; attempt correlated filtering using known table aliases.
            alias_pattern = re.compile(
                r"\b(from|join)\s+([`\"\[]?[\w\.]+[`\"\]]?)\s+(?:as\s+)?([`\"\[]?[\w]+[`\"\]]?)",
                re.IGNORECASE,
            )
            table_aliases: Dict[str, str] = {}
            for _, table, alias in alias_pattern.findall(raw):
                clean_table = table.strip("`\"[]").split(".")[-1].lower()
                clean_alias = alias.strip("`\"[]")
                table_aliases[clean_table] = clean_alias

            if "location" in table_aliases:
                predicate = _permission_clause_for_location(table_aliases["location"])
                logger.info("[GeneralSQLQueryTool] Applied permissions using location alias %s", table_aliases["location"])
                return _append_where_predicate(statement, predicate)

            if "instrum" in table_aliases:
                instr_alias = table_aliases["instrum"]
                predicate = (
                    "EXISTS (SELECT 1 FROM location l_perm "
                    "JOIN user_access_groups_permissions uagp ON 1=1 "
                    "JOIN user_access_groups uag ON uagp.user_group_id = uag.id "
                    "JOIN user_access_groups_users uagu ON uag.id = uagu.group_id "
                    "JOIN geo_12_users u ON uagu.user_id = u.id "
                    f"WHERE l_perm.id = {instr_alias}.location_id "
                    f"AND u.id = {self.user_id} "
                    "AND uagu.user_deleted = 0 "
                    "AND u.prohibit_portal_access NOT IN (1, 2, 3) "
                    "AND ("
                    "(uagp.project = 0 OR uagp.project = l_perm.project_id) "
                    "AND (uagp.contract = 0 OR uagp.contract = l_perm.contract_id OR uagp.project = 0) "
                    "AND (uagp.site = 0 OR uagp.site = l_perm.site_id OR uagp.contract = 0 OR uagp.project = 0)"
                    ")"
                    ")"
                )
                logger.info("[GeneralSQLQueryTool] Applied permissions using instrum alias %s", instr_alias)
                return _append_where_predicate(statement, predicate)

            alias_instr_match = re.search(r"\b([A-Za-z_][\w]*)\.instr_id\b", raw)
            if alias_instr_match:
                instr_alias = alias_instr_match.group(1)
                predicate = _permission_clause_for_instr_id(f"{instr_alias}.instr_id")
                logger.info("[GeneralSQLQueryTool] Applied permissions using instr_id alias %s", instr_alias)
                return _append_where_predicate(statement, predicate)

            for table_name, col in (("contracts", "contract_id"), ("projects", "project_id"), ("sites", "site_id")):
                if table_name in table_aliases:
                    t_alias = table_aliases[table_name]
                    predicate = (
                        "EXISTS (SELECT 1 FROM location l_perm "
                        "JOIN user_access_groups_permissions uagp ON 1=1 "
                        "JOIN user_access_groups uag ON uagp.user_group_id = uag.id "
                        "JOIN user_access_groups_users uagu ON uag.id = uagu.group_id "
                        "JOIN geo_12_users u ON uagu.user_id = u.id "
                        f"WHERE l_perm.{col} = {t_alias}.id "
                        f"AND u.id = {self.user_id} "
                        "AND uagu.user_deleted = 0 "
                        "AND u.prohibit_portal_access NOT IN (1, 2, 3) "
                        "AND ("
                        "(uagp.project = 0 OR uagp.project = l_perm.project_id) "
                        "AND (uagp.contract = 0 OR uagp.contract = l_perm.contract_id OR uagp.project = 0) "
                        "AND (uagp.site = 0 OR uagp.site = l_perm.site_id OR uagp.contract = 0 OR uagp.project = 0)"
                        ")"
                        ")"
                    )
                    logger.info("[GeneralSQLQueryTool] Applied permissions using %s alias %s", table_name, t_alias)
                    return _append_where_predicate(statement, predicate)

            logger.warning("[GeneralSQLQueryTool] Unable to apply permissions filter to query; executing unmodified")
            return query

        def process_results(results):
            """Process query results, handling empty results consistently."""
            if isinstance(results, str) and not results.strip():
                logger.info("[GeneralSQLQueryTool] Empty string result; returning NO_DATA message")
                return "No data was found in the database matching the specified search criteria."
            logger.info("[GeneralSQLQueryTool] Returning result type=%s", type(results).__name__)
            return results

        write_error = check_for_write_statements(query)
        if write_error:
            logger.info("[GeneralSQLQueryTool] Write statement detected -> %s", write_error)
            return write_error
        
        try:
            if controller:
                controller.raise_if_cancelled("sql-tool:before-exec")
            if self.global_hierarchy_access:
                logger.info("[GeneralSQLQueryTool] Global hierarchy access granted; executing original query")
                logger.info("[GeneralSQLQueryTool] Executing query full: %s", query)
                results = self.db.run_no_throw(query, include_columns=True)
            else:
                extended_query = extend_query(query=query)
                logger.info("[GeneralSQLQueryTool] Executing permissions-filtered query")
                logger.info("[GeneralSQLQueryTool] Executing query full: %s", extended_query)
                results = self.db.run_no_throw(extended_query, include_columns=True)

            if isinstance(results, list):
                logger.info("[GeneralSQLQueryTool] Query returned %d row(s)", len(results))
            else:
                logger.info(
                    "[GeneralSQLQueryTool] Query returned %s", _truncate(str(results), 300)
                )
            return process_results(results)
        except RunCancelledError:
            logger.info("[GeneralSQLQueryTool] Query cancelled mid-flight")
            raise
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            raise

