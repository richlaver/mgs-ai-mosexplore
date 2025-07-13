from typing import Any, Dict, Optional, Sequence, Type, Union, List, Tuple

from sqlalchemy.engine import Result

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
import streamlit as st
import sqlparse
from collections import deque
from tools.get_user_permissions import HierarchyPermissionsDict
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class BaseUserPermissionsTool(BaseModel):
    """Base tool for applying with user permissions."""

    user_permissions: List[HierarchyPermissionsDict] = Field(exclude=True)
    table_relationship_graph: Dict[str, List[Tuple]] = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _CustomQuerySQLDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")
    # user_permissions: HierarchyPermissionsDict = Field(..., description="""
    #     A dictionary describing the projects, contracts and sites that a user can access.
    # """)


class CustomQuerySQLDatabaseTool(BaseSQLDatabaseTool, BaseUserPermissionsTool, BaseTool):
    """Tool for querying a SQL database.
    The tool will modify the inputted query to ensure the user cannot access 
    any information beyond the user's hierarchy permissions.
    """

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _CustomQuerySQLDatabaseToolInput


    def _run(
        self,
        query: str,
        # user_permissions: List[HierarchyPermissionsDict],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        def unpack_permissions() -> dict:
            """Collate lists of project, contract and site IDs for WHERE condition."""
            project_ids = [p.project_id for u in self.user_permissions for p in u.projects if not p.specific_contracts]
            contract_ids = [c.contract_id for u in self.user_permissions for p in u.projects for c in p.specific_contracts if not c.specific_sites]
            site_ids = [s.site_id for u in self.user_permissions for p in u.projects for c in p.specific_contracts for s in c.specific_sites]
            logging.debug(f'Unpacked project_ids: {project_ids}')
            logging.debug(f'Unpacked contract_ids: {contract_ids}')
            logging.debug(f'Unpacked site_ids: {site_ids}')
            return {'project_ids': project_ids, 'contract_ids': contract_ids, 'site_ids': site_ids}


        def get_all_tables(query):
            """Extract all table names and their aliases from FROM and JOIN clauses."""
            parsed = sqlparse.parse(query)[0]
            logging.debug(f'Parsed SQL query: {parsed}')
            tables = []
            for i, token in enumerate(parsed.tokens):
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    # Look for identifiers after FROM
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
                            break  # Stop at JOIN to handle it separately
                elif token.ttype is sqlparse.tokens.Keyword and token.value.upper().startswith('JOIN'):
                    # Find the identifier after JOIN
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
            queue = deque([(start_table, [(start_table, None, None)])])
            visited = set([start_table])
            
            while queue:
                table, path = queue.popleft()
                if table == target_table:
                    logging.debug(f'Found shortest path: {path}')
                    return path
                for next_table, src_col, tgt_col in self.table_relationship_graph[table]:
                    if next_table not in visited:
                        visited.add(next_table)
                        queue.append((next_table, path + [(next_table, src_col, tgt_col)]))
            return None


        def extend_query(query, project_ids, contract_ids, site_ids):
            """Dynamically rewrite the query to add JOIN and WHERE clauses."""
            logging.debug(f'Original query: {query}')
            # Get all tables and aliases
            table_aliases = get_all_tables(query)
            if not table_aliases:
                logging.debug('Didn\'t find any tables in the query. Returning the original query.')
                return query  # No tables found, return original query
            
            # Check if location is already in the query
            location_alias = None
            for table_name, alias in table_aliases:
                if table_name == "location":
                    logging.debug('Detected table "location" in the query.')
                    location_alias = alias
                    break
            
            # Parse query
            parsed = sqlparse.parse(query)[0]
            where_clause = None
            where_index = None
            for i, token in enumerate(parsed.tokens):
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'WHERE':
                    # The WHERE clause includes the WHERE keyword and the following condition
                    where_clause = token
                    where_index = i
                    # Include the condition tokens that follow WHERE
                    for j in range(i + 1, len(parsed.tokens)):
                        if parsed.tokens[j].ttype is sqlparse.tokens.Keyword and parsed.tokens[j].value.upper() in ('GROUP', 'ORDER', 'LIMIT'):
                            break
                        where_clause = ''.join(str(t) for t in parsed.tokens[i:j + 1])
                    break
            
            def format_ids(ids: list) -> str:
                return ",".join(str(id) for id in ids)

            # Build conditions only for non-empty ID lists
            conditions = []
            if project_ids:
                conditions.append(f"location.project_id IN ({format_ids(project_ids)})")
            if contract_ids:
                conditions.append(f"location.contract_id IN ({format_ids(contract_ids)})")
            if site_ids:
                conditions.append(f"location.site_id IN ({format_ids(site_ids)})")
            
            # If no conditions, return original query
            if not conditions:
                logging.debug('No valid permissions to enforce. Returning original query.')
                return query

            new_where_condition = f"({' OR '.join(conditions)})"
            
            # Initialize tokens with parsed tokens
            tokens = list(parsed.tokens)
            
            if location_alias:
                # Edge Case 1: location is in the query, only add WHERE condition
                if where_clause:
                    new_where = f"{where_clause} AND {new_where_condition}"
                    tokens = tokens[:where_index] + [new_where] + tokens[where_index + 1:]
                else:
                    new_where = f" WHERE {new_where_condition}"
                    tokens = tokens + [new_where]
                extended_query = "".join(str(token) for token in tokens)
                logging.debug(f'Returning extended query: {extended_query}')
                return extended_query
            
            # Find shortest join path to location
            min_distance = float('inf')
            selected_path = None
            selected_alias = None
            for table_name, alias in table_aliases:
                path = find_shortest_path(table_name)
                if path and len(path) - 1 < min_distance:
                    min_distance = len(path) - 1
                    selected_path = path
                    selected_alias = alias
            logging.debug(f'Found shortest join path to location: {selected_path} with alias {selected_alias}')
            
            # Edge Case 2: No table connects to location
            if not selected_path:
                logging.debug('No table connects to location. Returning original query.')
                return query
            
            # Build JOIN clauses
            join_clauses = []
            prev_table = selected_alias
            for table, src_col, tgt_col in selected_path[1:]:  # Skip first (start table)
                if table == "location":
                    join_clauses.append(f" JOIN location ON {prev_table}.{src_col} = location.{tgt_col}")
                else:
                    join_clauses.append(f" JOIN {table} ON {prev_table}.{src_col} = {table}.{tgt_col}")
                prev_table = table
            
            # Insert JOIN clauses after FROM and table name
            from_index = None
            table_index = None
            for i, token in enumerate(tokens):
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    from_index = i
                    # Look for the table name (Identifier) after FROM
                    for j in range(i + 1, len(tokens)):
                        if isinstance(tokens[j], sqlparse.sql.Identifier):
                            table_index = j
                            break
                        elif isinstance(tokens[j], sqlparse.sql.IdentifierList):
                            table_index = j
                            break
                        elif tokens[j].ttype is sqlparse.tokens.Keyword and tokens[j].value.upper().startswith('JOIN'):
                            break
                    break
            if from_index is None:
                logging.error("No FROM clause found in query")
                raise ValueError("No FROM clause found")
            if table_index is None:
                logging.error("No table name found after FROM clause")
                raise ValueError("No table name found after FROM clause")
            
            # Insert JOIN clauses immediately after the table name
            insert_index = table_index + 1
            
            # Insert JOIN clauses
            tokens = (
                tokens[:insert_index] +
                [join_clause for join_clause in join_clauses] +
                tokens[insert_index:]
            )
            
            # Apply WHERE clause
            if where_clause:
                new_where = f"{where_clause} AND {new_where_condition}"
                tokens = tokens[:where_index] + [new_where] + tokens[where_index + 1:]
            else:
                new_where = f" WHERE {new_where_condition}"
                tokens = tokens + [new_where]
            
            # Reconstruct query
            extended_query = "".join(str(token) for token in tokens)
            logging.debug(f'Returning extended query: {extended_query}')
            
            return extended_query
        
        unpacked_permissions = unpack_permissions()
        extended_query = extend_query(
            query=query,
            project_ids=unpacked_permissions['project_ids'],
            contract_ids=unpacked_permissions['contract_ids'],
            site_ids=unpacked_permissions['site_ids'])
        return self.db.run_no_throw(extended_query)