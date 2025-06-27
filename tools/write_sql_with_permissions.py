from typing import Any, Dict, Optional, Sequence, Type, Union

from sqlalchemy.engine import Result

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
import streamlit as st
import sqlparse
from collections import deque
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


class _CustomQuerySQLDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


class CustomQuerySQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""


        def get_all_tables(query):
            """Extract all table names and their aliases from FROM and JOIN clauses."""
            parsed = sqlparse.parse(query)[0]
            logging.debug(f'Parsed SQL query: {parsed}')
            tables = []
            for token in parsed.tokens:
                if isinstance(token, sqlparse.sql.From):
                    for identifier in token.get_identifiers():
                        table_name = identifier.get_real_name()
                        alias = identifier.get_alias() or table_name
                        if table_name:
                            tables.append((table_name.lower(), alias))
                elif isinstance(token, sqlparse.sql.Token) and token.value.upper().startswith('JOIN'):
                    # Find the identifier after JOIN
                    for sub_token in token.parent.tokens[token.parent.token_index(token) + 1:]:
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
                for next_table, src_col, tgt_col in st.session_state.table_relationship_graph[table]:
                    if next_table not in visited:
                        visited.add(next_table)
                        queue.append((next_table, path + [(next_table, src_col, tgt_col)]))
            return None


        def extend_query(query, project_ids):
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
                if isinstance(token, sqlparse.sql.Where):
                    where_clause = token
                    where_index = i
                    break
            
            # Build WHERE condition with placeholders
            project_ids_str = ",".join(f"%s" for _ in project_ids) if project_ids else "NULL"
            new_where_condition = f"location.project_id NOT IN ({project_ids_str})"
            
            if location_alias:
                # Edge Case 1: location is in the query, only add WHERE condition
                if where_clause:
                    new_where = f"{where_clause} AND {new_where_condition}"
                    tokens = parsed.tokens[:where_index] + [new_where] + parsed.tokens[where_index + 1:]
                else:
                    new_where = f"WHERE {new_where_condition}"
                    tokens = parsed.tokens + [new_where]
                extended_query = "".join(str(token) for token in tokens)
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
                return query, []  # Return original query unchanged
            
            # Build JOIN clauses
            join_clauses = []
            prev_table = selected_alias
            for table, src_col, tgt_col in selected_path[1:]:  # Skip first (start table)
                if table == "location":
                    join_clauses.append(f"JOIN location ON {prev_table}.{src_col} = location.{tgt_col}")
                else:
                    join_clauses.append(f"JOIN {table} ON {prev_table}.{src_col} = {table}.{tgt_col}")
                prev_table = table
            
            # Insert JOIN clauses after FROM and existing JOINs
            from_index = None
            for i, token in enumerate(parsed.tokens):
                if isinstance(token, sqlparse.sql.From):
                    from_index = i
                    break
            if from_index is None:
                raise ValueError("No FROM clause found")
            
            insert_index = from_index + 1
            for i in range(from_index + 1, len(parsed.tokens)):
                if not parsed.tokens[i].value.upper().startswith('JOIN'):
                    insert_index = i
                    break
            
            # Apply WHERE clause
            if where_clause:
                new_where = f"{where_clause} AND {new_where_condition}"
                tokens = parsed.tokens[:where_index] + [new_where] + parsed.tokens[where_index + 1:]
            else:
                new_where = f"WHERE {new_where_condition}"
                tokens = parsed.tokens[:insert_index] + parsed.tokens[insert_index:]
                insert_index = len(tokens)  # Append WHERE at the end
            
            # Insert JOINs
            tokens = (
                tokens[:insert_index] +
                [join_clause for join_clause in join_clauses] +
                tokens[insert_index:]
            )
            
            # Reconstruct query
            extended_query = "".join(str(token) for token in tokens)
            logging.debug(f'Returning extended query: {extend_query}')
            
            return extended_query
        

        extended_query = extend_query(query=query, project_ids=[4, 5])
        return self.db.run_no_throw(extended_query)