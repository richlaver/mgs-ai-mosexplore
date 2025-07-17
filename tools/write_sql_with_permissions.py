from typing import Any, Dict, Optional, Sequence, Type, Union, List, Tuple

from sqlalchemy.engine import Result

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
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


class BaseUserPermissionsTool(BaseModel):
    """Base tool for applying with user permissions."""

    table_relationship_graph: Dict[str, List[Tuple]] = Field(exclude=True)
    user_id: int = Field(exclude=True)
    global_hierarchy_access: bool = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _CustomQuerySQLDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        def get_all_tables(query):
            """Extract all table names and their aliases from FROM and JOIN clauses."""
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
            """Dynamically rewrite the query to add JOIN and WHERE clauses for user permissions."""
            logging.debug(f'Original query: {query}')
            
            table_aliases = get_all_tables(query)
            if not table_aliases:
                logging.debug('Didn\'t find any tables in the query. Returning the original query.')
                return query

            # Check if location, contracts, projects, or sites tables are in the query
            location_alias = None
            contracts_alias = None
            projects_alias = None
            sites_alias = None
            for table_name, alias in table_aliases:
                if table_name == "location":
                    location_alias = alias
                elif table_name == "contracts":
                    contracts_alias = alias
                elif table_name == "projects":
                    projects_alias = alias
                elif table_name == "sites":
                    sites_alias = alias

            parsed = sqlparse.parse(query)[0]
            where_clause = None
            where_index = None
            for i, token in enumerate(parsed.tokens):
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'WHERE':
                    where_clause = token
                    where_index = i
                    for j in range(i + 1, len(parsed.tokens)):
                        if parsed.tokens[j].ttype is sqlparse.tokens.Keyword and parsed.tokens[j].value.upper() in ('GROUP', 'ORDER', 'LIMIT'):
                            break
                        where_clause = ''.join(str(t) for t in parsed.tokens[i:j + 1])
                    break

            tokens = list(parsed.tokens)
            new_where_condition = (
                f"u.id = {self.user_id} AND uagu.user_deleted = 0 AND u.prohibit_portal_access NOT IN (1, 2, 3)"
            )
            base_permission_joins = [
                " JOIN user_access_groups_permissions uagp ON (p.id = uagp.project OR uagp.project = 0) AND (c.id = uagp.contract OR uagp.contract = 0) AND (s.id = uagp.site OR uagp.site = 0) ",
                " JOIN user_access_groups uag ON uagp.user_group_id = uag.id ",
                " JOIN user_access_groups_users uagu ON uag.id = uagu.group_id ",
                " JOIN geo_12_users u ON uagu.user_id = u.id "
            ]

            # Define valid target tables that exist in the table_relationship_graph
            valid_targets = [t for t in ["location", "contracts", "projects", "sites"] if t in self.table_relationship_graph]

            if location_alias:
                # Location is in the query
                permission_joins = [
                    f" LEFT JOIN projects p ON {location_alias}.project_id = p.id ",
                    f" LEFT JOIN contracts c ON {location_alias}.contract_id = c.id ",
                    f" LEFT JOIN sites s ON {location_alias}.site_id = s.id ",
                ] + base_permission_joins
                if where_clause:
                    new_where = f"{where_clause} AND ({new_where_condition})"
                    tokens = tokens[:where_index] + [new_where] + tokens[where_index + 1:]
                else:
                    new_where = f" WHERE {new_where_condition}"
                    tokens = tokens + [new_where]
                tokens = tokens + permission_joins
            elif contracts_alias:
                # Contracts is in the query
                permission_joins = [
                    f" LEFT JOIN projects p ON {contracts_alias}.project_id = p.id ",
                    f" LEFT JOIN location ON {contracts_alias}.id = location.contract_id ",
                    f" LEFT JOIN sites s ON location.site_id = s.id ",
                ] + base_permission_joins
                if where_clause:
                    new_where = f"{where_clause} AND ({new_where_condition})"
                    tokens = tokens[:where_index] + [new_where] + tokens[where_index + 1:]
                else:
                    new_where = f" WHERE {new_where_condition}"
                    tokens = tokens + [new_where]
                tokens = tokens + permission_joins
            elif projects_alias:
                # Projects is in the query
                permission_joins = [
                    f" LEFT JOIN contracts c ON {projects_alias}.id = c.project_id ",
                    f" LEFT JOIN location ON c.id = location.contract_id ",
                    f" LEFT JOIN sites s ON location.site_id = s.id ",
                ] + base_permission_joins
                if where_clause:
                    new_where = f"{where_clause} AND ({new_where_condition})"
                    tokens = tokens[:where_index] + [new_where] + tokens[where_index + 1:]
                else:
                    new_where = f" WHERE {new_where_condition}"
                    tokens = tokens + [new_where]
                tokens = tokens + permission_joins
            elif sites_alias:
                # Sites is in the query
                permission_joins = [
                    f" LEFT JOIN contracts c ON {sites_alias}.contract_id = c.id ",
                    f" LEFT JOIN projects p ON c.project_id = p.id ",
                    f" LEFT JOIN location ON {sites_alias}.id = location.site_id ",
                ] + base_permission_joins
                if where_clause:
                    new_where = f"{where_clause} AND ({new_where_condition})"
                    tokens = tokens[:where_index] + [new_where] + tokens[where_index + 1:]
                else:
                    new_where = f" WHERE {new_where_condition}"
                    tokens = tokens + [new_where]
                tokens = tokens + permission_joins
            else:
                # Find path to location, contracts, projects, or sites
                min_distance = float('inf')
                selected_path = None
                selected_alias = None
                target_table = None
                for table_name, alias in table_aliases:
                    for target in valid_targets:
                        path = find_shortest_path(table_name, target)
                        if path and len(path) - 1 < min_distance:
                            min_distance = len(path) - 1
                            selected_path = path
                            selected_alias = alias
                            target_table = target

                if not selected_path:
                    logging.debug('No table connects to location, contracts, projects, or sites in table_relationship_graph. Returning original query.')
                    return query

                join_clauses = []
                prev_table = selected_alias
                for table, src_col, tgt_col in selected_path[1:]:
                    join_clauses.append(f" LEFT JOIN {table} ON {prev_table}.{src_col} = {table}.{tgt_col} ")
                    prev_table = table

                if target_table == "location":
                    join_clauses.extend([
                        f" LEFT JOIN projects p ON {prev_table}.project_id = p.id ",
                        f" LEFT JOIN contracts c ON {prev_table}.contract_id = c.id ",
                        f" LEFT JOIN sites s ON {prev_table}.site_id = s.id ",
                    ] + base_permission_joins)
                elif target_table == "contracts":
                    join_clauses.extend([
                        f" LEFT JOIN projects p ON {prev_table}.project_id = p.id ",
                        f" LEFT JOIN location ON {prev_table}.id = location.contract_id ",
                        f" LEFT JOIN sites s ON location.site_id = s.id ",
                    ] + base_permission_joins)
                elif target_table == "projects":
                    join_clauses.extend([
                        f" LEFT JOIN contracts c ON {prev_table}.id = c.project_id ",
                        f" LEFT JOIN location ON c.id = location.contract_id ",
                        f" LEFT JOIN sites s ON location.site_id = s.id ",
                    ] + base_permission_joins)
                elif target_table == "sites":
                    join_clauses.extend([
                        f" LEFT JOIN contracts c ON {prev_table}.contract_id = c.id ",
                        f" LEFT JOIN projects p ON c.project_id = p.id ",
                        f" LEFT JOIN location ON {prev_table}.id = location.site_id ",
                    ] + base_permission_joins)

                from_index = None
                table_index = None
                for i, token in enumerate(parsed.tokens):
                    if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                        from_index = i
                        for j in range(i + 1, len(parsed.tokens)):
                            if isinstance(tokens[j], sqlparse.sql.Identifier):
                                table_index = j
                                break
                            elif isinstance(tokens[j], sqlparse.sql.IdentifierList):
                                table_index = j
                                break
                            elif tokens[j].ttype is sqlparse.tokens.Keyword and tokens[j].value.upper().startswith('JOIN'):
                                break
                        break
                if from_index is None or table_index is None:
                    logging.error("Invalid FROM clause in query")
                    raise ValueError("Invalid FROM clause")

                insert_index = table_index + 1
                logging.debug(f'Inserting join clauses at index {insert_index}: {join_clauses}')
                tokens = (
                    tokens[:insert_index] +
                    [join_clause for join_clause in join_clauses] +
                    tokens[insert_index:]
                )

                if where_clause:
                    new_where = f"{where_clause} AND ({new_where_condition})"
                    tokens = tokens[:where_index] + [new_where] + tokens[where_index + 1:]
                else:
                    new_where = f" WHERE {new_where_condition}"
                    tokens = tokens + [new_where]

            extended_query = "".join(str(token) for token in tokens)
            logging.debug(f'Returning extended query: {extended_query}')
            return extended_query

        if self.global_hierarchy_access:
            logging.debug("User has global hierarchy access, returning original query.")
            return self.db.run_no_throw(query)
        try:
            extended_query = extend_query(query=query)
            return self.db.run_no_throw(extended_query)
        except Exception as e:
            logging.error(f"Error extending query: {str(e)}")
            raise