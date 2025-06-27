from typing import List, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, ConfigDict
from parameters import table_info
from langchain_community.utilities.sql_database import SQLDatabase
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


class _CustomInfoSQLDatabaseToolInput(BaseModel):
    table_names: List[str] = Field(..., description='List of names of tables for which to get descriptions, schema and relationships')


class CustomInfoSQLDatabaseTool(BaseTool):
    name: str = 'SchemaGetter'
    description: str = 'Use to decide which tables to use, and at the same time get the schema for the chosen tables.'
    args_schema: Type[BaseModel] = _CustomInfoSQLDatabaseToolInput
    return_direct: bool = False

    def _run(
        self, table_names: List[str], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        '''
            Retrieves schema metadata for specified database tables. Input: a list of table names. Output: a list of dictionaries, each representing a table with:
                1. name (string): Table name in the database.
                2. description (string): Table purpose and usage notes.
                3. columns (list): List of column dictionaries, each with:
                    - name (string): Column name in the database.
                    - description (string): Column purpose, usage, and references to other tables.
                4. relationships (list): List of relationship dictionaries, each with:
                    - column (string): Column in this table referencing another table.
                    - referenced_table (string): Name of the referenced table.
                    - referenced_column (string): Name of the referenced column.
            Use this tool to select relevant tables and obtain their schema for query construction.
        '''
        return [table for table in table_info if table['name'] in table_names]