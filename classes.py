"""Defines state classes for the MissionOSExplore Demo application.

This module provides a typed dictionary for structuring the application's state
in the LangGraph workflow.
"""

from typing import List, TypedDict, Optional, Dict
from typing_extensions import Annotated
from langgraph.graph import MessagesState
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from pydantic import BaseModel, Field
from parameters import table_info
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import InjectedToolArg
import ast
import json
from collections import defaultdict
from parameters import context1
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


class State(MessagesState):
    """State class for managing conversation and multimedia in LangGraph.

    Attributes:
        messages: List of conversation messages (human, AI, or tool).
        timings: List of dictionaries with timing details for nodes and components.
    """
    is_valid_request: bool
    timings: List[dict]


# class UserPermissionsToolInput(BaseModel):
#     user_id: int = Field(description='Unique identifier for user')
#     db: Annotated[SQLDatabase, InjectedToolArg] = Field(description='SQL database instance to query for instrument type information')

#     class Config:
#         arbitrary_types_allowed = True


# class UserPermissionsTool(BaseTool):
#     name: str = 'UserPermissionsGetter',
#     description: str = 'Use to find what projects, contracts and sites the user is allowed to access.'
#     args_schema: Optional[ArgsSchema] = UserPermissionsToolInput
#     return_direct: bool = False
    
#     def _run(
#         self,
#         user_id: int,
#         db: SQLDatabase,
#         run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> List[dict]:
#         query = """
#             SELECT 
#                 gu.id AS user_id,
#                 gu.username,
#                 gu.name AS user_name,
#                 gu.prohibit_portal_access,
#                 mut.name AS user_type_name,
#                 uagu.group_id,
#                 uagu.user_deleted,
#                 uag.group_name,
#                 p.id AS project_id,
#                 p.name AS project_name,
#                 c.id AS contract_id,
#                 c.name AS contract_name,
#                 s.id AS site_id,
#                 s.name AS site_name
#             FROM 
#                 geo_12_users gu
#                 INNER JOIN mg_user_types mut ON gu.user_type = mut.id
#                 INNER JOIN user_access_groups_users uagu ON gu.id = uagu.user_id
#                 LEFT JOIN user_access_groups uag ON uagu.group_id = uag.id AND uagu.group_id != 0
#                 LEFT JOIN user_access_groups_permissions uagp ON uag.id = uagp.user_group_id
#                 LEFT JOIN projects p ON uagp.project = p.id
#                 LEFT JOIN contracts c ON uagp.contract = c.id
#                 LEFT JOIN sites s ON uagp.site = s.id
#             WHERE 
#                 gu.id = {user_id};
#         """.format(user_id = user_id)
#         execute_query = QuerySQLDatabaseTool(db=db)
#         result = execute_query.invoke({'query': query})
#         user_permission_data = [{
#             'user_id': row[0],
#             'username': row[1],
#             'user_name': row[2],
#             'prohibit_portal_access': row[3],
#             'user_type_name': row[4],
#             'group_id': row[5],
#             'user_deleted': row[6],
#             'group_name': row[7],
#             'project_id': row[8],
#             'project_name': row[9],
#             'contract_id': row[10],
#             'contract_name': row[11],
#             'site_id': row[12],
#             'site_name': row[13]
#         } for row in result]
#         return user_permission_data


class CustomInfoSQLDatabaseToolInput(BaseModel):
    table_names: List[str] = Field(description='List of names of tables for which to get descriptions, schema and relationships')


class CustomInfoSQLDatabaseTool(BaseTool):
    name: str = 'SchemaGetter'
    description: str = 'Use to decide which tables to use, and at the same time get the schema for the chosen tables.'
    args_schema: Optional[ArgsSchema] = CustomInfoSQLDatabaseToolInput
    return_direct: bool = False

    def _run(
        self, table_names: List[str], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        """Use the tool."""
        return [table for table in table_info if table['name'] in table_names]
    

class InstrTypeToolInput00(BaseModel):
    instr_ids: List[str] = Field(description='List of instrument IDs for which to get type information')
    db: Annotated[SQLDatabase, InjectedToolArg] = Field(description='SQL database instance to query for instrument type information')
    llm: Annotated[BaseChatModel, InjectedToolArg] = Field(description='Language model instance for processing instrument type information')

    class Config:
        arbitrary_types_allowed = True


class InstrTypeTool00(BaseTool):
    name: str = 'InstrTypeGetter'
    description: str = 'Use to find the type of an instrument and information related to the type.'
    args_schema: Optional[ArgsSchema] = InstrTypeToolInput00
    return_direct: bool = False

    def get_field_metadata(instr_ids: list, db: SQLDatabase) -> dict:
        """
        Retrieve text and unit fields for given instrument IDs.
        
        Args:
            instr_ids (list): List of instrument IDs to query.
            db (SQLDatabase): SQL database.
        
        Returns:
            dict: JSON-serializable dictionary mapping instr_id to a dictionary with
                'text' (list of concatenated name/label/description) and
                'unit' (list of units).
        """
        query = f"""
        SELECT i.instr_id, t.user_field_name, t.user_label, t.user_description, t.units
        FROM instrum i
        JOIN type_config_normalized t ON i.object_ID = t.object_ID
        WHERE i.instr_id IN ('{"', '".join(instr_ids)}')
        AND t.user_field_name <> t.field_name
        AND t.field_name NOT IN ('taken_on', 'remarks')
        ORDER BY i.instr_id
        """
        result = db.run(query)
        parsed_result = ast.literal_eval(result)
        output = defaultdict(lambda: {"description": "", "unit": ""})
        for row in parsed_result:
            instr_id = str(row[0])
            description_parts = [part or "" for part in row[1:4]]
            description = " ".join(part.strip() for part in description_parts if part.strip())
            unit = row[4] or ""
            output[instr_id]["description"] += (description + " ") if description else ""
            output[instr_id]["unit"] += (unit + " ") if unit else ""
        for instr_id in output:
            output[instr_id]["description"] = output[instr_id]["description"].rstrip()
            output[instr_id]["unit"] = output[instr_id]["unit"].rstrip()
        
        return dict(output)


    def _run(
        self,
        instr_ids: List[str],
        db: SQLDatabase,
        llm: BaseChatModel,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        """Use the tool."""
        field_metadata = self.get_field_metadata(instr_ids, db)
        instr_type_context = context1['instrument_types']
        instr_ids_string = ", ".join(instr_ids)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
             You are an agent designed to find the types of the following instruments 
             and information related to the types:
             {instr_ids_string}

             ALWAYS begin by referring to the following JSON data for information on 
             the context of instrument types:
             {instr_type_context}
             The JSON maps instrument type names e.g. settlement marker to descriptions 
             of the instrument types and their data fields.

             Information on the data fields of the instruments is provided in the 
             following JSON data:
             {field_metadata}
             The JSON maps instrument IDs to a dictionary with the keys:
             - 'description' -> words found in descriptions from all instrument fields
             - 'unit' -> measurement units from all instrument fields
             
             Once you have understood the instrument type context, go through each 
             instrument ID and using information on the data fields of that instrument, 
             consider the likelihood of the instrument belonging to each instrument type.
             After evaluating the likelihoods, determine the most likely instrument type 
             for each instrument ID.

             Return a JSON object mapping each instrument ID to its most likely type e.g. 
             settlement marker.
             The JSON should look like this:
             {{
                 "instr_id_1": "most_likely_type_1",
                 "instr_id_2": "most_likely_type_2",
                 ...
             }}
             """).format(
                 instr_ids_string=instr_ids_string,
                 instr_type_context=json.dumps(instr_type_context),
                 field_metadata=json.dumps(field_metadata)
             )
        ])
        llm_with_structured_output = llm.with_structured_output(dict)
        result = llm_with_structured_output.invoke(
            prompt.invoke({"instr_ids_string": instr_ids_string,
                           "instr_type_context": json.dumps(instr_type_context),
                           "field_metadata": json.dumps(field_metadata)})
        )
        return json.dumps(result)


