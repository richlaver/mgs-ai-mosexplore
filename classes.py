"""Defines state classes for the MissionOSExplore Demo application.

This module provides a typed dictionary for structuring the application's state
in the LangGraph workflow.
"""

from typing import List
from langgraph.graph import MessagesState


class State(MessagesState):
    """State class for managing conversation and multimedia in LangGraph.

    Attributes:
        messages: List of conversation messages (human, AI, or tool).
        timings: List of dictionaries with timing details for nodes and components.
    """
    is_valid_request: bool
    timings: List[dict]
    

# class InstrTypeToolInput00(BaseModel):
#     instr_ids: List[str] = Field(description='List of instrument IDs for which to get type information')
#     db: Annotated[SQLDatabase, InjectedToolArg] = Field(description='SQL database instance to query for instrument type information')
#     llm: Annotated[BaseLanguageModel, InjectedToolArg] = Field(description='Language model instance for processing instrument type information')

#     class Config:
#         arbitrary_types_allowed = True


# class InstrTypeTool00(BaseTool):
#     name: str = 'InstrTypeGetter'
#     description: str = 'Use to find the type of an instrument and information related to the type.'
#     args_schema: Optional[ArgsSchema] = InstrTypeToolInput00
#     return_direct: bool = False

#     def get_field_metadata(instr_ids: list, db: SQLDatabase) -> dict:
#         """
#         Retrieve text and unit fields for given instrument IDs.
        
#         Args:
#             instr_ids (list): List of instrument IDs to query.
#             db (SQLDatabase): SQL database.
        
#         Returns:
#             dict: JSON-serializable dictionary mapping instr_id to a dictionary with
#                 'text' (list of concatenated name/label/description) and
#                 'unit' (list of units).
#         """
#         query = f"""
#         SELECT i.instr_id, t.user_field_name, t.user_label, t.user_description, t.units
#         FROM instrum i
#         JOIN type_config_normalized t ON i.object_ID = t.object_ID
#         WHERE i.instr_id IN ('{"', '".join(instr_ids)}')
#         AND t.user_field_name <> t.field_name
#         AND t.field_name NOT IN ('taken_on', 'remarks')
#         ORDER BY i.instr_id
#         """
#         result = db.run(query)
#         parsed_result = ast.literal_eval(result)
#         output = defaultdict(lambda: {"description": "", "unit": ""})
#         for row in parsed_result:
#             instr_id = str(row[0])
#             description_parts = [part or "" for part in row[1:4]]
#             description = " ".join(part.strip() for part in description_parts if part.strip())
#             unit = row[4] or ""
#             output[instr_id]["description"] += (description + " ") if description else ""
#             output[instr_id]["unit"] += (unit + " ") if unit else ""
#         for instr_id in output:
#             output[instr_id]["description"] = output[instr_id]["description"].rstrip()
#             output[instr_id]["unit"] = output[instr_id]["unit"].rstrip()
        
#         return dict(output)


#     def _run(
#         self,
#         instr_ids: List[str],
#         db: SQLDatabase,
#         llm: BaseLanguageModel,
#         run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> List[dict]:
#         """Use the tool."""
#         field_metadata = self.get_field_metadata(instr_ids, db)
#         instr_type_context = context1['instrument_types']
#         instr_ids_string = ", ".join(instr_ids)
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              """
#              You are an agent designed to find the types of the following instruments 
#              and information related to the types:
#              {instr_ids_string}

#              ALWAYS begin by referring to the following JSON data for information on 
#              the context of instrument types:
#              {instr_type_context}
#              The JSON maps instrument type names e.g. settlement marker to descriptions 
#              of the instrument types and their data fields.

#              Information on the data fields of the instruments is provided in the 
#              following JSON data:
#              {field_metadata}
#              The JSON maps instrument IDs to a dictionary with the keys:
#              - 'description' -> words found in descriptions from all instrument fields
#              - 'unit' -> measurement units from all instrument fields
             
#              Once you have understood the instrument type context, go through each 
#              instrument ID and using information on the data fields of that instrument, 
#              consider the likelihood of the instrument belonging to each instrument type.
#              After evaluating the likelihoods, determine the most likely instrument type 
#              for each instrument ID.

#              Return a JSON object mapping each instrument ID to its most likely type e.g. 
#              settlement marker.
#              The JSON should look like this:
#              {{
#                  "instr_id_1": "most_likely_type_1",
#                  "instr_id_2": "most_likely_type_2",
#                  ...
#              }}
#              """).format(
#                  instr_ids_string=instr_ids_string,
#                  instr_type_context=json.dumps(instr_type_context),
#                  field_metadata=json.dumps(field_metadata)
#              )
#         ])
#         llm_with_structured_output = llm.with_structured_output(dict)
#         result = llm_with_structured_output.invoke(
#             prompt.invoke({"instr_ids_string": instr_ids_string,
#                            "instr_type_context": json.dumps(instr_type_context),
#                            "field_metadata": json.dumps(field_metadata)})
#         )
#         return json.dumps(result)


