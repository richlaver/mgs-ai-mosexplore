from pydantic import BaseModel, Field
from typing import List, Optional, Type
from langchain_core.tools import BaseTool
import json
import re
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


class DataFieldsDict(BaseModel):
    field_name: str = Field(description="""
The name of the data field in the database.""")
    field_type: str = Field(description="""
The type of the data field, e.g., `data`, `calc`.""")
    field_units: Optional[str] = Field(default=None, description="""
The units of the data field, if known.""")
    
class DataSourcesDict(BaseModel):
    instrument_type: str = Field(description="""
The type of instrument as stored in column `type1` of table `instrum` in the 
database.""")
    instrument_subtype: str = Field(description="""
The subtype of instrument as stored in column `subtype1` of table `instrum` in 
the database.""")
    data_fields: List[DataFieldsDict] = Field(description="""
A list of data fields to reference in table `mydata` in the database to 
retrieve readings.""")

class QueryWords(BaseModel):
    query_words: str = Field(description="""
Words in the user query pertaining to a particular set of instrument types, 
subtypes and data fields.""")
    data_sources: List[DataSourcesDict] = Field(description="""
A list of dictionaries detailing the instrument type, subtype and data fields
to reference in the database that relate to words in the user query.""")
    
class InstrumentContextToolOutput(BaseModel):
    query_words: List[QueryWords] = Field(description="""
A list of dictionaries containing words in the user query pertaining to
instruments and their readings, 
along with instrument types, subtypes and data fields implied by those words.
""")
    
class InstrumentContextTool(BaseTool):
    """A tool to extract instrument context from a user query."""
    name: str = "get_instrument_context"
    description: str = """
    Extracts instrument context from a user query.
    Input:
    - A string containing the user's query about instruments and their readings
    Returns:
    - InstrumentContextToolOutput containing matched instruments and their data fields.
    Example: "show me all vibration readings"
    """
    
    def _run(
        self,
        query: str
    ) -> InstrumentContextToolOutput:
        """Run the tool to extract instrument context from a query."""
        if 'settlement' in query.lower():
            return InstrumentContextToolOutput(
                query_words=[
                    QueryWords(
                        query_words="settlement",
                        data_sources=[
                            DataSourcesDict(
                                instrument_type="LP",
                                instrument_subtype="MOVEMENT",
                                data_fields=[
                                    DataFieldsDict(
                                        field_name="calculation1",
                                        field_type="calc",
                                        field_units="mm"
                                    )
                                ]
                            ),
                            DataSourcesDict(
                                instrument_type="OT",
                                instrument_subtype="MOVEMENT",
                                data_fields=[
                                    DataFieldsDict(
                                        field_name="data1",
                                        field_type="data",
                                        field_units="mm"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        if all(word in query.lower() for word in ['groundwater', 'level']):
            return InstrumentContextToolOutput(
                query_words=[
                    QueryWords(
                        query_words="groundwater level",
                        data_sources=[
                            DataSourcesDict(
                                instrument_type="CASA",
                                instrument_subtype="DEFAULT",
                                data_fields=[
                                    DataFieldsDict(
                                        field_name="calculation1",
                                        field_type="calc",
                                        field_units="m"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        if 'vibration' in query.lower():
            return InstrumentContextToolOutput(
                query_words=[
                    QueryWords(
                        query_words="vibration",
                        data_sources=[
                            DataSourcesDict(
                                instrument_type="VIBR",
                                instrument_subtype="GSS",
                                data_fields=[
                                    DataFieldsDict(
                                        field_name="data4",
                                        field_type="data",
                                        field_units="mm/s"
                                    ),
                                    DataFieldsDict(
                                        field_name="data5",
                                        field_type="data",
                                        field_units="mm/s"
                                    ),
                                    DataFieldsDict(
                                        field_name="data6",
                                        field_type="data",
                                        field_units="mm/s"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
