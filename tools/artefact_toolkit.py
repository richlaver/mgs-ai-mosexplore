from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import b2sdk.v1 as b2
import psycopg2.extensions
import artefact_management
import datetime
import logging
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)
class WriteArtefactInput(BaseModel):
    blob: bytes | str = Field(
        description="The artefact itself (as bytes or string; strings will be encoded to UTF-8 bytes)."
    )
    thread_id: str = Field(
        description="ID of the thread associated with the artefact."
    )
    user_id: int = Field(
        description="Integer ID of the user creating the artefact."
    )
    generating_tool: str = Field(
        description="Name of the tool that generated the artefact."
    )
    generating_parameters: Dict[str, Any] = Field(
        description="Dictionary of parameters used by the generating tool."
    )
    description: str = Field(
        description="Concise description of the artefact."
    )

class WriteArtefactOutput(BaseModel):
    artefact_id: Optional[str] = Field(
        description="ID of the artefact if successfully written, else None."
    )
    error: Optional[str] = Field(
        description="Error message if the write failed, else None."
    )

class WriteArtefactTool(BaseTool):
    name: str = "write_artefact_tool"
    description: str = (
        "Writes an artefact blob to Backblaze B2 and stores its metadata in PostgreSQL RDS.\n\n"
        "Input must be a dictionary matching the WriteArtefactInput schema:\n"
        "- 'blob': bytes or str (required) - The artefact data; strings are auto-encoded to bytes.\n"
        "- 'thread_id': str (required) - Thread ID.\n"
        "- 'user_id': int (required) - Integer user ID.\n"
        "- 'generating_tool': str (required) - Tool name that generated it.\n"
        "- 'generating_parameters': dict (required) - Parameters used for generation.\n"
        "- 'description': str (required) - Concise description.\n\n"
        "Example input dict:\n"
        "{'blob': 'CSV content here', 'thread_id': 'thread-123', 'user_id': 456, "
        "'generating_tool': 'timeseries_plot_tool', 'generating_parameters': {'param1': 'value1'}, "
        "'description': 'Timeseries data CSV'}\n\n"
        "Output is a dict matching WriteArtefactOutput:\n"
        "- 'artefact_id': str or None\n"
        "- 'error': str or None"
    )
    args_schema: type[BaseModel] = WriteArtefactInput
    blob_db: b2.Bucket
    metadata_db: psycopg2.extensions.connection

    def _run(self, blob: bytes | str, thread_id: str, user_id: int, generating_tool: str, generating_parameters: Dict[str, Any], description: str) -> Dict[str, Any]:
        return artefact_management.write_artefact(
            blob_db=self.blob_db,
            metadata_db=self.metadata_db,
            blob=blob,
            thread_id=thread_id,
            user_id=user_id,
            generating_tool=generating_tool,
            generating_parameters=generating_parameters,
            description=description
        )

class ReadArtefactsInput(BaseModel):
    metadata_only: bool = Field(
        description="True to return only metadata, False to include blobs."
    )
    artefact_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of artefact IDs to filter."
    )
    thread_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of thread IDs to filter."
    )
    user_ids: Optional[List[int]] = Field(
        default=None,
        description="Optional list of integer user IDs to filter."
    )
    generating_tools_parameters: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of dicts with 'tool' (str) and optional 'parameters' (dict)."
    )
    description_keywords: Optional[List[str]] = Field(
        default=None,
        description="Optional list of keywords for vector search on descriptions."
    )
    start_time: Optional[datetime.datetime] = Field(
        default=None,
        description="Optional start timestamp for filtering (requires end_time)."
    )
    end_time: Optional[datetime.datetime] = Field(
        default=None,
        description="Optional end timestamp for filtering (requires start_time)."
    )

class ReadArtefactsOutput(BaseModel):
    artefacts: List[Dict[str, Any]] = Field(
        description="List of dicts with 'blob' (bytes or None) and 'metadata' (dict)."
    )
    success: bool = Field(
        description="True if successful."
    )
    error: Optional[str] = Field(
        description="Error message if any."
    )

class ReadArtefactsTool(BaseTool):
    name: str = "read_artefacts_tool"
    description: str = (
        "Reads artefacts from Backblaze B2 and RDS based on filters. Returns all if no filters.\n"
        "Filters use union within lists (OR) and intersection across types (AND).\n\n"
        "Input must be a dictionary matching the ReadArtefactsInput schema:\n"
        "- 'metadata_only': bool (required) - True for metadata only.\n"
        "- 'artefact_ids': list[str] (optional) - Artefact IDs.\n"
        "- 'thread_ids': list[str] (optional) - Thread IDs.\n"
        "- 'user_ids': list[int] (optional) - Integer user IDs.\n"
        "- 'generating_tools_parameters': list[dict] (optional) - Each dict: {'tool': str, 'parameters': dict (optional)}.\n"
        "- 'description_keywords': list[str] (optional) - Keywords for semantic search.\n"
        "- 'start_time': datetime (optional) - Start timestamp (needs 'end_time').\n"
        "- 'end_time': datetime (optional) - End timestamp (needs 'start_time').\n\n"
        "Example input dict:\n"
        "{'metadata_only': True, 'user_ids': [456], "
        "'generating_tools_parameters': [{'tool': 'timeseries_plot_tool'}], "
        "'description_keywords': ['timeseries', 'plot']}\n\n"
        "Output is a dict matching ReadArtefactsOutput:\n"
        "- 'artefacts': list[dict] - Each with 'blob' and 'metadata'.\n"
        "- 'success': bool\n"
        "- 'error': str or None"
    )
    args_schema: type[BaseModel] = ReadArtefactsInput
    blob_db: b2.Bucket
    metadata_db: psycopg2.extensions.connection

    def _run(
        self,
        metadata_only: bool,
        artefact_ids: Optional[List[str]] = None,
        thread_ids: Optional[List[str]] = None,
        user_ids: Optional[List[int]] = None,
        generating_tools_parameters: Optional[List[Dict[str, Any]]] = None,
        description_keywords: Optional[List[str]] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        return artefact_management.read_artefacts(
            blob_db=self.blob_db,
            metadata_db=self.metadata_db,
            metadata_only=metadata_only,
            artefact_ids=artefact_ids,
            thread_ids=thread_ids,
            user_ids=user_ids,
            generating_tools_parameters=generating_tools_parameters,
            description_keywords=description_keywords,
            start_time=start_time,
            end_time=end_time
        )

class DeleteArtefactsInput(BaseModel):
    artefact_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of artefact IDs to filter for deletion."
    )
    thread_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of thread IDs to filter for deletion."
    )
    user_ids: Optional[List[int]] = Field(
        default=None,
        description="Optional list of integer user IDs to filter for deletion."
    )
    generating_tools: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tool names to filter for deletion."
    )
    start_time: Optional[datetime.datetime] = Field(
        default=None,
        description="Optional start timestamp for filtering (requires end_time)."
    )
    end_time: Optional[datetime.datetime] = Field(
        default=None,
        description="Optional end timestamp for filtering (requires start_time)."
    )

class DeleteArtefactsOutput(BaseModel):
    artefact_ids: List[str] = Field(
        description="List of successfully deleted artefact IDs."
    )
    error: Optional[str] = Field(
        description="Error message if any (partial failures aggregated)."
    )

class DeleteArtefactsTool(BaseTool):
    name: str = "delete_artefacts_tool"
    description: str = (
        "Deletes artefacts from Backblaze B2 and RDS based on filters. Deletes all if no filters.\n"
        "Filters use union within lists (OR) and intersection across types (AND).\n\n"
        "Input must be a dictionary matching the DeleteArtefactsInput schema:\n"
        "- 'artefact_ids': list[str] (optional) - Artefact IDs.\n"
        "- 'thread_ids': list[str] (optional) - Thread IDs.\n"
        "- 'user_ids': list[int] (optional) - Integer user IDs.\n"
        "- 'generating_tools': list[str] (optional) - Tool names.\n"
        "- 'start_time': datetime (optional) - Start timestamp (needs 'end_time').\n"
        "- 'end_time': datetime (optional) - End timestamp (needs 'start_time').\n\n"
        "Example input dict:\n"
        "{'user_ids': [456], 'generating_tools': ['timeseries_plot_tool'], "
        "'start_time': datetime.datetime(2023, 1, 1), 'end_time': datetime.datetime(2023, 12, 31)}\n\n"
        "Output is a dict matching DeleteArtefactsOutput:\n"
        "- 'artefact_ids': list[str] - Deleted IDs.\n"
        "- 'error': str or None"
    )
    args_schema: type[BaseModel] = DeleteArtefactsInput
    blob_db: b2.Bucket
    metadata_db: psycopg2.extensions.connection

    def _run(
        self,
        artefact_ids: Optional[List[str]] = None,
        thread_ids: Optional[List[str]] = None,
        user_ids: Optional[List[int]] = None,
        generating_tools: Optional[List[str]] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        return artefact_management.delete_artefacts(
            blob_db=self.blob_db,
            metadata_db=self.metadata_db,
            artefact_ids=artefact_ids,
            thread_ids=thread_ids,
            user_ids=user_ids,
            generating_tools=generating_tools,
            start_time=start_time,
            end_time=end_time
        )