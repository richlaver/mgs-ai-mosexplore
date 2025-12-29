import datetime
import json
import uuid
import numpy as np
import logging

import psycopg2
from b2sdk.v1 import DownloadDestBytes, AbstractProgressListener
from langchain_google_vertexai import VertexAIEmbeddings
from utils.run_cancellation import get_active_run_controller, RunCancelledError

TABLE_NAME = "mgs-explore-specificquery-artefacts"
VECTOR_DIM = 768
SIMILARITY_THRESHOLD = 0.7
COSINE_DISTANCE_THRESHOLD = 1 - SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

def _get_embeddings():
    """
    Initialize VertexAI embeddings. Assumes Google credentials are set via os.environ["GOOGLE_APPLICATION_CREDENTIALS"].
    Project ID should be set appropriately; here assuming it's configured externally.
    """
    return VertexAIEmbeddings(model="text-embedding-004")  # Dimension: 768

def _vector_to_str(vec):
    """Convert vector list to PostgreSQL-compatible string for insertion/query."""
    return '[' + ','.join(map(str, vec)) + ']'

def _delete_all_versions_by_name(blob_db, file_name: str) -> bool:
    """Delete all versions of a B2 file by listing all versions for exact file_name.
    Returns True if at least one version was deleted.
    """
    versions: list[tuple[str, str]] = []
    try:
        for file_version, _ in blob_db.ls(file_name, show_versions=True, recursive=True):
            if getattr(file_version, 'file_name', None) == file_name:
                file_id = getattr(file_version, 'id_', None) or getattr(file_version, 'file_id', None)
                if file_id:
                    versions.append((file_id, file_name))
    except Exception:
        pass

    if not versions:
        try:
            for file_version, _ in blob_db.ls('', show_versions=True, recursive=True):
                if getattr(file_version, 'file_name', None) == file_name:
                    file_id = getattr(file_version, 'id_', None) or getattr(file_version, 'file_id', None)
                    if file_id:
                        versions.append((file_id, file_name))
        except Exception as e:
            logger.error("Error listing versions for '%s': %s", file_name, e)

    seen = set()
    unique_versions = []
    for fid, fname in versions:
        if fid not in seen:
            seen.add(fid)
            unique_versions.append((fid, fname))

    deleted_any = False
    for fid, fname in unique_versions:
        try:
            blob_db.delete_file_version(fid, fname)
            deleted_any = True
        except Exception as e:
            msg = str(e).lower()
            if 'not found' in msg or 'no such file' in msg or 'does not exist' in msg:
                continue
            logger.error("Error deleting version '%s' for '%s': %s", fid, file_name, e)
    return deleted_any

def write_artefact(blob_db, metadata_db, blob, thread_id, user_id: int, generating_tool, generating_parameters, description):
    """
    Writes the blob to Backblaze B2, generates metadata including embedded vector, and stores it in RDS.
    """
    try:
        controller = get_active_run_controller()
        if controller:
            controller.raise_if_cancelled("artefact:prepare")

        class _CancellationAwareListener(AbstractProgressListener):
            def __init__(self, description: str = "upload"):
                super().__init__(description=description)
                self._controller = controller

            def set_total_bytes(self, total_byte_count: int) -> None:
                if self._controller:
                    self._controller.raise_if_cancelled("artefact:b2:init")

            def bytes_completed(self, byte_count: int) -> None:
                if self._controller:
                    self._controller.raise_if_cancelled("artefact:b2:stream")

        progress_listener = _CancellationAwareListener() if controller else None

        if isinstance(blob, str):
            blob = blob.encode('utf-8')
        
        artefact_id = str(uuid.uuid4())
        uploaded_file = blob_db.upload_bytes(
            data_bytes=blob,
            file_name=artefact_id,
            progress_listener=progress_listener,
        )
        timestamp = datetime.datetime.now()
        
        embeddings = _get_embeddings()
        if controller:
            controller.raise_if_cancelled("artefact:embeddings")
        description_vector = embeddings.embed_query(description)
        
        metadata = {
            'timestamp': timestamp,
            'artefact_id': artefact_id,
            'thread_id': thread_id,
            'user_id': user_id,
            'generating_tool': generating_tool,
            'generating_parameters': generating_parameters,
            'description_text': description,
            'description_vector': description_vector
        }
        
        vector_str = _vector_to_str(description_vector)
        cur = metadata_db.cursor()
        cur.execute(f"""
            INSERT INTO "{TABLE_NAME}" 
            (timestamp, artefact_id, thread_id, user_id, generating_tool, generating_parameters, description_text, description_vector)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
        """, (
            timestamp, artefact_id, thread_id, user_id, generating_tool, 
            json.dumps(generating_parameters), description, vector_str
        ))
        metadata_db.commit()
        cur.close()
        
        return {'artefact_id': artefact_id, 'error': None}
    except RunCancelledError:
        raise
    except Exception as e:
        # Clean up orphan blob if metadata insertion fails
        try:
            _delete_all_versions_by_name(blob_db, artefact_id)
        except:
            pass
        return {'artefact_id': None, 'error': str(e)}

def read_artefacts(blob_db, metadata_db, metadata_only, artefact_ids=None, thread_ids=None, user_ids: list[int]|None=None, 
                   generating_tools_parameters=None, description_keywords=None, start_time=None, end_time=None):
    """
    Reads artefacts based on filters, including vector search for descriptions if keywords provided.
    Returns sorted by timestamp DESC. For description_keywords, embeds (average if multiple) and filters by cosine similarity > 0.7.
    """
    try:
        where_clauses = []
        params = []
        
        if artefact_ids:
            where_clauses.append("artefact_id IN %s")
            params.append(tuple(artefact_ids or []))
        
        if thread_ids:
            where_clauses.append("thread_id IN %s")
            params.append(tuple(thread_ids or []))
        
        if user_ids:
            where_clauses.append("user_id IN %s")
            params.append(tuple(user_ids or []))
        
        if generating_tools_parameters:
            tool_clauses = []
            for gtp in generating_tools_parameters or []:
                tool = gtp.get('tool')
                if tool:
                    if gtp.get('parameters'):
                        tool_clauses.append("(generating_tool = %s AND generating_parameters @> %s)")
                        params.extend([tool, json.dumps(gtp['parameters'])])
                    else:
                        tool_clauses.append("generating_tool = %s")
                        params.append(tool)
            if tool_clauses:
                where_clauses.append("(" + " OR ".join(tool_clauses) + ")")
        
        if description_keywords:
            embeddings = _get_embeddings()
            keyword_embeds = [embeddings.embed_query(kw) for kw in description_keywords]
            avg_embed = np.mean(keyword_embeds, axis=0).tolist()  # Average for multiple keywords to capture combined semantics
            vector_str = _vector_to_str(avg_embed)
            where_clauses.append(f"description_vector <=> %s::vector < {COSINE_DISTANCE_THRESHOLD}")
            params.append(vector_str)
        
        if start_time and end_time:
            where_clauses.append("timestamp BETWEEN %s AND %s")
            params.extend([start_time, end_time])
        
        where = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        query = f"""
            SELECT timestamp, artefact_id, thread_id, user_id, generating_tool, generating_parameters, description_text, description_vector 
            FROM "{TABLE_NAME}"{where} 
            ORDER BY timestamp DESC
        """
        
        cur = metadata_db.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        
        artefacts = []
        blob_errors = []
        for row in rows:
            metadata = {
                'timestamp': row[0],
                'artefact_id': row[1],
                'thread_id': row[2],
                'user_id': row[3],
                'generating_tool': row[4],
                'generating_parameters': row[5],
                'description_text': row[6],
                'description_vector': row[7]
            }
            blob = None
            if not metadata_only:
                try:
                    download_dest = DownloadDestBytes()
                    blob_db.download_file_by_name(metadata['artefact_id'], download_dest)
                    blob = download_dest.get_bytes_written()
                except Exception as be:
                    blob_errors.append(f"Failed to download blob for artefact_id {metadata['artefact_id']}: {str(be)}")
            artefacts.append({'blob': blob, 'metadata': metadata})
        
        error = "; ".join(blob_errors) if blob_errors else None
        success = len(blob_errors) == 0 if not metadata_only else True
        return {'artefacts': artefacts, 'success': success, 'error': error}
    except Exception as e:
        return {'artefacts': [], 'success': False, 'error': str(e)}

def delete_artefacts(blob_db, metadata_db, artefact_ids=None, thread_ids=None, user_ids: list[int]|None=None, 
                     generating_tools=None, start_time=None, end_time=None):
    """
    Deletes artefacts based on filters, from both B2 and RDS. Deletes blob first, then metadata if successful.
    Returns list of successfully deleted IDs; partial success if some fail.
    """
    try:
        where_clauses = []
        params = []
        
        if artefact_ids:
            where_clauses.append("artefact_id IN %s")
            params.append(tuple(artefact_ids or []))
        
        if thread_ids:
            where_clauses.append("thread_id IN %s")
            params.append(tuple(thread_ids or []))
        
        if user_ids:
            where_clauses.append("user_id IN %s")
            params.append(tuple(user_ids or []))
        
        if generating_tools:
            where_clauses.append("generating_tool IN %s")
            params.append(tuple(generating_tools or []))
        
        if start_time and end_time:
            where_clauses.append("timestamp BETWEEN %s AND %s")
            params.extend([start_time, end_time])
        
        where = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        query = f"SELECT artefact_id FROM \"{TABLE_NAME}\"{where}"
        
        cur = metadata_db.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        
        deleted_ids = []
        errors = []
        for row in rows:
            art_id = row[0]
            try:
                logger.debug("Deleting all B2 versions for artefact_id=%s", art_id)
                deleted_blob = _delete_all_versions_by_name(blob_db, art_id)
                
                if deleted_blob:
                    # Delete from metadata if blob deleted
                    cur = metadata_db.cursor()
                    cur.execute(f"DELETE FROM \"{TABLE_NAME}\" WHERE artefact_id = %s", (art_id,))
                    metadata_db.commit()
                    cur.close()
                    deleted_ids.append(art_id)
                else:
                    errors.append(f"No blob found for artefact_id {art_id}")
            except Exception as e:
                errors.append(f"Failed to delete artefact_id {art_id}: {str(e)}")
        
        error = "; ".join(errors) if errors else None
        return {'artefact_ids': deleted_ids, 'error': error}
    except Exception as e:
        return {'artefact_ids': [], 'error': str(e)}