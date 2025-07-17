"""Set up dependencies and configuration for the MissionHelp Demo application.

This module initializes the LLM, embeddings, Qdrant vector store, and database,
ensuring all components are ready for the RAG pipeline.
"""

import os
import streamlit as st
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table, Column, Integer
from geoalchemy2 import Geometry
from parameters import include_tables, table_info
from collections import defaultdict
from typing import List, Tuple
# from tools.get_user_permissions import UserPermissionsTool, UserPermissionsToolOutput
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def enable_tracing():
    """Enables LangSmith tracing."""
    os.environ['LANGSMITH_TRACING'] = st.secrets['LANGSMITH_TRACING']
    os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']


def build_relationship_graph(table_info=table_info) -> defaultdict[str, List[Tuple]]:
    """Build a relationship graph from table_info."""
    st.toast("Building table relationship graph...", icon=":material/account_tree:")
    graph = defaultdict(list)
    for table in table_info:
        table_name = table['name']
        for rel in table.get('relationships', []):
            graph[table_name].append(
                (rel['referenced_table'], rel['column'], rel['referenced_column'])
            )
    return graph


# def get_user_permissions() -> UserPermissionsToolOutput:
#     st.toast("Loading user permissions...", icon=":material/lock_open:")
#     return UserPermissionsTool(db=st.session_state.db).invoke(
#         input={'user_id': st.session_state.selected_user_id}
#     )


def get_global_hierarchy_access() -> bool:
    """Check if the user has global hierarchy access."""
    st.toast("Checking global hierarchy access...", icon=":material/lock_open:")
    query = """
            SELECT uagu.group_id
            FROM user_access_groups_users uagu
            RIGHT JOIN geo_12_users gu ON uagu.user_id = gu.id
            WHERE gu.id = %s
        """ % st.session_state.selected_user_id
    result = st.session_state.db.run(query)
    if not result:
        logging.warning("No user access groups found for user ID %s", st.session_state.selected_user_id)
        return False
    
    parsed_result = eval(result)
    row = parsed_result[0]
    logging.debug(f"Global hierarchy access check for user ID {st.session_state.selected_user_id}: {row[0]}")
    return row[0] == 0


def set_google_credentials() -> None:
    """Set Google Cloud credentials for database access.

    Writes credentials from secrets to a temporary file and sets the environment variable.
    """
    st.toast("Setting Google credentials...", icon=":material/build:")
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    temp_file_path = "google_credentials.json"
    with open(temp_file_path, "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path


def get_llm() -> ChatVertexAI:
    """Initialize the Grok 3 Beta language model.

    Returns:
        A ChatOpenAI instance configured with xAI API.
    """
    st.toast("Setting up the Gemini 2.0 Flash LLM...", icon=":material/build:")
    return ChatVertexAI(
        model="gemini-2.0-flash-001",
        temperature = 0.1
        # model="gemini-2.5-pro-preview-05-06"        
    )


def get_db() -> SQLDatabase:
    """Initialize the SQL database connection.

    Returns:
        An SQLDatabase instance connected to the MissionOS Hanoi CP03 database.
    """
    st.toast("Connecting to the MissionOS CP03 database...", icon=":material/build:")
    try:
        db_host = st.secrets["database"]["db_host"]
        db_user = st.secrets["database"]["db_user"]
        db_pass = st.secrets["database"]["db_pass"]
        db_name = st.secrets["database"]["db_name"]
        port = st.secrets["database"]["port"]

        db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{port}/{db_name}"
        engine = create_engine(db_uri, echo=False)

        # Define metadata with custom Geometry type
        metadata = MetaData()
        Table('3d_condours', metadata,
              Column('id', Integer, primary_key=True),
              Column('contour_bound', Geometry)
        )

        # Initialize SQLDatabase with custom metadata
        db = SQLDatabase(
            engine=engine,
            metadata=metadata,
            include_tables=include_tables,
            # custom_table_info=custom_table_info,
            sample_rows_in_table_info=3,
            lazy_table_reflection=True
        )
        logging.debug(f"Available tables: {db.get_usable_table_names()}")

        st.toast("Connected to the MissionOS CP03 database", icon=":material/check_circle:")
        return db

    except Exception as e:
        raise Exception(f"Failed to connect to database: {str(e)}")


def set_google_credentials() -> None:
    """Set Google Cloud credentials for database access.

    Writes credentials from secrets to a temporary file and sets the environment variable.
    """
    st.toast("Setting Google credentials...", icon=":material/build:")
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    temp_file_path = "google_credentials.json"
    with open(temp_file_path, "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path