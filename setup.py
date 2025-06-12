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

from parameters import custom_table_info, include_tables


def get_llm() -> ChatVertexAI:
    """Initialize the Grok 3 Beta language model.

    Returns:
        A ChatOpenAI instance configured with xAI API.
    """
    st.toast("Setting up the Gemini 2.0 Flash LLM...", icon=":material/build:")
    return ChatVertexAI(
        model="gemini-2.0-flash-001"
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
        Table('3d_condours', metadata,  # Replace 'your_table' with actual table name
              Column('id', Integer, primary_key=True),
              Column('contour_bound', Geometry)
        )

        # Initialize SQLDatabase with custom metadata
        db = SQLDatabase(
            engine=engine,
            metadata=metadata,
            include_tables=include_tables,
            custom_table_info=custom_table_info,
            sample_rows_in_table_info=3,
            lazy_table_reflection=True
        )

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