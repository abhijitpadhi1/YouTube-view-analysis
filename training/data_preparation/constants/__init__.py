import os, sys
from datetime import datetime
import numpy as np
import pandas as pd

TRAINING_ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Define common constants for data preparation pipelines.
"""
ARTIFACT_DIR: str = os.path.join(TRAINING_ROOT_DIR, "artifact")
PIPELINE_NAME: str = "data_preparation"
PIPELINE_STAGE: str = "data_preparation"
FILE_NAME: str = "youtube_data.parquet"
LOCAL_DATA_DIR: str = "YouTubeData"
LOCAL_DATA_FILE_NAME: str = "YouTube_demo_data.csv"
LOCAL_DATA_FILE: str = os.path.join(LOCAL_DATA_DIR, LOCAL_DATA_FILE_NAME)
DATA_SCHEMA_DIR: str = "YouTubeDataSchema"
SCHEMA_FILE_NAME: str = "schema.yaml"
SCHEMA_FILE_PATH: str = os.path.join(DATA_SCHEMA_DIR, SCHEMA_FILE_NAME)


"""
Data Ingestion related constant start with DATA_INGESTION variable name.
"""
DATA_INGESTION_COLLECTION_NAME: str = "youtube_data"
DATA_INGESTION_DATABASE_NAME: str = "youtube_db"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"


"""
Data Validation related constant start with DATA_VALIDATION variable name.
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalidated"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"


"""
Data Transformation related constant start with DATA_TRANSFORMATION variable name.
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_DATA_FILE_NAME: str = "transformed_data.parquet"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"