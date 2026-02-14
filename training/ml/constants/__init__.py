import os, sys
from datetime import datetime
import numpy as np
import pandas as pd



"""
Define common constants for training pipelines.
"""
TRAINING_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
SCHEMA_FILE_NAME = "schema.yaml"


"""
Data Ingestion related constant start with DATA_INGESTION variable name.
"""
DATA_INGESTION_COLLECTION_NAME = "youtube_data"
DATA_INGESTION_DATABASE_NAME = "youtube_db"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2