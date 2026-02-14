from datetime import datetime
import os

# Import constants from __init__.py
from training.data_preparation.constants import *

class PreparationPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        self.timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        self.pipeline_name = PIPELINE_NAME
        self.pipeline_stage = PIPELINE_STAGE
        self.artifact_dir = os.path.join(ARTIFACT_DIR, self.pipeline_name, self.timestamp)
        self.schema_file_path = os.path.join(self.artifact_dir, SCHEMA_FILE_NAME)
        

class DataIngestionConfig:
    def __init__(self, preparation_pipeline_config:PreparationPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            preparation_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
        )
        # MongoDB related constants
        self.database_name: str = DATA_INGESTION_DATABASE_NAME
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME
        # Local data file path for csv source
        self.local_data_file_path: str = LOCAL_DATA_FILE


class DataValidationConfig:
    def __init__(self, preparation_pipeline_config:PreparationPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            preparation_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_VALID_DIR, FILE_NAME
        )
        self.invalid_data_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_INVALID_DIR, FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_REPORT_FILE_NAME
        )
    