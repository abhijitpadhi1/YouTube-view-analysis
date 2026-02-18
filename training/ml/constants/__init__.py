import os, sys
from datetime import datetime
import numpy as np
import pandas as pd


TRAINING_ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Define common constants for feature impact model training pipelines.
"""
ARTIFACT_DIR: str = os.path.join(TRAINING_ROOT_DIR, "artifact", "ml")
PIPELINE_NAME: str = "feature_impact_model"
PIPELINE_STAGE: str = "ml_model1_training"
FILE_NAME: str = "youtube_data.parquet"

DATA_SCHEMA_DIR: str = os.path.join(TRAINING_ROOT_DIR, "ml", "ML_DataSchema")
SCHEMA_FILE_NAME: str = "schema.yaml"
SCHEMA_FILE_PATH: str = os.path.join(DATA_SCHEMA_DIR, SCHEMA_FILE_NAME)

TRAINING_FILE_NAME: str = "train_data.parquet"
TEST_FILE_NAME: str = "test_data.parquet"

"""
Data Ingestion related constant start with DATA_INGESTION variable name.
"""
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

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

"""
Model Training related constant start with MODEL_TRAINING variable name.
"""
MODEL_TRAINING_DIR_NAME: str = "model_training"
MODEL_TRAINING_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINING_TRAINED_RF_MODEL_FILE_NAME: str = "rf_model.pkl"
MODEL_TRAINING_TRAINED_SHAP_EXPLAINER_FILE_NAME: str = "shap_explainer.pkl"
MODEL_TRAINER_SHAP_VALUES_FILE_NAME: str = "shap_values.csv"
MODEL_TRAINER_SHAP_EXPLANATION_PLOTS_DIR: str = "shap_explanation_plots"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05

"""
Save the created model to a outer directry to use in serving
"""
MODEL_REGISTRY: str = "model_registry"
MODEL_REGISTRY_ML_MODEL_DIR: str = os.path.join(MODEL_REGISTRY, "ml")
MODEL_REGISTRY_ML_FEATURE_IMPACT_MODEL = os.path.join(MODEL_REGISTRY_ML_MODEL_DIR, "feature_impact_model")
