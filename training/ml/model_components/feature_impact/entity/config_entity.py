from datetime import datetime
import os

# Import constants from __init__.py
from training.ml.constants import *
class FeatureImpactModelTrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        self.timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        self.pipeline_name = PIPELINE_NAME
        self.pipeline_stage = PIPELINE_STAGE
        self.artifact_dir = os.path.join(ARTIFACT_DIR, self.pipeline_name, self.timestamp)
        self.schema_file_path = os.path.join(self.artifact_dir, SCHEMA_FILE_NAME)


class DataIngestionConfig:
    def __init__(self, feature_impact_model_training_pipeline_config:FeatureImpactModelTrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            feature_impact_model_training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAINING_FILE_NAME
        )
        self.test_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


class DataValidationConfig:
    def __init__(self, feature_impact_model_training_pipeline_config:FeatureImpactModelTrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            feature_impact_model_training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_VALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, TRAINING_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, TEST_FILE_NAME
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_INVALID_DIR
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, TRAINING_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, TEST_FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_REPORT_FILE_NAME
        )


class DataTransformationConfig:
    def __init__(self, feature_impact_model_training_pipeline_config:FeatureImpactModelTrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            feature_impact_model_training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAINING_FILE_NAME
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME
        )
        
class ModelTrainerConfig:
    def __init__(self, feature_impact_model_training_pipeline_config:FeatureImpactModelTrainingPipelineConfig):
        self.model_training_dir: str = os.path.join(
            feature_impact_model_training_pipeline_config.artifact_dir, MODEL_TRAINING_DIR_NAME
        )
        self.trained_rf_model_file_path: str = os.path.join(
            self.model_training_dir, MODEL_TRAINING_TRAINED_MODEL_DIR, MODEL_TRAINING_TRAINED_RF_MODEL_FILE_NAME
        )
        self.trained_shap_explainer_file_path: str = os.path.join(
            self.model_training_dir, MODEL_TRAINING_TRAINED_MODEL_DIR, MODEL_TRAINING_TRAINED_SHAP_EXPLAINER_FILE_NAME
        )
        self.trained_shap_values_file_path: str = os.path.join(
            self.model_training_dir, MODEL_TRAINING_TRAINED_MODEL_DIR, MODEL_TRAINER_SHAP_VALUES_FILE_NAME
        )
        self.trained_shap_explanation_plots_dir: str = os.path.join(
            self.model_training_dir, MODEL_TRAINING_TRAINED_MODEL_DIR, MODEL_TRAINER_SHAP_EXPLANATION_PLOTS_DIR
        )
        self.expected_score: float = MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float = MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD