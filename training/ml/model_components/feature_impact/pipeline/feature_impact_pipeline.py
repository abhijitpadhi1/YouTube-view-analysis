import os, sys

# Import custom modules and classes
from training.logging.logger import Logger
from training.exception.exception import CustomException

# Importing artifact entities
from training.ml.model_components.feature_impact.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact, 
    ModelTrainerArtifact,
)
# Importing configuration entities
from training.ml.model_components.feature_impact.entity.config_entity import (
    FeatureImpactModelTrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from training.data_preparation.entity.artifact_entity import DataTransformationArtifact as DataPreparationDataTransformationArtifact 
# Importing components
from training.ml.model_components.feature_impact.components.data_ingestion import DataIngestion
from training.ml.model_components.feature_impact.components.data_validation import DataValidation
from training.ml.model_components.feature_impact.components.data_transformation import DataTransformation
from training.ml.model_components.feature_impact.components.model_trainer import ModelTrainer


class FeatureImpactModelTrainingPipeline:
    def __init__(self, data_preparation_data_transformation_artifact:DataPreparationDataTransformationArtifact):
        try:
            # Logger().log("Initializing FeatureImpactModelTrainingPipeline class.")
            self.data_preparation_data_transformation_artifact = data_preparation_data_transformation_artifact
            self.feature_impact_model_training_pipeline_config = FeatureImpactModelTrainingPipelineConfig()
            self.data_ingestion_config = DataIngestionConfig(self.feature_impact_model_training_pipeline_config)
            self.data_validation_config = DataValidationConfig(self.feature_impact_model_training_pipeline_config)
            self.data_transformation_config = DataTransformationConfig(self.feature_impact_model_training_pipeline_config)
            self.model_trainer_config = ModelTrainerConfig(self.feature_impact_model_training_pipeline_config)
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in FeatureImpactModelTrainingPipeline __init__: {err}", level='error')

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            Logger().log("Starting data ingestion.")
            data_ingestion = DataIngestion(self.data_ingestion_config, self.data_preparation_data_transformation_artifact)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            Logger().log("Data ingestion completed.")
            return data_ingestion_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in start_data_ingestion: {err}", level='error')
            raise err
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            Logger().log("Starting data validation.")
            data_validation = DataValidation(self.data_validation_config, data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            Logger().log("Data validation completed.")
            return data_validation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in start_data_validation: {err}", level='error')
            raise err
        
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact) -> DataTransformationArtifact:
        try:
            Logger().log("Starting data transformation.")
            data_transformation = DataTransformation(data_validation_artifact, self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            Logger().log("Data transformation completed.")
            return data_transformation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in start_data_transformation: {err}", level='error')
            raise err
        
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            Logger().log("Starting model training.")
            model_trainer = ModelTrainer(data_transformation_artifact, self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            Logger().log("Model training completed.")
            return model_trainer_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in start_model_trainer: {err}", level='error')
            raise err
        
    def run_feature_impact_model_training_pipeline(self) -> ModelTrainerArtifact:
        try:
            Logger().log("="*50 + "\tFeature Impact Model\t" + "="*50)
            Logger().log("="*120)
            data_ingestion_artifact = self.start_data_ingestion()
            Logger().log("="*120)
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            Logger().log("="*120)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            Logger().log("="*120)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            Logger().log("Feature impact model training pipeline completed.")
            Logger().log("="*120)
            return model_trainer_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in run_feature_impact_model_training_pipeline: {err}", level='error')
            raise err