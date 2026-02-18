import sys

# Import custom modules
from training.exception.exception import CustomException
from training.logging.logger import Logger
# Import component modules
from training.data_preparation.components.data_ingestion import DataIngestion
from training.data_preparation.components.data_validation import DataValidation
from training.data_preparation.components.data_transformation import DataTransformation
# Import entity modules
from training.data_preparation.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    PreparationPipelineConfig
)
from training.data_preparation.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)


class PreparationPipeline:
    def __init__(self):
        try:
            self.preparation_pipeline_config = PreparationPipelineConfig()
            self.data_ingestion_config = DataIngestionConfig(self.preparation_pipeline_config)
            self.data_validation_config = DataValidationConfig(self.preparation_pipeline_config)
            self.data_transformation_config = DataTransformationConfig(self.preparation_pipeline_config)
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(str(err))
            raise err
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            Logger().log("Starting the Data Ingestion step of Data Preparation Pipeline")
            data_ingestion = DataIngestion(self.data_ingestion_config, data_source="local")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            Logger().log(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            Logger().log("Data Ingestion step of Data Preparation Pipeline completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(str(err))
            raise err
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            Logger().log("Starting the Data Validation step of Data Preparation Pipeline")
            data_validation = DataValidation(self.data_validation_config, data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            Logger().log(f"Data Validation Artifact: {data_validation_artifact}")
            Logger().log("Data Validation step of Data Preparation Pipeline completed successfully.")
            return data_validation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(str(err))
            raise err
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            Logger().log("Starting the Data Transformation step of Data Preparation Pipeline")
            data_transformation = DataTransformation(self.data_transformation_config, data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            Logger().log(f"Data Transformation Artifact: {data_transformation_artifact}")
            Logger().log("Data Transformation step of Data Preparation Pipeline completed successfully.")
            return data_transformation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(str(err))
            raise err

    def run_data_preparation_pipeline(self) -> DataTransformationArtifact:
        try:
            Logger().log("="*50 + "\tData Preparation\t" + "="*50)
            Logger().log("="*100)
            data_ingestion_artifact = self.start_data_ingestion()
            Logger().log("="*100)
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            Logger().log("="*100)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            Logger().log("="*100)
            Logger().log("Data Preparation Pipeline completed successfully.")
            Logger().log("="*100)
            return data_transformation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(str(err))
            raise err