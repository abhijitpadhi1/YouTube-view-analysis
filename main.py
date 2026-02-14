# Importing necessary modules and classes
import os
import sys

# Import custom modules
from training.data_preparation.components.data_transformation import DataTransformation
from training.logging.logger import Logger
from training.exception.exception import CustomException

# Import entity and component modules
from training.data_preparation.entity.config_entity import (
    DataTransformationConfig,
    PreparationPipelineConfig, 
    DataIngestionConfig,
    DataValidationConfig
)
# Import components for testing
from training.data_preparation.components.data_ingetion import DataIngestion
from training.data_preparation.components.data_validation import DataValidation

def main():
    try:

        Logger().log("Testing the Data Preparation Pipeline")
        Logger().log("="*100)
        ## ============= Testing the Data Preparation Pipeline - Data Ingestion ============= ##
        # Initialize the preparation pipeline configuration
        Logger().log("Testing the Data Ingestion step of Data Preparation Pipeline")
        preparation_pipeline_config = PreparationPipelineConfig()
        # Initialize the data ingestion configuration
        data_ingestion_config = DataIngestionConfig(preparation_pipeline_config)
        # Initialize the data ingestion component
        data_ingestion = DataIngestion(data_ingestion_config, data_source="local")
        # Test the ingestion step of the data preparation pipeline
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        # print(f"Data Ingestion Artifact: {data_ingestion_artifact}")
        # print("Data Ingestion step of Data Preparation Pipeline tested successfully.")
        print("Data Preparation Pipeline - Data Ingestion ✅.")
        Logger().log(f"Data Ingestion Artifact: {data_ingestion_artifact}")
        Logger().log("Data Ingestion step of Data Preparation Pipeline tested successfully.")
        ## =================================================================================== ##

        Logger().log("="*100)

        ## ============= Testing the Data Preparation Pipeline - Data Validation ============= ##
        # Initialize the data validation configuration
        Logger().log("Testing the Data Validation step of Data Preparation Pipeline")
        data_validation_config = DataValidationConfig(preparation_pipeline_config)
        # Initialize the data validation component
        data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
        # Test the validation step of the data preparation pipeline
        data_validation_artifact = data_validation.initiate_data_validation()
        # print(f"Data Validation Artifact: {data_validation_artifact}")
        # print("Data Validation step of Data Preparation Pipeline tested successfully.")
        print("Data Preparation Pipeline - Data Validation ✅.")
        Logger().log(f"Data Validation Artifact: {data_validation_artifact}")
        Logger().log("Data Validation step of Data Preparation Pipeline tested successfully.")
        ## =================================================================================== ##

        Logger().log("="*100)

        ## ============= Testing the Data Preparation Pipeline - Data Transformation ============= ##
        # Initialize the data transformation configuration
        Logger().log("Testing the Data Transformation step of Data Preparation Pipeline")
        data_transformation_config = DataTransformationConfig(preparation_pipeline_config)
        # Initialize the data transformation component
        data_transformation = DataTransformation(data_transformation_config, data_validation_artifact)
        # Test the transformation step of the data preparation pipeline
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        # print(f"Data Transformation Artifact: {data_transformation_artifact}")
        # print("Data Transformation step of Data Preparation Pipeline tested successfully.")
        print("Data Preparation Pipeline - Data Transformation ✅.")
        Logger().log(f"Data Transformation Artifact: {data_transformation_artifact}")
        Logger().log("Data Transformation step of Data Preparation Pipeline tested successfully.")
        ## =================================================================================== ##

    except Exception as e:
        err = CustomException(str(e), sys)
        Logger().log(f"Error in main function: {err}", level="error")
        raise err


if __name__ == "__main__":
    main()
