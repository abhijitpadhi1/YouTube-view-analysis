import os, sys
import pandas as pd

from sklearn.model_selection import train_test_split

# Importing custom modules 
from training.logging.logger import Logger
from training.exception.exception import CustomException

# Importing entity modules
from training.data_preparation.entity.artifact_entity import DataTransformationArtifact
from training.ml.model_components.feature_impact.entity.config_entity import DataIngestionConfig
from training.ml.model_components.feature_impact.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(
            self, 
            data_ingestion_config: DataIngestionConfig,
            preparation_artifact: DataTransformationArtifact
        ):
        try:
            Logger().log(f"Initializing DataIngestion with config: {data_ingestion_config}")
            self.data_ingestion_config = data_ingestion_config
            self.preparation_artifact = preparation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in DataIngestion __init__: {err}", level="error")
            raise err

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads data from the feature store file path and returns it as a DataFrame."""
        try:
            Logger().log(f"Reading data from: {file_path}")
            return pd.read_parquet(file_path)
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in read_data: {err}", level="error")
            raise err
        
    def export_data_into_feature_store(self, df: pd.DataFrame) -> None:
        """
        Method to export the DataFrame into a feature store (Parquet file).
        
        Args:
            df (pd.DataFrame): The DataFrame to be exported.

        Returns:
            None
        """
        try:
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            df.to_parquet(self.data_ingestion_config.feature_store_file_path, index=False)
            Logger().log(f"Data exported to feature store at: {self.data_ingestion_config.feature_store_file_path}")

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in exporting data to feature store: {err}", level="error")
            raise err

    def spliting_of_data(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42
            )

            # Save the train and test data to respective file paths
            train_file_path = self.data_ingestion_config.training_file_path
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            train_set.to_parquet(train_file_path, index=False)
            Logger().log(f"Train file saved at: {train_file_path}")

            test_file_path = self.data_ingestion_config.test_file_path
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            test_set.to_parquet(test_file_path, index=False)
            Logger().log(f"Test file saved at: {test_file_path}")

            Logger().log("Train and test data saved successfully")

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in spliting_of_data: {err}", level="error")
            raise err
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            Logger().log("Starting data ingestion process")
            # Read the data from the feature store
            df = self.read_data(self.preparation_artifact.transformed_data_file_path)
            Logger().log(f"Data read successfully with shape: {df.shape}")

            # Export the data into the feature store
            self.export_data_into_feature_store(df)
            Logger().log("Data exported to feature store successfully")

            # Split the data into train and test sets and save them
            self.spliting_of_data(df)
            Logger().log("Data splitting and saving completed successfully")

            # Create and return the DataIngestionArtifact
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                training_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            Logger().log(f"Data Ingestion Artifact created: {data_ingestion_artifact}")
            Logger().log("Data ingestion process completed successfully")
            return data_ingestion_artifact

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in initiate_data_ingestion: {err}", level="error")
            raise err