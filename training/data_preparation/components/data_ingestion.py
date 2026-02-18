import os, sys
import pandas as pd
import certifi
# Import the costom modules
from training.logging.logger import Logger
from training.exception.exception import CustomException

# Import the entity modules
from training.data_preparation.entity.config_entity import DataIngestionConfig
from training.data_preparation.entity.artifact_entity import DataIngestionArtifact

# Load the .env file
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Root certificates for secure HTTPS connection
ca = certifi.where()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, data_source: str = "csv"):
        try:
            Logger().log(f"Init Data Ingestion for DataPreparation with config: {data_ingestion_config}")
            self.data_ingestion_config = data_ingestion_config
            self.data_source = data_source
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in Data Ingestion: {err}", level="error")
            raise err

    def collection_as_dataframe(
            self, 
            collection_name: str | None = None, 
            database_name: str | None = None
        ) -> pd.DataFrame:
        """
        Method to read data from MongoDB collection and convert it to a pandas DataFrame.
        
        Args:
            collection_name (str, optional): Name of the MongoDB collection to read from. Defaults to None, in which case it will use the collection name from the config.
            database_name (str, optional): Name of the MongoDB database to read from. Defaults to None, in which case it will use the database name from the config.

        Returns:
            pd.DataFrame: DataFrame containing the data read from the MongoDB collection.
        """
        try:
            from pymongo import MongoClient, errors
            if collection_name is None:
                collection_name = self.data_ingestion_config.collection_name
            if database_name is None:
                database_name = self.data_ingestion_config.database_name
            client = MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            db = client[database_name]
            collection = db[collection_name]

            # Connect to MongoDB and read the collection data into a DataFrame
            Logger().log(f"Reading data from MongoDB collection: {collection_name} in database: {database_name}")
            try:
                client.admin.command('ping')
                Logger().log("Successfully connected to MongoDB")
            except errors.ConnectionFailure as e:
                Logger().log(f"Failed to connect to MongoDB: {e}", level="error")
                raise CustomException(f"Failed to connect to MongoDB: {e}", sys)
            
            # Read the collection data into a DataFrame
            data = list(collection.find())
            df = pd.DataFrame(data)
            Logger().log(f"Successfully read data from MongoDB collection: {collection_name}")
            return df

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in connecting to MongoDB: {err}", level="error")
            raise err

    def local_data_as_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Method to read data from a local file (CSV or Parquet) and convert it to a pandas DataFrame.
        
        Args:
            file_path (str): The path to the local file to be read.

        Returns:
            pd.DataFrame: DataFrame containing the data read from the local file.
        """
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            else:
                err = ValueError(f"Unsupported file format: {file_path}")
                Logger().log(f"Error in reading local data: {err}", level="error")
                raise err
            Logger().log(f"Successfully read data from local file: {file_path}")
            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in reading local data: {err}", level="error")
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

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method to initiate the data ingestion process. It reads data from the specified source (MongoDB or local file), processes it, and exports it to the feature store. Finally, it creates and returns a DataIngestionArtifact containing the path to the feature store file.
        
        Returns:
            DataIngestionArtifact: An artifact containing the path to the feature store file.
        """
        try:
            Logger().log("Starting data ingestion process")
            if self.data_source == "mongodb":
                df = self.collection_as_dataframe()
            elif self.data_source == "local":
                df = self.local_data_as_dataframe(
                    self.data_ingestion_config.local_data_file_path
                )
            else:
                err = ValueError(f"Unsupported data source: {self.data_source}")
                Logger().log(f"Error in data ingestion: {err}", level="error")
                raise err

            Logger().log(f"Data ingestion successful, Dataframe shape: {df.shape}")
            self.export_data_into_feature_store(df)
            Logger().log("Saving data to feature store completed.")
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            )
            Logger().log(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            Logger().log("Data ingestion process completed successfully")
            return data_ingestion_artifact
            
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in Data Ingestion: {err}", level="error")
            raise err