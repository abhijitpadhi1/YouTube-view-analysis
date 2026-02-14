# Importing necessary libraries for data ingestion
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing custom modules for logging and exception handling
from training.logging.logger import Logger
from training.exception.exception import CustomException

# Importing the artifact entity for data ingestion
from training.entity.artifact_entity import DataIngetionArtifact

# Defining the DataIngetion class to handle the data ingestion process
class DataIngetion:
    def __init__(self, data_ingestion_config: DataIngetionArtifact, data_source: str) -> None:
        try:
            Logger().log("DataIngetion with provided configuration %s.", str(data_ingestion_config))
            self.data_ingestion_config = data_ingestion_config
            self.data_source = data_source
        except Exception as e:
            custom_exception = CustomException(str(e), sys)
            custom_exception.log_exception()
            raise custom_exception
        
    def collectio_as_dataframe(self) -> pd.DataFrame:
        try:
            pass
            
        except Exception as e:
            custom_exception = CustomException(str(e), sys)
            custom_exception.log_exception()
            raise custom_exception