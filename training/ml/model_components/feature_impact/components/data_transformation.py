import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Importing custom modules and classes
from training.logging.logger import Logger
from training.exception.exception import CustomException
# Importing configuration entities
from training.ml.model_components.feature_impact.entity.config_entity import DataTransformationConfig
# Importing artifact entities
from training.ml.model_components.feature_impact.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
# Importing constants
from training.ml.constants import SCHEMA_FILE_PATH
# Importing utility functions
from training.utils.util import read_yaml_file, write_yaml_file


class DataTransformation:
    def __init__(
            self, 
            data_validation_artifact:DataValidationArtifact,
            data_transformation_config:DataTransformationConfig
        ):
        try:
            Logger().log("Initializing DataTransformation class.")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in DataTransformation __init__: {err}", level="error")
            raise err

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads data from a parquet file and returns it as a DataFrame."""
        try:
            Logger().log(f"Reading data from: {file_path}")
            return pd.read_parquet(file_path)
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in read_data: {err}", level="error")
            raise err
        
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects the features for feature impact analysis based on the expected schema."""
        try:
            schema = read_yaml_file(SCHEMA_FILE_PATH)
            selected_structural_cols = schema['structural_columns']
            
            Logger().log(f"Selecting features: {selected_structural_cols}")
            return df[selected_structural_cols]
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in select_features: {err}", level="error")
            raise err
        
    def create_target_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'video_view_count' not in df.columns:
                raise CustomException("video_view_count column is missing in the DataFrame", sys)
            df['log_view_count'] = np.log1p(df['video_view_count'])
            df = df.drop(columns=['video_view_count'])
            # Logger().log("Dropped 'video_view_count' column after creating log-transformed target feature")
            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in create_target_feature: {err}", level="error")
            raise err

    def apply_Ordinal_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Applies ordinal encoding to the categorical columns in the DataFrame."""
        try:
            # Get the categorical columns
            categorical_cols = [
                'video_category_id',
                'duration_bucket',
                'channel_subscriber_bucket',
                'video_trending_country'
            ]

            # Apply ordinal encoding to the categorical columns
            cat_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )

            # Fit the encoder on the training data and transform both training and test data
            train_df[categorical_cols] = cat_encoder.fit_transform(train_df[categorical_cols])
            test_df[categorical_cols] = cat_encoder.transform(test_df[categorical_cols])    
            
            return train_df, test_df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in apply_Ordinal_encoding: {err}", level="error")
            raise err
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            Logger().log("Starting data transformation process.")
            # Read the training and test data
            train_df = self.read_data(self.data_validation_artifact.validated_training_data_path)
            test_df = self.read_data(self.data_validation_artifact.validated_test_data_path)

            # Select the features for feature impact analysis
            train_df = self.select_features(train_df)
            test_df = self.select_features(test_df)

            # Create the target feature for feature impact analysis
            train_df = self.create_target_feature(train_df)
            test_df = self.create_target_feature(test_df)

            # Apply ordinal encoding to the categorical features
            train_df, test_df = self.apply_Ordinal_encoding(train_df, test_df)

            # Save the transformed data to respective file paths
            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            os.makedirs(os.path.dirname(transformed_train_file_path), exist_ok=True)
            train_df.to_parquet(transformed_train_file_path, index=False)
            Logger().log(f"Transformed train file saved at: {transformed_train_file_path}")
            
            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            os.makedirs(os.path.dirname(transformed_test_file_path), exist_ok=True)
            test_df.to_parquet(transformed_test_file_path, index=False)
            Logger().log(f"Transformed test file saved at: {transformed_test_file_path}")
            
            # Prepare the artifact to be returned
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path
            )

            Logger().log(f"Data transformation completed successfully. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in initiate_data_transformation: {err}", level="error")
            raise err