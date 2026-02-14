import os, sys
import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import skew, kurtosis
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

# Importing custom modules and classes
from training.logging.logger import Logger
from training.exception.exception import CustomException

# Importing constants 
from training.data_preparation.constants import SCHEMA_FILE_PATH
# Importing configuration entities
from training.data_preparation.entity.config_entity import DataValidationConfig
# Importing artifact entities
from training.data_preparation.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
# Importing utility functions
from training.utils.util import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
            self, 
            data_validation_config:DataValidationConfig, 
            data_ingestion_artifact:DataIngestionArtifact
        ):
        try:
            Logger().log("Initializing DataValidation class.")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in DataValidation __init__: {err}", level="error")
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


    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        """Validates the number of columns in the DataFrame against the expected schema."""
        try:
            number_of_columns = len(self._schema_config['columns'])
            if df.shape[1] == number_of_columns:
                Logger().log("Number of columns validation passed.")
                return True
            else:
                Logger().log(f"Number of columns validation failed. Expected: {number_of_columns}, Found: {df.shape[1]}", level="warning")
                return False
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in validate_number_of_columns: {err}", level="error")
            raise err

        
    def detect_data_drift1(
            self, base_df: pd.DataFrame, current_df: pd.DataFrame,
            numeric_cols: list, categorical_cols: list,
            alpha: float = 0.05, min_effect_size: float = 0.1
        ) -> bool:
        """
        Detects data drift between the base and current DataFrames using statistical tests.
        
        Args:            
            base_df (pd.DataFrame): The reference DataFrame.
            current_df (pd.DataFrame): The current DataFrame to compare against the base.
            numeric_cols (list): List of numeric column names.
            categorical_cols (list): List of categorical column names.
            alpha (float): Significance level for statistical tests.
            min_effect_size (float): Minimum effect size to consider drift significant.

        Returns:
            dict: A dictionary containing drift detection results for each column and overall status.
        """

        report = {}
        drift_count = 0
        total_cols = len(numeric_cols) + len(categorical_cols)
        # Bonferroni correction
        corrected_alpha = alpha / max(total_cols, 1)

        # -----------------------
        # Numeric Drift (KS Test)
        # -----------------------
        for col in numeric_cols:
            d1 = base_df[col].dropna()
            d2 = current_df[col].dropna()
            if len(d1) == 0 or len(d2) == 0:
                continue

            ks_result: Any = ks_2samp(d1, d2)
            drift_detected = (ks_result.pvalue < corrected_alpha and ks_result.statistic > min_effect_size)

            severity = "none"
            if drift_detected:
                if ks_result.statistic > 0.3:
                    severity = "strong"
                else:
                    severity = "moderate"
                drift_count += 1

            report[col] = {
                "test": "ks_test",
                "statistic": float(ks_result.statistic),
                "p_value": float(ks_result.pvalue),
                "severity": severity,
                "drift_detected": drift_detected
            }

        # -----------------------------
        # Categorical Drift (Chi-Square)
        # -----------------------------
        for col in categorical_cols:
            contingency = pd.crosstab(base_df[col], current_df[col])
            if contingency.empty:
                continue

            chi2_result: Any = chi2_contingency(contingency)
            chi2, p, _, _ = chi2_result
            drift_detected = p < corrected_alpha

            severity = "none"
            if drift_detected:
                severity = "moderate"
                drift_count += 1

            report[col] = {
                "test": "chi_square",
                "statistic": float(chi2),
                "p_value": float(p),
                "severity": severity,
                "drift_detected": drift_detected
            }

        drift_ratio = drift_count / max(total_cols, 1)
        overall_status = drift_ratio < 0.3  # configurable threshold

        # Write in the report
        final_report = {
            "overall_status": overall_status,
            "drift_ratio": drift_ratio,
            "columns": report
        }
        
        # Save the report to a YAML file 
        drift_report_file_path = self.data_validation_config.drift_report_file_path
        dir_path = os.path.dirname(drift_report_file_path)
        os.makedirs(dir_path, exist_ok=True)
        write_yaml_file(drift_report_file_path, final_report)

        return overall_status


    def detect_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detects data drift in the given DataFrame and returns a report."""
        try:
            numeric_cols = self._schema_config['numeric_columns']
            categorical_cols = self._schema_config['categorical_columns']

            # Check if all features are present in the data
            missing_numeric = [col for col in numeric_cols if col not in data.columns]
            missing_categorical = [col for col in categorical_cols if col not in data.columns]
            return {
                "missing_numeric": missing_numeric,
                "missing_categorical": missing_categorical
            }
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in detect_features: {err}", level="error")
            raise err

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Load the ingested data
            ingested_data_path = self.data_ingestion_artifact.feature_store_file_path
            df = DataValidation.read_data(ingested_data_path)

            # Validate number of columns
            status = self.validate_number_of_columns(df)
            if not status:
                Logger().log("Data validation failed due to column mismatch.", level="error")
                raise CustomException("Data validation failed due to column mismatch.", sys)

            # Validate the data drift
            base_df = df.sample(frac=0.5, random_state=42)  # Using a sample as base
            current_df = df.drop(base_df.index)
            status = self.detect_data_drift1(base_df, current_df, self._schema_config['numeric_columns'], self._schema_config['categorical_columns'])
            if not status:
                Logger().log("Data validation failed due to data drift.", level="error")
                raise CustomException("Data validation failed due to data drift.", sys)

            # Validate the features
            feature_report = self.detect_features(df)
            if feature_report["missing_numeric"] or feature_report["missing_categorical"]:
                status = False
                Logger().log(f"Data validation failed due to missing features: {feature_report}", level="error")
                raise CustomException(f"Data validation failed due to missing features: {feature_report}", sys)



            # If all validations pass, save the validated data to the validated directory
            validated_data_path = self.data_validation_config.validated_data_path
            os.makedirs(os.path.dirname(validated_data_path), exist_ok=True)
            df.to_parquet(validated_data_path, index=False)

            # Create the artifact
            data_validation_artifact = DataValidationArtifact(
                valid_status=status,
                validated_data_path=self.data_validation_config.validated_data_path,
                invalidated_data_path=self.data_validation_config.invalidated_data_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            Logger().log("Data validation completed successfully.")
            return data_validation_artifact
        
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in initiate_data_validation: {err}", level="error")
            raise err