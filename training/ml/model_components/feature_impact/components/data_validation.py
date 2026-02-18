import os, sys
import pandas as pd
from typing import Any, Dict
from scipy.stats import skew, kurtosis
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

# Importing custom modules and classes
from training.logging.logger import Logger
from training.exception.exception import CustomException
# Importing constants
from training.ml.constants import SCHEMA_FILE_PATH
# Importing configuration entities
from training.ml.model_components.feature_impact.entity.config_entity import DataValidationConfig
# Importing artifact entities
from training.ml.model_components.feature_impact.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
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
        
    def detect_data_drift(
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
    

    def detect_missing_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detects missing features in the DataFrame based on the expected schema."""
        try:
            numeric_cols = self._schema_config['numeric_columns']
            categorical_cols = self._schema_config['categorical_columns']
            flag_cols = self._schema_config['flag_columns']

            # Check for missing numeric features
            missing_numeric = [col for col in numeric_cols if col not in df.columns]
            # Check for missing categorical features
            missing_categorical = [col for col in categorical_cols if col not in df.columns]
            # Check for missing flag features
            missing_flags = [col for col in flag_cols if col not in df.columns]
            return {
                "numeric": missing_numeric,
                "categorical": missing_categorical,
                "flag": missing_flags
            }
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in detect_missing_features: {err}", level="error")
            raise err
        
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """Initiates the data validation process and returns a DataValidationArtifact."""
        try:
            Logger().log("Starting data validation process.")
            status = True
            # Read the ingested data
            train_file_path = self.data_ingestion_artifact.training_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # # Validate number of columns
            # status = self.validate_number_of_columns(train_df)
            # if not status:
            #     Logger().log("Data validation failed due to column mismatch in training data.", level="error")
            #     raise ValueError("Data validation failed due to column mismatch in training data.")

            # status = self.validate_number_of_columns(test_df)
            # if not status:
            #     Logger().log("Data validation failed due to column mismatch in test data.", level="error")
            #     raise ValueError("Data validation failed due to column mismatch in test data.")

            # Detect missing features
            feature_report_train = self.detect_missing_features(train_df)
            feature_report_test = self.detect_missing_features(test_df)
            if feature_report_train["numeric"] or feature_report_train["categorical"] or feature_report_train["flag"]:
                Logger().log(f"Data validation failed due to missing features in training data: {feature_report_train}", level="error")
                raise ValueError(f"Data validation failed due to missing features in training data: {feature_report_train}")
            if feature_report_test["numeric"] or feature_report_test["categorical"] or feature_report_test["flag"]:
                Logger().log(f"Data validation failed due to missing features in test data: {feature_report_test}", level="error")
                raise ValueError(f"Data validation failed due to missing features in test data: {feature_report_test}")

            # Validate data drift
            numeric_cols = self._schema_config['numeric_columns']
            categorical_cols = self._schema_config['categorical_columns']
            status = self.detect_data_drift(train_df, test_df, numeric_cols, categorical_cols)
            if not status:
                Logger().log("Data validation failed due to data drift between training and test data.", level="error")
                raise ValueError("Data validation failed due to data drift between training and test data.")

            # If all validations pass, save the validated data to the validated directory
            validated_train_file_path = self.data_validation_config.valid_train_file_path
            os.makedirs(os.path.dirname(validated_train_file_path), exist_ok=True)
            train_df.to_parquet(validated_train_file_path, index=False)
            Logger().log(f"Validated training data saved at: {validated_train_file_path}")

            validated_test_file_path = self.data_validation_config.valid_test_file_path
            os.makedirs(os.path.dirname(validated_test_file_path), exist_ok=True)
            test_df.to_parquet(validated_test_file_path, index=False)
            Logger().log(f"Validated test data saved at: {validated_test_file_path}")

            # Create DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                valid_status=status,
                validated_test_data_path=validated_test_file_path,
                validated_training_data_path=validated_train_file_path,
                invalidated_test_data_path=self.data_validation_config.invalid_test_file_path,
                invalidated_training_data_path=self.data_validation_config.invalid_train_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            Logger().log(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in initiate_data_validation: {err}", level="error")
            raise err