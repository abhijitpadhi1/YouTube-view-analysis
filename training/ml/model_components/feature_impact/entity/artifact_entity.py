from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    training_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    valid_status: bool
    validated_training_data_path: str
    validated_test_data_path: str
    invalidated_training_data_path: str
    invalidated_test_data_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class RandomForestMetricArtifact:
    training_score: float
    test_score: float
    feature_importances: list

@dataclass
class SHAPMetricArtifact:
    shap_values: list
    shap_explanation_plot_paths: list

@dataclass
class ModelTrainerArtifact:
    trained_rf_model_file_path: str
    trained_shap_explainer_file_path: str
    trained_shap_values_file_path: str
    rf_metric_artifact: RandomForestMetricArtifact
    shap_metric_artifact: SHAPMetricArtifact