from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


@dataclass
class DataValidationArtifact:
    valid_status: bool
    validated_data_path: str
    invalidated_data_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_data_dir: str
    transformed_data_file_path: str
    transformed_object_dir: str | None = None