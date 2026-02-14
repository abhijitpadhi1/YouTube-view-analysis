from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


@dataclass
class DataValidationArtifact:
    valid_status: bool
    valid_data_path: str
    invalid_data_path: str
    drift_report_file_path: str