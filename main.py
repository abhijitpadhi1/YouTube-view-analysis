# Importing necessary modules and classes
import os
import sys

# Import custom modules
from training.exception.exception import CustomException
from training.logging.logger import Logger
# Import component module
from training.data_preparation.pipeline.preparation_pipeline import PreparationPipeline
from training.ml.model_components.feature_impact.pipeline.feature_impact_pipeline import FeatureImpactModelTrainingPipeline


def main():
    try:
        data_preparation = PreparationPipeline()
        data_transformation_artifact = data_preparation.run_data_preparation_pipeline()
        print(f"Data Transformation Artifact: {data_transformation_artifact}")

        feature_impact_model_training_pipeline = FeatureImpactModelTrainingPipeline(data_transformation_artifact)
        model_trainer_artifact = feature_impact_model_training_pipeline.run_feature_impact_model_training_pipeline()
        print(f"Model Trainer Artifact: {model_trainer_artifact}")

    except Exception as e:
        err = CustomException(str(e), sys)
        Logger().log(f"Error in main function: {err}", level="error")
        raise err


if __name__ == "__main__":
    main()
