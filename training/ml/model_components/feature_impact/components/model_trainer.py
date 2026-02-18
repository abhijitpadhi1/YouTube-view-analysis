import os, sys
from matplotlib.pylab import geometric
import pandas as pd
import mlflow
import dagshub
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# from dagshub.core import init

import shap
from sklearn.ensemble import RandomForestRegressor

# Importing custom modules and classes
from training.logging.logger import Logger
from training.exception.exception import CustomException
# Importing artifact entities
from training.ml.model_components.feature_impact.entity.artifact_entity import (
    DataTransformationArtifact, 
    ModelTrainerArtifact,
    RandomForestMetricArtifact,
    SHAPMetricArtifact
)
# Importing configuration entities
from training.ml.model_components.feature_impact.entity.config_entity import ModelTrainerConfig
# Importing constants
from training.ml.constants import (
    SCHEMA_FILE_PATH,
    MODEL_REGISTRY_ML_FEATURE_IMPACT_MODEL,
    MODEL_TRAINING_TRAINED_RF_MODEL_FILE_NAME,
    MODEL_TRAINING_TRAINED_SHAP_EXPLAINER_FILE_NAME,
    MODEL_TRAINER_SHAP_VALUES_FILE_NAME,
    MODEL_TRAINER_SHAP_EXPLANATION_PLOTS_DIR
)
# Importing utility functions
from training.utils.util import save_object, load_object, read_yaml_file

# Initialize the Dagshub for MLFlow tracking
dagshub.init(repo_owner='abhijitpadhi1', repo_name='YouTube-view-analysis', mlflow=True) # type: ignore

class ModelTrainer:
    def __init__(
            self,
            data_transformation_artifact:DataTransformationArtifact,
            model_trainer_config:ModelTrainerConfig
        ):
        try:
            Logger().log("Initializing ModelTrainer class.")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in ModelTrainer __init__: {err}", level='error')


    def track_mlflow(self, model_name, model, model_metric, model_path) -> None:
        try:
            with mlflow.start_run():
                if model_name == "RandomForest":
                    rf_train_score = model_metric['train_score']
                    rf_test_score = model_metric['test_score']
                    mlflow.log_metric("rf_train_score", rf_train_score)
                    mlflow.log_metric("rf_test_score", rf_test_score)
                    mlflow.sklearn.log_model(model, "model")
                    mlflow.log_artifact(model_path, artifact_path="rf_model")

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in track_mlflow: {err}", level='error')
            raise err
        
    def train_rf_model(
            self, X_train: pd.DataFrame, 
                y_train: pd.Series,
                X_test: pd.DataFrame, 
                y_test: pd.Series
        ) -> RandomForestMetricArtifact:
        try:
            Logger().log("Training Random Forest Regressor model.")

            # Define the Random Forest Regressor with specified hyperparameters
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=50,
                random_state=42,
                n_jobs=-1,
                verbose=2
            )

            # Train the model on the training data
            rf_model.fit(X_train, y_train)

            # Save the trained model to the specified file path
            model_path = self.model_trainer_config.trained_rf_model_file_path
            save_object(file_path=model_path, obj=rf_model)
            Logger().log(f"Trained Random Forest model saved at: {model_path}")

            # Find the training score and test score
            train_score = rf_model.score(X_train, y_train)
            test_score = rf_model.score(X_test, y_test)

            # Track with mlflow
            rf_metric = {
                "train_score": train_score,
                "test_score": test_score
            }
            self.track_mlflow("RandomForest", rf_model, rf_metric, self.model_trainer_config.trained_rf_model_file_path)

            Logger().log(f"Random Forest training score: {train_score}")
            Logger().log(f"Random Forest test score: {test_score}")

            # Get the feature impotance
            feture_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

            # Create the metric artifact
            rf_metric_artifact = RandomForestMetricArtifact(
                training_score = float(train_score),
                test_score = float(test_score),
                feature_importances=feture_importance.tolist()
            )


            # Save the trained model to the model registry directory for serving
            model_registry_path = os.path.join(MODEL_REGISTRY_ML_FEATURE_IMPACT_MODEL, MODEL_TRAINING_TRAINED_RF_MODEL_FILE_NAME)
            save_object(file_path=model_registry_path, obj=rf_model)
            Logger().log(f"Trained Random Forest model saved at model registry path: {model_registry_path}")
            return rf_metric_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in train_rf_model: {err}", level='error')
            raise err
        
    def train_shap_explainer(
            self, X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                model: object
        ) -> SHAPMetricArtifact:
        try:
            Logger().log("Training SHAP Explainer.")

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Save the SHAP explainer to the specified file path
            shap_explainer_path = self.model_trainer_config.trained_shap_explainer_file_path
            save_object(file_path=shap_explainer_path, obj=explainer)
            Logger().log(f"Trained SHAP explainer saved at: {shap_explainer_path}")

            # Save the SHAP explainer to the model registry directory for serving
            shap_explainer_registry_path = os.path.join(MODEL_REGISTRY_ML_FEATURE_IMPACT_MODEL, MODEL_TRAINING_TRAINED_SHAP_EXPLAINER_FILE_NAME)
            save_object(file_path=shap_explainer_registry_path, obj=explainer)
            Logger().log(f"Trained SHAP explainer saved at model registry path: {shap_explainer_registry_path}")

            # Find the SHAP values for the test set
            shap_values = explainer.shap_values(X_test)
            # Save the SHAP values as a pandas DataFrame
            shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
            shap_values_path = self.model_trainer_config.trained_shap_values_file_path
            shap_values_df.to_csv(shap_values_path, index=False)
            Logger().log(f"Trained SHAP values saved at: {shap_values_path}")

            # Save the SHAP values to the model registry directory for serving
            shap_values_registry_path = os.path.join(MODEL_REGISTRY_ML_FEATURE_IMPACT_MODEL, MODEL_TRAINER_SHAP_VALUES_FILE_NAME)
            shap_values_df.to_csv(shap_values_registry_path, index=False)
            Logger().log(f"Trained SHAP values saved at model registry path: {shap_values_registry_path}")

            # Create SHAP explanation plots for the top 10 features
            shap_explanation_plots_dir = self.model_trainer_config.trained_shap_explanation_plots_dir
            os.makedirs(shap_explanation_plots_dir, exist_ok=True)

            # Summary plot
            shap.summary_plot(shap_values, X_test, show=False)
            summary_path = os.path.join(shap_explanation_plots_dir, "summary_plot.png")
            plt.savefig(summary_path, bbox_inches="tight", dpi=300)
            plt.close()

            # Dependence plot - hours_to_trend
            shap.dependence_plot('hours_to_trend', shap_values, X_test, show=False)
            dep1_path = os.path.join(shap_explanation_plots_dir, "dependence_plot_hours_to_trend.png")
            plt.savefig(dep1_path, bbox_inches="tight", dpi=300)
            plt.close()

            # Dependence plot - duration_seconds
            shap.dependence_plot('duration_seconds', shap_values, X_test, show=False)
            dep2_path = os.path.join(shap_explanation_plots_dir, "dependence_plot_duration_seconds.png")
            plt.savefig(dep2_path, bbox_inches="tight", dpi=300)
            plt.close()

            shap_explanation_plot_paths = [summary_path, dep1_path, dep2_path]
            Logger().log(f"SHAP explanation plots saved at: {shap_explanation_plots_dir}")

            # Cerate the SHAP metric artifact
            shap_metric_artifact = SHAPMetricArtifact(
                shap_values=shap_values.tolist(),
                shap_explanation_plot_paths=shap_explanation_plot_paths
            )
            return shap_metric_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in train_shap_explainer: {err}", level='error')
            raise err
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            Logger().log("Initiating model training process.")

            # Load the transformed training and test data
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_df = pd.read_parquet(transformed_train_file_path)
            Logger().log(f"Transformed training data loaded from: {transformed_train_file_path}")

            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_df = pd.read_parquet(transformed_test_file_path)
            Logger().log(f"Transformed test data loaded from: {transformed_test_file_path}")

            # Separate features and target variable
            Logger().log("Separating features and target variable.")
            schema = read_yaml_file(SCHEMA_FILE_PATH)
            target_column = schema['target_column']
            X_train = train_df.drop(columns=target_column)
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=target_column)
            y_test = test_df[target_column]

            # Train the Random Forest model and get the metric artifact
            rf_metric_artifact = self.train_rf_model(X_train, y_train, X_test, y_test)
            # Load the trained Random Forest model
            rf_model = load_object(file_path=self.model_trainer_config.trained_rf_model_file_path)
            # Train the SHAP explainer and get the metric artifact
            shap_metric_artifact = self.train_shap_explainer(X_train, X_test, rf_model)
            # Create the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_rf_model_file_path=self.model_trainer_config.trained_rf_model_file_path,
                trained_shap_explainer_file_path=self.model_trainer_config.trained_shap_explainer_file_path,
                trained_shap_values_file_path=self.model_trainer_config.trained_shap_values_file_path,
                rf_metric_artifact=rf_metric_artifact,
                shap_metric_artifact=shap_metric_artifact
            )
            Logger().log("Model training process completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in initiate_model_trainer: {err}", level='error')
            raise err
        
