# import numpy as np
# import pandas as pd
# import pickle
# import json
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# import logging
# import mlflow
# import mlflow.sklearn
# import dagshub
# import os
# from src.logger import logging


# Below code block is for production use
# -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
# #     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# # dagshub_url = "https://dagshub.com"
# # repo_owner = "vikashdas770"
# # repo_name = "YT-Capstone-Project"

# # # Set up MLflow tracking URI
# # mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# # Below code block is for local use
# # -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/premmotetech1/MLOPS_CAPSTONE.mlflow')
# dagshub.init(repo_owner='premmotetech1', repo_name='MLOPS_CAPSTONE', mlflow=True)
# # -------------------------------------------------------------------------------------


# def load_model(file_path: str):
#     """Load the trained model from a file."""
#     try:
#         with open(file_path, 'rb') as file:
#             model = pickle.load(file)
#         logging.info('Model loaded from %s', file_path)
#         return model
#     except FileNotFoundError:
#         logging.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error occurred while loading the model: %s', e)
#         raise

# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         logging.info('Data loaded from %s', file_path)
#         return df
#     except pd.errors.ParserError as e:
#         logging.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
#     """Evaluate the model and return the evaluation metrics."""
#     try:
#         y_pred = clf.predict(X_test)
#         y_pred_proba = clf.predict_proba(X_test)[:, 1]

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_pred_proba)

#         metrics_dict = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'auc': auc
#         }
#         logging.info('Model evaluation metrics calculated')
#         return metrics_dict
#     except Exception as e:
#         logging.error('Error during model evaluation: %s', e)
#         raise

# def save_metrics(metrics: dict, file_path: str) -> None:
#     """Save the evaluation metrics to a JSON file."""
#     try:
#         with open(file_path, 'w') as file:
#             json.dump(metrics, file, indent=4)
#         logging.info('Metrics saved to %s', file_path)
#     except Exception as e:
#         logging.error('Error occurred while saving the metrics: %s', e)
#         raise

# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         model_info = {'run_id': run_id, 'model_path': model_path}
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logging.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logging.error('Error occurred while saving the model info: %s', e)
#         raise

# def main():
#     mlflow.set_experiment("my-dvc-pipeline")
#     with mlflow.start_run() as run:  # Start an MLflow run
#         try:
#             clf = load_model('./models/model.pkl')
#             test_data = load_data('./data/processed/test_bow.csv')
            
#             X_test = test_data.iloc[:, :-1].values
#             y_test = test_data.iloc[:, -1].values

#             metrics = evaluate_model(clf, X_test, y_test)
            
#             save_metrics(metrics, 'reports/metrics.json')
            
#             # Log metrics to MLflow
#             for metric_name, metric_value in metrics.items():
#                 mlflow.log_metric(metric_name, metric_value)
            
#             # Log model parameters to MLflow
#             if hasattr(clf, 'get_params'):
#                 params = clf.get_params()
#                 for param_name, param_value in params.items():
#                     mlflow.log_param(param_name, param_value)
            
#             # Log model to MLflow
#             mlflow.sklearn.log_model(clf, "model")
            
#             # Save model info
#             save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
#             # Log the metrics file to MLflow
#             mlflow.log_artifact('reports/metrics.json')

#         except Exception as e:
#             logging.error('Failed to complete the model evaluation process: %s', e)
#             print(f"Error: {e}")

# if __name__ == '__main__':
#     main()
import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging

warnings.filterwarnings("ignore")

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"

# mlflow.set_tracking_uri('https://dagshub.com/premmotetech1/MLOPS_CAPSTONE.mlflow')
# dagshub.init(repo_owner='premmotetech1', repo_name='MLOPS_CAPSTONE', mlflow=True)

import os
import mlflow

from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env locally

dagshub_token = os.getenv("CAPSTONE_TEST")

if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")

# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "premmotetech1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "premmotetech1"
repo_name = "MLOPS_CAPSTONE"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'auc': float(roc_auc_score(y_test, y_pred_proba))
    }


def main():
    mlflow.set_experiment("my-dvc-pipeline")
    
    model_info_path = r"C:\Users\motep\Desktop\MLOPS\MLOPS_CAPSTONE\reports\experiment_info.json"
    reports_dir = os.path.dirname(model_info_path)
    metrics_path = os.path.join(reports_dir, "metrics.json")
    
    with mlflow.start_run() as run:
        try:
            logging.info("Starting isolated model evaluation stage...")
            
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            os.makedirs(reports_dir, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            mlflow.log_metrics(metrics)
            if hasattr(clf, 'get_params'):
                mlflow.log_params({k: str(v) for k, v in clf.get_params().items()})
            
            mlflow.log_artifact(metrics_path)

            # 🎯 THE FIX: Log using modern MLflow 'mlflow.models.log_model' with direct signature definition
            # This completely bypasses any DVC/Git tracking conflicts on files
            logging.info("Uploading model binaries directly to DagsHub...")
            
            from mlflow.models import ModelSignature
            from mlflow.types.schema import Schema, TensorSpec
            
            # Create a clean input signature layout
            input_schema = Schema([TensorSpec(np.dtype(np.float64), (-1, X_test.shape[1]))])
            output_schema = Schema([TensorSpec(np.dtype(np.int64), (-1,))])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.sklearn.log_model(
                sk_model=clf, 
                artifact_path="model",
                signature=signature
            )
            
            # Verify upload stream integrity
            tracking_client = mlflow.tracking.MlflowClient()
            logging.info("Verifying upload stream integrity with remote registry...")
            
            artifacts = tracking_client.list_artifacts(run.info.run_id)
            artifact_paths = [a.path for a in artifacts]
            logging.info("Remote confirmation received. Artifacts saved: %s", artifact_paths)

            # Only write info if 'model' path exists in the validation array
            if not any("model" in p for p in artifact_paths):
                logging.warning("⚠️ Warning: 'model' folder still missing from remote artifacts. Forcing manual upload fallback...")
                # Fallback: Save model.pkl as a direct artifact backup
                temp_model_dir = os.path.join(reports_dir, "model")
                os.makedirs(temp_model_dir, exist_ok=True)
                with open(os.path.join(temp_model_dir, "model.pkl"), "wb") as f:
                    pickle.dump(clf, f)
                mlflow.log_artifact(temp_model_dir, artifact_path="model")
                
                # Re-verify
                artifacts = tracking_client.list_artifacts(run.info.run_id)
                artifact_paths = [a.path for a in artifacts]

            model_info = {'run_id': run.info.run_id, 'model_path': "model"}
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=4)

            print(f"\n[EVALUATION STAGE COMPLETE] Metadata written. Verified remote artifacts: {artifact_paths}")

        except Exception as e:
            logging.error('Failed processing evaluation step: %s', e)
            print(f"Error: {e}")
            raise e


if __name__ == '__main__':
    main()