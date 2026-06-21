# # register model

# import json
# import mlflow
# import logging
# from src.logger import logging
# import os
# import dagshub

# import warnings
# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")

# # Below code block is for production use
# # -------------------------------------------------------------------------------------
# # # Set up DagsHub credentials for MLflow tracking
# # dagshub_token = os.getenv("CAPSTONE_TEST")
# # if not dagshub_token:
# #     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# # dagshub_url = "https://dagshub.com"
# # repo_owner = "vikashdas770"
# # repo_name = "YT-Capstone-Project"
# # # Set up MLflow tracking URI
# # mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# # -------------------------------------------------------------------------------------


# # Below code block is for local use
# # -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/premmotetech1/MLOPS_CAPSTONE.mlflow')
# dagshub.init(repo_owner='premmotetech1', repo_name='MLOPS_CAPSTONE', mlflow=True)
# # -------------------------------------------------------------------------------------


# def load_model_info(file_path: str) -> dict:
#     """Load the model info from a JSON file."""
#     try:
#         with open(file_path, 'r') as file:
#             model_info = json.load(file)
#         logging.debug('Model info loaded from %s', file_path)
#         return model_info
#     except FileNotFoundError:
#         logging.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error occurred while loading the model info: %s', e)
#         raise

# def register_model(model_name: str, model_info: dict):
#     """Register the model to the MLflow Model Registry."""
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
#         # Register the model
#         model_version = mlflow.register_model(model_uri, model_name)
        
#         # Transition the model to "Staging" stage
#         client = mlflow.tracking.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage="Staging"
#         )
        
#         logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
#     except Exception as e:
#         logging.error('Error during model registration: %s', e)
#         raise

# def main():
#     try:
#         model_info_path = 'reports/experiment_info.json'
#         model_info = load_model_info(model_info_path)
        
#         model_name = "my_model"
#         register_model(model_name, model_info)
#     except Exception as e:
#         logging.error('Failed to complete the model registration process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()

import json
import mlflow
import logging
import os
import dagshub
import warnings
from src.logger import logging

# Suppress system and package deprecation warnings cleanly
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# # --- MLflow / DagsHub Configuration ---
# mlflow.set_tracking_uri('https://dagshub.com/premmotetech1/MLOPS_CAPSTONE.mlflow')
# dagshub.init(repo_owner='premmotetech1', repo_name='MLOPS_CAPSTONE', mlflow=True)
import os
import mlflow

# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")

# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env locally

dagshub_token = os.getenv("CAPSTONE_TEST")

if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "premmotetech1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "premmotetech1"
repo_name = "MLOPS_CAPSTONE"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

def load_model_info(file_path: str) -> dict:
    """Load the model run info using a Windows normalized path string."""
    try:
        safe_path = os.path.normpath(file_path)
        with open(safe_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info metadata loaded successfully from %s', safe_path)
        return model_info
    except FileNotFoundError:
        logging.error('Experiment metadata file not found at: %s.', file_path)
        raise


def register_model(model_name: str, model_info: dict):
    """Verify artifacts on DagsHub and register via robust pathing parsing."""
    try:
        run_id = model_info["run_id"]
        model_path = model_info.get("model_path", "model")

        client = mlflow.tracking.MlflowClient()

        # 1. Fetch remote artifacts from DagsHub to verify existence
        logging.info("Checking remote artifacts for Run ID: %s", run_id)
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [artifact.path for artifact in artifacts]

        logging.info("Available artifacts found on DagsHub: %s", artifact_paths)

        if not any(model_path in path for path in artifact_paths):
            raise ValueError(f"Model artifact path '{model_path}' not found on remote server.")

        # 2. Get the explicit storage path of the run to bypass flavor schema checks
        run_info = client.get_run(run_id)
        artifact_uri = run_info.info.artifact_uri
        
        # Build a robust direct location URI to the model directory
        model_uri = f"{artifact_uri}/{model_path}"
        logging.info("Registering model into MLflow Registry using direct URI: %s", model_uri)

        # 3. Create a model version directly via the tracking client to bypass 'logged_model' constraints
        logging.info("Creating model version entry...")
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        # 4. Modern Alias Assignment to Staging
        logging.info("Assigning stage alias 'Staging' to model version %s...", model_version.version)
        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=str(model_version.version)
        )

        logging.info(
            "Model '%s' version %s registered and successfully transitioned to 'Staging'.",
            model_name, model_version.version
        )

    except Exception as e:
        logging.error("Error encountered during model registration stage: %s", e)
        raise


def main():
    try:
        # model_info_path = r"C:\Users\motep\Desktop\MLOPS\MLOPS_CAPSTONE\reports\experiment_info.json"
        # model_info = load_model_info(model_info_path)
        
        # model_name = "my_model"
        # register_model(model_name, model_info)
        # print("\n🎉 Success! Model registration stage executed successfully without errors!")
        model_info_path = os.path.join("reports", "experiment_info.json")

        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)

        print("\n🎉 Success! Model registration stage executed successfully without errors!")
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"\nExecution Failed: {e}")


if __name__ == '__main__':
    main()