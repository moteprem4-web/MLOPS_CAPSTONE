# # promote model

# import os
# import mlflow
# from dotenv import load_dotenv
# import os

# load_dotenv()

# def promote_model():
#     # Set up DagsHub credentials for MLflow tracking
#     dagshub_token = os.getenv("CAPSTONE_TEST")
#     if not dagshub_token:
#         raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

#     os.environ["MLFLOW_TRACKING_USERNAME"] = "premmotetech1"
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#     dagshub_url = "https://dagshub.com"
#     repo_owner = "premmotetech1"
#     repo_name = "MLOPS_CAPSTONE"

#     # Set up MLflow tracking URI
#     mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

#     client = mlflow.MlflowClient()

#     model_name = "my_model"
#     # Get the latest version in staging
#     # latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

#     staging_versions = client.get_latest_versions(
#         model_name,
#         stages=["Staging"]
#         )
#     if not staging_versions:
#         raise Exception(
#             f"No model version found in Staging for '{model_name}'"
#     )
#     latest_version_staging = staging_versions[0].version

#     # Archive the current production model
#     prod_versions = client.get_latest_versions(model_name, stages=["Production"])
#     for version in prod_versions:
#         client.transition_model_version_stage(
#             name=model_name,
#             version=version.version,
#             stage="Archived"
#         )

#     # Promote the new model to production
#     client.transition_model_version_stage(
#         name=model_name,
#         version=latest_version_staging,
#         stage="Production"
#     )
#     print(f"Model version {latest_version_staging} promoted to Production")

# if __name__ == "__main__":
#     promote_model()

# promote_model.py

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

def promote_model():
    # Load DagsHub token
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "premmotetech1"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # MLflow Tracking URI
    mlflow.set_tracking_uri(
        "https://dagshub.com/premmotetech1/MLOPS_CAPSTONE.mlflow"
    )

    client = mlflow.MlflowClient()

    model_name = "my_model"

    try:
        # Get model version from alias @Staging
        staging_model = client.get_model_version_by_alias(
            model_name,
            "Staging"
        )

        version = staging_model.version

        print(f"Found Staging model version: {version}")

        # Assign Production alias to this version
        client.set_registered_model_alias(
            model_name,
            "Production",
            version
        )

        print(
            f"✅ Model version {version} promoted to @Production successfully"
        )

    except Exception as e:
        print(f"❌ Promotion failed: {e}")
        raise

if __name__ == "__main__":
    promote_model()
