# from flask import Flask, render_template, request
# import mlflow
# import pickle
# import os
# import pandas as pd
# from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
# import time
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# import string
# import re
# import dagshub

# import warnings
# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")

# def lemmatization(text):
#     """Lemmatize the text."""
#     lemmatizer = WordNetLemmatizer()
#     text = text.split()
#     text = [lemmatizer.lemmatize(word) for word in text]
#     return " ".join(text)

# def remove_stop_words(text):
#     """Remove stop words from the text."""
#     stop_words = set(stopwords.words("english"))
#     text = [word for word in str(text).split() if word not in stop_words]
#     return " ".join(text)

# def removing_numbers(text):
#     """Remove numbers from the text."""
#     text = ''.join([char for char in text if not char.isdigit()])
#     return text

# def lower_case(text):
#     """Convert text to lower case."""
#     text = text.split()
#     text = [word.lower() for word in text]
#     return " ".join(text)

# def removing_punctuations(text):
#     """Remove punctuations from the text."""
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = text.replace('؛', "")
#     text = re.sub('\s+', ' ', text).strip()
#     return text

# def removing_urls(text):
#     """Remove URLs from the text."""
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def remove_small_sentences(df):
#     """Remove sentences with less than 3 words."""
#     for i in range(len(df)):
#         if len(df.text.iloc[i].split()) < 3:
#             df.text.iloc[i] = np.nan

# def normalize_text(text):
#     text = lower_case(text)
#     text = remove_stop_words(text)
#     text = removing_numbers(text)
#     text = removing_punctuations(text)
#     text = removing_urls(text)
#     text = lemmatization(text)

#     return text

# # Below code block is for local use
# # -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/premmotetech1/MLOPS_CAPSTONE.mlflow')
# dagshub.init(repo_owner='premmotetech1', repo_name='MLOPS_CAPSTONE', mlflow=True)
# # -------------------------------------------------------------------------------------

# # Below code block is for production use
# # -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
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


# # Initialize Flask app
# app = Flask(__name__)

# # from prometheus_client import CollectorRegistry

# # Create a custom registry
# registry = CollectorRegistry()

# # Define your custom metrics using this registry
# REQUEST_COUNT = Counter(
#     "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
# )
# REQUEST_LATENCY = Histogram(
#     "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
# )
# PREDICTION_COUNT = Counter(
#     "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
# )

# # ------------------------------------------------------------------------------------------
# # Model and vectorizer setup
# model_name = "my_model"
# def get_latest_model_version(model_name):
#     client = mlflow.MlflowClient()
#     latest_version = client.get_latest_versions(model_name, stages=["Production"])
#     if not latest_version:
#         latest_version = client.get_latest_versions(model_name, stages=["None"])
#     return latest_version[0].version if latest_version else None

# model_version = get_latest_model_version(model_name)
# model_uri = f'models:/{model_name}/{model_version}'
# print(f"Fetching model from: {model_uri}")
# model = mlflow.pyfunc.load_model(model_uri)
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# # Routes
# @app.route("/")
# def home():
#     REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
#     start_time = time.time()
#     response = render_template("index.html", result=None)
#     REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
#     return response

# @app.route("/predict", methods=["POST"])
# def predict():
#     REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
#     start_time = time.time()

#     text = request.form["text"]
#     # Clean text
#     text = normalize_text(text)
#     # Convert to features
#     features = vectorizer.transform([text])
#     features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

#     # Predict
#     result = model.predict(features_df)
#     prediction = result[0]

#     # Increment prediction count metric
#     PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

#     # Measure latency
#     REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

#     return render_template("index.html", result=prediction)

# @app.route("/metrics", methods=["GET"])
# def metrics():
#     """Expose only custom Prometheus metrics."""
#     return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# if __name__ == "__main__":
#     # app.run(debug=True) # for local use
#     app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker

from flask import Flask, render_template, request
import mlflow
from mlflow.artifacts import download_artifacts
import pickle
import os
import shutil  
import pandas as pd
import numpy as np  
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# from flask_app.metrics import metrics
import string
import re
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# Below code block is for local use
# # -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/premmotetech1/MLOPS_CAPSTONE.mlflow')
# dagshub.init(repo_owner='premmotetech1', repo_name='MLOPS_CAPSTONE', mlflow=True)
# # -------------------------------------------------------------------------------------

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
# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
# ------------------------------------------------------------------------------------------
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from DagsHub: {model_uri}")

# Pre-initialize model globally
model = None

try:
    # 1. Download the artifacts folder from DagsHub
    remote_cache_path = download_artifacts(artifact_uri=model_uri)

    # 2. Copy it locally to your project directory
    local_model_dir = os.path.join(os.getcwd(), "local_model_cache")
    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)
    shutil.copytree(remote_cache_path, local_model_dir)

    # 3. Search recursively for the serialized model file
    model_file_found = None
    for root, dirs, files in os.walk(local_model_dir):
        for file in files:
            if file in ["model.pkl", "model.bin", "model.pkl.gz"]:
                model_file_found = os.path.join(root, file)
                break
        if model_file_found:
            break

    if model_file_found:
        print(f"Found model binary at: {model_file_found}")
        with open(model_file_found, 'rb') as f:
            model = pickle.load(f)
        print("Model successfully loaded into memory!")
    else:
        raise FileNotFoundError("Could not find model.pkl or model.bin in downloaded artifacts.")

except Exception as e:
    print(f"CRITICAL ERROR loading the model binary: {e}")
    class DummyModel:
        def predict(self, df): return ["Fallback Error: Model missing"]
    model = DummyModel()

# 4. Load your vectorizer file safely
pkl_file_path = 'models/vectorizer.pkl'
if os.path.exists(pkl_file_path):
    vectorizer = pickle.load(open(pkl_file_path, 'rb'))
    print("Vectorizer loaded successfully.")
else:
    print(f"Vectorizer file not found: {pkl_file_path}")

# ------------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    # Clean text using identical preprocessing pipeline
    text = normalize_text(text)
    
    # Convert to features
    features = vectorizer.transform([text])
    
    # FIXED: Use the actual word tokens as column names to align with training data format
    # Instead of string integers like "0", "1", "2" which scramble the feature ordering.
    try:
        feature_names = vectorizer.get_feature_names_out()
        features_df = pd.DataFrame(features.toarray(), columns=feature_names)
        
        # Make the prediction
        result = model.predict(features_df)
    except Exception:
        # Fallback if your original model was trained strictly on raw arrays without feature names
        result = model.predict(features.toarray())
        
    prediction = result[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)