# 🚀 MLOps Capstone Project - End-to-End Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?style=for-the-badge)
![DVC](https://img.shields.io/badge/DVC-Pipeline-green?style=for-the-badge)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?style=for-the-badge&logo=amazonaws)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-black?style=for-the-badge&logo=githubactions)

### ⚡ End-to-End Production Grade MLOps Project

</div>

---

# 📌 Project Overview

This project demonstrates a full MLOps lifecycle including MLflow, DVC, AWS S3, Docker, GitHub Actions, EKS, Prometheus, and Grafana.

## Major Steps

1. Project Setup using Cookiecutter
2. MLflow Tracking with DagsHub
3. DVC Pipeline Creation
4. AWS S3 Remote Storage
5. Flask Application Development
6. Docker Containerization
7. GitHub Actions CI/CD
8. AWS ECR Integration
9. AWS EKS Deployment
10. Prometheus Monitoring
11. Grafana Dashboards

---

# 🏗️ Project Setup

```bash
git clone <repo-url>
cd <repo-name>

conda create -n atlas python=3.10
conda activate atlas

pip install cookiecutter

cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

Rename:

```text
src/models -> src/model
```

---

# 📊 MLflow + DagsHub

```bash
pip install dagshub mlflow
```

- Create DagsHub Repository
- Connect GitHub Repository
- Enable MLflow Tracking
- Run Experiments
- Track Metrics in MLflow UI

---

# 📦 DVC Pipeline

```bash
dvc init

mkdir local_s3

dvc remote add -d mylocal local_s3
```

Create:

- data_ingestion.py
- data_preprocessing.py
- feature_engineering.py
- model_building.py
- model_evaluation.py
- register_model.py

Create:

```text
dvc.yaml
params.yaml
```

Run:

```bash
dvc repro
dvc status
```

---

# ☁️ AWS S3 Integration

```bash
pip install dvc[s3] awscli

aws configure

dvc remote add -d myremote s3://your-bucket-name
```

---

# 🌐 Flask Application

```bash
pip install flask

python app.py
```

---

# 🐳 Docker

```bash
pip install pipreqs

pipreqs . --force

docker build -t capstone-app .

docker run -p 8888:5000 capstone-app

docker run -p 8888:5000 -e CAPSTONE_TEST=<token> capstone-app
```

---

# 🔄 GitHub Actions CI/CD

Create:

```text
.github/workflows/ci.yaml
```

Secrets:

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
AWS_ACCOUNT_ID=
ECR_REPOSITORY=
DAGSHUB_TOKEN=
```

Pipeline:

GitHub Push → Docker Build → ECR Push → Deployment

---

# ☸️ Kubernetes (AWS EKS)

Create Cluster:

```bash
eksctl create cluster --name flask-app-cluster --region us-east-1 --nodegroup-name flask-app-nodes --node-type t3.small --nodes 1
```

Connect:

```bash
aws eks update-kubeconfig --name flask-app-cluster --region us-east-1

kubectl get nodes
kubectl get pods
kubectl get svc
```

Delete Cluster:

```bash
eksctl delete cluster --name flask-app-cluster --region us-east-1
```

---

# 📊 Prometheus

Install:

```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz
```

Run:

```bash
prometheus --config.file=/etc/prometheus/prometheus.yml
```

---

# 📈 Grafana

Install:

```bash
wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb

sudo apt install ./grafana_10.1.5_amd64.deb -y

sudo systemctl start grafana-server

sudo systemctl enable grafana-server
```

Access:

```text
http://<ec2-public-ip>:3000
```

---

# 🧹 AWS Cleanup

```bash
kubectl delete deployment flask-app

kubectl delete service flask-app-service

eksctl delete cluster --name flask-app-cluster --region us-east-1
```

---

# 👨‍💻 Author

Prem Mote

GitHub: https://github.com/moteprem4-web

---

# ⭐ Support

If you like this project, give it a star on GitHub.
