# Customer Segmentation MLOps

## Project Overview

This project implements an end-to-end MLOps pipeline for customer segmentation using K-Means clustering.

The system takes customer information such as age, income, spending score, and purchase frequency and assigns customers to different segments.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- DVC
- FastAPI
- Docker
- GitHub Actions
- Evidently AI

## MLOps Pipeline

Data
↓
Data Processing
↓
K-Means Model Training
↓
MLflow Experiment Tracking
↓
DVC Dataset Versioning
↓
FastAPI API
↓
Docker Container
↓
GitHub Actions CI
↓
Evidently Data Monitoring

## Project Structure

```text
customer-segmentation-mlops/
│
├── data/
│   ├── customers.csv
│   └── segmented_customers.csv
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── src/
│   └── train.py
│
├── app/
│   └── main.py
│
├── monitoring.py
├── Dockerfile
├── .gitignore
├── README.md
│
└── .github/
    └── workflows/
        └── ci.yml