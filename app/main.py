from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(
    title="Customer Segmentation API",
    description="Customer segmentation using K-Means",
    version="1.0"
)

model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.get("/")
def home():
    return {
        "message": "Customer Segmentation API is running"
    }


@app.post("/predict")
def predict(
    age: int,
    income: float,
    spending_score: float,
    frequency: int
):

    customer_data = np.array([
        [age, income, spending_score, frequency]
    ])

    scaled_data = scaler.transform(customer_data)

    segment = model.predict(scaled_data)

    return {
        "customer_segment": int(segment[0])
    }