import pandas as pd
import joblib
import mlflow

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# 1. Load dataset
df = pd.read_csv("data/customers.csv")

print("Dataset loaded successfully!")
print(df.head())


# 2. Select features
features = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency"
]

X = df[features]


# 3. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. Start MLflow experiment
mlflow.set_experiment("Customer-Segmentation")


with mlflow.start_run():

    # 5. Create K-Means model
    n_clusters = 3

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    # 6. Train model
    clusters = model.fit_predict(X_scaled)

    # 7. Add cluster to dataset
    df["Segment"] = clusters

    # 8. Log experiment information
    mlflow.log_param("algorithm", "KMeans")
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_metric("inertia", model.inertia_)

    # 9. Save model
    joblib.dump(
        model,
        "models/kmeans_model.pkl"
    )

    # 10. Save scaler
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    # 11. Save segmented data
    df.to_csv(
        "data/segmented_customers.csv",
        index=False
    )


print("\nModel trained successfully!")
print("\nCustomer Segments:")
print(df[["CustomerID", "Segment"]])

print("\nModel saved successfully!")