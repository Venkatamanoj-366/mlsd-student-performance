import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# Create Models Directory
# ─────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("data/student_dataset.csv")

# ─────────────────────────────────────────────────────────────
# Features and Target
# ─────────────────────────────────────────────────────────────

X = df.drop("final_score", axis=1)

y = df["final_score"]

# ─────────────────────────────────────────────────────────────
# Encode Categorical Columns
# ─────────────────────────────────────────────────────────────

label_encoders = {}

for column in X.select_dtypes(include=["object"]).columns:

    le = LabelEncoder()

    X[column] = le.fit_transform(X[column])

    label_encoders[column] = le

# ─────────────────────────────────────────────────────────────
# Save Feature Columns
# ─────────────────────────────────────────────────────────────

joblib.dump(

    list(X.columns),

    "models/feature_columns.pkl"

)

# ─────────────────────────────────────────────────────────────
# Train Test Split
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

# ─────────────────────────────────────────────────────────────
# MLflow Tracking
# ─────────────────────────────────────────────────────────────

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(

    "student-performance-prediction"

)

# ─────────────────────────────────────────────────────────────
# Start MLflow Run
# ─────────────────────────────────────────────────────────────

with mlflow.start_run():

    # ─────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────

    model = LinearRegression()

    # ─────────────────────────────────────────────────────────
    # Train Model
    # ─────────────────────────────────────────────────────────

    model.fit(

        X_train,
        y_train

    )

    # ─────────────────────────────────────────────────────────
    # Predictions
    # ─────────────────────────────────────────────────────────

    y_pred = model.predict(X_test)

    # ─────────────────────────────────────────────────────────
    # Regression Metrics
    # ─────────────────────────────────────────────────────────

    mae = mean_absolute_error(

        y_test,
        y_pred

    )

    mse = mean_squared_error(

        y_test,
        y_pred

    )

    rmse = mse ** 0.5

    r2 = r2_score(

        y_test,
        y_pred

    )

    # ─────────────────────────────────────────────────────────
    # Log Parameters
    # ─────────────────────────────────────────────────────────

    mlflow.log_param(

        "model_type",
        "LinearRegression"

    )

    mlflow.log_param(

        "test_size",
        0.2

    )

    mlflow.log_param(

        "random_state",
        42

    )

    # ─────────────────────────────────────────────────────────
    # Log Metrics
    # ─────────────────────────────────────────────────────────

    mlflow.log_metric(

        "mae",
        mae

    )

    mlflow.log_metric(

        "mse",
        mse

    )

    mlflow.log_metric(

        "rmse",
        rmse

    )

    mlflow.log_metric(

        "r2_score",
        r2

    )

    # ─────────────────────────────────────────────────────────
    # Save Model
    # ─────────────────────────────────────────────────────────

    joblib.dump(

        model,

        "models/model.pkl"

    )

    # ─────────────────────────────────────────────────────────
    # Log MLflow Model
    # ─────────────────────────────────────────────────────────

    mlflow.sklearn.log_model(

        model,

        artifact_path="model"

    )

    # ─────────────────────────────────────────────────────────
    # Log Artifacts
    # ─────────────────────────────────────────────────────────

    mlflow.log_artifact(

        "models/model.pkl"

    )

    mlflow.log_artifact(

        "models/feature_columns.pkl"

    )

# ─────────────────────────────────────────────────────────────
# Final Output
# ─────────────────────────────────────────────────────────────

print("\n✅ Model Training Completed")

print(f"\nMAE  : {mae:.4f}")

print(f"MSE  : {mse:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R²   : {r2:.4f}")

print("\n✅ MLflow Logging Completed")
    
print("✅ Model Saved -> models/model.pkl")