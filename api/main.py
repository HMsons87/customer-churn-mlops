from fastapi import FastAPI
from api.schemas import (
    CustomerInput,
    PredictionResponse,
)
from src.models.predict import predict


# ==========================================
# CREATE APP
# ==========================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability",
    version="1.0.0",
)


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():
    return {
        "message":
            "Customer Churn API is running"
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_endpoint(
    customer: CustomerInput,
):
    result = predict(
        customer.dict()
    )

    return {
        "prediction": int(result["prediction"]),
        "churn_probability": float(result["churn_probability"])
    }