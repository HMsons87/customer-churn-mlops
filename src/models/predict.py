import joblib
import pandas as pd

from src.config import (
    PREPROCESSOR_PATH,
    MODEL_PATH,
)
from src.logger import logger


# ======================================================
# 1. LOAD MODEL + PREPROCESSOR
# ======================================================

def load_artifacts():
    """
    Load trained model and preprocessor.
    """
    logger.info(
        "Loading prediction artifacts..."
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    logger.info(
        "Artifacts loaded."
    )

    return (
        preprocessor,
        model,
    )


# ======================================================
# 2. PREPARE INPUT DATA
# ======================================================

def prepare_input(
    customer_data,
):
    """
    Convert dictionary to DataFrame.
    """
    logger.info(
        "Preparing input..."
    )

    df = pd.DataFrame(
        [customer_data]
    )

    # Important:
    # Fix TotalCharges type
    if (
        "TotalCharges"
        in df.columns
    ):
        df[
            "TotalCharges"
        ] = pd.to_numeric(
            df[
                "TotalCharges"
            ],
            errors="coerce",
        )

    return df


# ======================================================
# 3. TRANSFORM INPUT
# ======================================================

def transform_input(
    preprocessor,
    input_df,
):
    """
    Apply preprocessing.
    """
    logger.info(
        "Transforming input..."
    )

    transformed = (
        preprocessor.transform(
            input_df
        )
    )

    return transformed


# ======================================================
# 4. PREDICT
# ======================================================

def predict_churn(
    model,
    transformed_input,
):
    """
    Predict churn class
    and probability.
    """
    logger.info(
        "Running prediction..."
    )

    prediction = (
        model.predict(
            transformed_input
        )[0]
    )

    probability = (
        model.predict_proba(
            transformed_input
        )[0][1]
    )

    result = {
        "prediction":
            prediction,
        "churn_probability":
            round(
                float(
                    probability
                ),
                4,
            ),
    }

    return result


# ======================================================
# 5. MAIN FUNCTION
# ======================================================

def predict(
    customer_data,
):
    """
    Full prediction pipeline.
    """

    (
        preprocessor,
        model,
    ) = load_artifacts()

    input_df = (
        prepare_input(
            customer_data
        )
    )

    transformed = (
        transform_input(
            preprocessor,
            input_df,
        )
    )

    result = (
        predict_churn(
            model,
            transformed,
        )
    )

    return result


# ======================================================
# LOCAL TEST
# ======================================================

if __name__ == "__main__":

    sample_customer = {
        "customerID": "1234-ABCDE",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.00,
        "numAdminTickets": 0,
        "numTechTickets": 1
    }

    result = predict(
        sample_customer
    )

    print(result)