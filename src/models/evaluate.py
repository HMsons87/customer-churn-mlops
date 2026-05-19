import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.config import (
    TEST_DATA_PATH,
    PREPROCESSOR_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    ROC_CURVE_PATH,
    FEATURE_IMPORTANCE_PATH,
)

from src.logger import logger


# ======================================================
# 1. LOAD TEST DATA
# ======================================================

def load_test_data():
    """
    Load test dataset.
    """
    logger.info(
        "Loading test data..."
    )

    df = pd.read_csv(
        TEST_DATA_PATH
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[
        TARGET_COLUMN
    ]

    logger.info(
        f"Test shape: {df.shape}"
    )

    return X, y


# ======================================================
# 2. LOAD MODEL + PREPROCESSOR
# ======================================================

def load_artifacts():
    """
    Load trained model
    and preprocessor.
    """
    logger.info(
        "Loading model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    logger.info(
        "Loading preprocessor..."
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return model, preprocessor


# ======================================================
# 3. TRANSFORM TEST DATA
# ======================================================

def transform_test_data(
    X,
    preprocessor,
):
    """
    Transform test data.
    """
    logger.info(
        "Transforming test data..."
    )

    X_processed = (
        preprocessor.transform(
            X
        )
    )

    return X_processed


# ======================================================
# 4. GENERATE PREDICTIONS
# ======================================================

def generate_predictions(
    model,
    X_processed,
):
    """
    Predict labels and probabilities.
    """
    logger.info(
        "Generating predictions..."
    )

    predictions = (
        model.predict(
            X_processed
        )
    )

    probabilities = (
        model.predict_proba(
            X_processed
        )[:, 1]
    )

    return (
        predictions,
        probabilities,
    )


# ======================================================
# 5. SAVE CLASSIFICATION REPORT
# ======================================================

def save_classification_report(
    y_true,
    y_pred,
):
    """
    Save classification report.
    """
    logger.info(
        "Saving classification report..."
    )

    report = (
        classification_report(
            y_true,
            y_pred,
        )
    )

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            report
        )

    logger.info(
        f"Saved: "
        f"{CLASSIFICATION_REPORT_PATH}"
    )


# ======================================================
# 6. SAVE CONFUSION MATRIX
# ======================================================

def save_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Save confusion matrix image.
    """
    logger.info(
        "Saving confusion matrix..."
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = (
        ConfusionMatrixDisplay(
            confusion_matrix=cm
        )
    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
    )

    plt.close()

    logger.info(
        f"Saved: "
        f"{CONFUSION_MATRIX_PATH}"
    )


# ======================================================
# 7. SAVE ROC CURVE
# ======================================================

def save_roc_curve(
    y_true,
    probabilities,
):
    """
    Save ROC curve.
    """
    logger.info(
        "Saving ROC curve..."
    )

    y_binary = (
        (y_true == "Yes")
        .astype(int)
    )

    RocCurveDisplay.from_predictions(
        y_binary,
        probabilities,
    )

    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_PATH,
        dpi=300,
    )

    plt.close()

    logger.info(
        f"Saved: "
        f"{ROC_CURVE_PATH}"
    )


# ======================================================
# 8. FEATURE IMPORTANCE
# ======================================================

def save_feature_importance(
    model,
):
    """
    Save feature importance
    if supported.
    """
    logger.info(
        "Checking feature importance..."
    )

    if not hasattr(
        model,
        "feature_importances_",
    ):
        logger.info(
            "Model has no "
            "feature_importances_. "
            "Skipping."
        )
        return

    importance = (
        model.feature_importances_
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        range(
            len(
                importance
            )
        ),
        importance,
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_PATH,
        dpi=300,
    )

    plt.close()

    logger.info(
        f"Saved: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )


# ======================================================
# 9. MAIN PIPELINE
# ======================================================

def run_evaluation():
    """
    Full evaluation pipeline.
    """
    logger.info(
        "Starting model evaluation..."
    )

    # Load test data
    X_test, y_test = (
        load_test_data()
    )

    # Load model/preprocessor
    (
        model,
        preprocessor,
    ) = load_artifacts()

    # Transform
    X_processed = (
        transform_test_data(
            X_test,
            preprocessor,
        )
    )

    # Predict
    (
        predictions,
        probabilities,
    ) = generate_predictions(
        model,
        X_processed,
    )

    # Save artifacts
    save_classification_report(
        y_test,
        predictions,
    )

    save_confusion_matrix(
        y_test,
        predictions,
    )

    save_roc_curve(
        y_test,
        probabilities,
    )

    save_feature_importance(
        model
    )

    logger.info(
        "Evaluation completed."
    )


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    run_evaluation()