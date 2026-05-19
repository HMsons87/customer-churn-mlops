import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PREPROCESSOR_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
)

from src.logger import logger


# ======================================================
# 1. LOAD DATA
# ======================================================

def load_data():
    """
    Load train and test CSV files.
    """
    logger.info("Loading train/test data...")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    logger.info(
        f"Train shape: {train_df.shape}"
    )
    logger.info(
        f"Test shape: {test_df.shape}"
    )

    return train_df, test_df


# ======================================================
# 2. SPLIT FEATURES AND TARGET
# ======================================================

def split_features_target(df):
    """
    Separate X and y.
    """
    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    if y.dtype == 'object':
        y = y.map({"Yes": 1, "No": 0})

    return X, y


# ======================================================
# 3. LOAD PREPROCESSOR
# ======================================================

def load_preprocessor():
    """
    Load saved preprocessing pipeline.
    """
    logger.info(
        "Loading preprocessor..."
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    logger.info(
        "Preprocessor loaded."
    )

    return preprocessor


# ======================================================
# 4. TRANSFORM DATA
# ======================================================

def transform_data(
    preprocessor,
    X_train,
    X_test,
):
    """
    Transform train and test data.
    """
    logger.info(
        "Transforming data..."
    )

    X_train_processed = (
        preprocessor.transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    logger.info(
        "Transformation complete."
    )

    return (
        X_train_processed,
        X_test_processed,
    )


# ======================================================
# 5. EVALUATE MODEL
# ======================================================

def evaluate_model(model, X_test, y_test):
    """
    Calculate metrics.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions), 
        "recall": recall_score(y_test, predictions),       
        "f1": f1_score(y_test, predictions),         
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return metrics


# ======================================================
# 6. TRAIN MODELS
# ======================================================

def train_models(X_train, y_train):
    """
    Train 3 models.
    """
    logger.info("Training models...")

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=RANDOM_STATE, 
            eval_metric="logloss"
        ),
    }

    trained_models = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # هندرب كل الموديلات بنفس الطريقة مباشرة
        model.fit(X_train, y_train)

        trained_models[name] = model
        logger.info(f"{name} trained.")

    return trained_models


# ======================================================
# 7. COMPARE MODELS
# ======================================================

def compare_models(
    trained_models,
    X_test,
    y_test,
):
    """
    Compare model metrics.
    """
    logger.info(
        "Evaluating models..."
    )

    best_model = None
    best_score = 0

    for name, model in (
        trained_models.items()
    ):
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        logger.info(
            f"{name} metrics:"
        )

        for metric, value in (
            metrics.items()
        ):
            logger.info(
                f"{metric}: "
                f"{value:.4f}"
            )

        # choose best by ROC-AUC
        if (
            metrics["roc_auc"]
            > best_score
        ):
            best_score = (
                metrics["roc_auc"]
            )
            best_model = model

    logger.info(
        f"Best model ROC-AUC: "
        f"{best_score:.4f}"
    )

    return best_model


# ======================================================
# 8. SAVE MODEL
# ======================================================

def save_model(
    model,
):
    """
    Save best model.
    """
    logger.info(
        "Saving best model..."
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    logger.info(
        f"Saved: {MODEL_PATH}"
    )


# ======================================================
# 9. MAIN PIPELINE
# ======================================================

def run_training():
    """
    Full training pipeline.
    """
    logger.info(
        "Starting model training..."
    )

    # Load data
    train_df, test_df = (
        load_data()
    )

    # Split
    X_train, y_train = (
        split_features_target(
            train_df
        )
    )

    X_test, y_test = (
        split_features_target(
            test_df
        )
    )

    # Load preprocessor
    preprocessor = (
        load_preprocessor()
    )

    # Transform
    (
        X_train_processed,
        X_test_processed,
    ) = transform_data(
        preprocessor,
        X_train,
        X_test,
    )

    # Train
    trained_models = (
        train_models(
            X_train_processed,
            y_train,
        )
    )

    # Compare
    best_model = (
        compare_models(
            trained_models,
            X_test_processed,
            y_test,
        )
    )

    # Save
    save_model(
        best_model
    )

    logger.info(
        "Training completed."
    )


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    run_training()