import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from src.logger import logger


# ======================================================
# 1. LOAD DATASET
# ======================================================

def load_dataset():
    """
    Load raw churn dataset.
    """
    logger.info("Loading raw dataset for preprocessing...")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info(f"Dataset loaded. Shape: {df.shape}")

    return df


# ======================================================
# 2. CLEAN TOTALCHARGES COLUMN
# ======================================================

def fix_total_charges(df):
    """
    Convert TotalCharges to numeric.
    Blank spaces become NaN.
    """
    logger.info("Fixing TotalCharges column...")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    logger.info(
        "TotalCharges converted to numeric."
    )

    return df


# ======================================================
# 3. SPLIT FEATURES AND TARGET
# ======================================================

def split_features_target(df):
    """
    Separate X and y.
    """
    logger.info(
        "Separating features and target..."
    )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    logger.info(
        f"Features shape: {X.shape}"
    )
    logger.info(
        f"Target shape: {y.shape}"
    )

    return X, y


# ======================================================
# 4. IDENTIFY COLUMN TYPES
# ======================================================

def get_column_types(X):
    """
    Find numeric and categorical columns.
    """
    logger.info(
        "Identifying column types..."
    )

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    logger.info(
        f"Numeric columns: {numeric_features}"
    )
    logger.info(
        f"Categorical columns: {categorical_features}"
    )

    return (
        numeric_features,
        categorical_features,
    )


# ======================================================
# 5. BUILD PREPROCESSING PIPELINE
# ======================================================

def build_preprocessor(
    numeric_features,
    categorical_features,
):
    """
    Create preprocessing pipeline.
    """

    logger.info(
        "Building preprocessing pipeline..."
    )

    # Numeric pipeline
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    logger.info(
        "Preprocessing pipeline built."
    )

    return preprocessor


# ======================================================
# 6. TRAIN / TEST SPLIT
# ======================================================

def split_train_test(X, y):
    """
    Split into training and testing sets.
    """
    logger.info(
        "Splitting train/test data..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    logger.info(
        f"Train shape: {X_train.shape}"
    )
    logger.info(
        f"Test shape: {X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ======================================================
# 7. FIT + TRANSFORM DATA
# ======================================================

def transform_data(
    preprocessor,
    X_train,
    X_test,
):
    """
    Fit preprocessor on train data
    and transform both train/test.
    """
    logger.info(
        "Fitting preprocessing pipeline..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    logger.info(
        "Data transformed successfully."
    )

    return (
        X_train_processed,
        X_test_processed,
    )


# ======================================================
# 8. SAVE PREPROCESSOR
# ======================================================

def save_preprocessor(
    preprocessor,
):
    """
    Save preprocessing pipeline.
    """
    logger.info(
        "Saving preprocessor..."
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    logger.info(
        f"Saved: {PREPROCESSOR_PATH}"
    )


# ======================================================
# 9. SAVE TRAIN / TEST DATA
# ======================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Save train/test CSVs.
    """
    logger.info(
        "Saving processed datasets..."
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(
        TRAIN_DATA_PATH,
        index=False,
    )

    test_df.to_csv(
        TEST_DATA_PATH,
        index=False,
    )

    logger.info(
        "Processed datasets saved."
    )


# ======================================================
# 10. MAIN PIPELINE
# ======================================================

def run_preprocessing():
    """
    Full preprocessing pipeline.
    """
    logger.info(
        "Starting preprocessing..."
    )

    # Load
    df = load_dataset()

    # Fix TotalCharges
    df = fix_total_charges(df)

    # Split X/y
    X, y = split_features_target(df)

    # Detect column types
    (
        numeric_features,
        categorical_features,
    ) = get_column_types(X)

    # Build pipeline
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    # Train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test(
        X,
        y,
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

    # Save preprocessor
    save_preprocessor(
        preprocessor
    )

    # Save split data
    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    logger.info(
        "Preprocessing completed."
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
    )


# ======================================================
# RUN SCRIPT
# ======================================================

if __name__ == "__main__":
    run_preprocessing()