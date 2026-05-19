import os
from pathlib import Path

# ================================
# PROJECT ROOT
# ================================

BASE_DIR = Path(__file__).resolve().parent.parent


DIRS_TO_CREATE = [
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "data" / "drift_reference",
    BASE_DIR / "models",
    BASE_DIR / "reports"
]

for dir_path in DIRS_TO_CREATE:
    dir_path.mkdir(parents=True, exist_ok=True)


# ================================
# DATA PATHS
# ================================

# Original raw dataset
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"

# Cleaned processed dataset
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "processed_churn.csv"

# Train and test split files
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test.csv"

# Reference data for drift monitoring
DRIFT_REFERENCE_PATH = BASE_DIR / "data" / "drift_reference" / "reference.csv"


# ================================
# MODEL PATHS
# ================================

# Saved preprocessing pipeline
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

# Saved trained model
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# ================================
# REPORT PATHS
# ================================

# Validation report
VALIDATION_REPORT_PATH = BASE_DIR / "reports" / "data_validation.txt"

# Drift report
DRIFT_REPORT_PATH = BASE_DIR / "reports" / "drift_report.html"

# Log file
LOG_FILE_PATH = BASE_DIR / "reports" / "project.log"


# ================================
# TARGET COLUMN
# ================================

TARGET_COLUMN = "Churn"


# ================================
# RANDOM SEED
# ================================

RANDOM_STATE = 42

# ================================
# EVALUATION REPORT PATHS
# ================================

CLASSIFICATION_REPORT_PATH = (
    BASE_DIR / "reports" / "classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    BASE_DIR / "reports" / "confusion_matrix.png"
)

ROC_CURVE_PATH = (
    BASE_DIR / "reports" / "roc_curve.png"
)

FEATURE_IMPORTANCE_PATH = (
    BASE_DIR / "reports" / "feature_importance.png"
)
