import os
from pathlib import Path

# ================================
# PROJECT ROOT
# ================================

# Gets the absolute path of your project root folder
# Example: C:/Users/Hassan/customer-churn-mlops
BASE_DIR = Path(__file__).resolve().parent.parent


# ================================
# ENSURE DIRECTORIES EXIST (التعديل الجديد)
# ================================
# هذا الجزء يضمن إنشاء المجلدات لو لم تكن موجودة لمنع أي أخطاء لاحقاً

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