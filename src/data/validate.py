# import pandas as pd

# from src.config import (
#     RAW_DATA_PATH,
#     TARGET_COLUMN,
#     VALIDATION_REPORT_PATH
# )
# from src.logger import logger
# def load_dataset():
#     """
#     Load dataset for validation.
#     """
#     logger.info("Loading dataset for validation...")

#     df = pd.read_csv(RAW_DATA_PATH)

#     logger.info(f"Dataset loaded. Shape: {df.shape}")

#     return df

# def validate_target_column(df):
#     """
#     Check if target column exists.
#     """
#     logger.info("Checking target column...")

#     if TARGET_COLUMN not in df.columns:
#         logger.error(f"Missing target column: {TARGET_COLUMN}")
#         raise ValueError(
#             f"Missing target column: {TARGET_COLUMN}"
#         )

#     logger.info("Target column found.")

# TARGET_COLUMN = "Churn"

# def check_duplicates(df):
#     """
#     Count duplicate rows.
#     """
#     logger.info("Checking duplicate rows...")

#     duplicate_count = df.duplicated().sum()

#     logger.info(
#         f"Duplicate rows found: {duplicate_count}"
#     )

#     return duplicate_count

# def check_missing_values(df):
#     """
#     Count missing values per column.
#     """
#     logger.info("Checking missing values...")

#     missing = df.isnull().sum()

#     logger.info("Missing value check completed.")

#     return missing

# def check_data_types(df):
#     """
#     Show column data types.
#     """
#     logger.info("Checking data types...")

#     dtypes = df.dtypes

#     logger.info("Data type check completed.")

#     return dtypes

# def validate_target_values(df):
#     """
#     Check target labels.
#     """
#     logger.info("Checking target values...")

#     unique_values = df[TARGET_COLUMN].unique()

#     logger.info(
#         f"Target values found: {unique_values}"
#     )

#     expected_values = {"Yes", "No"}

#     if not set(unique_values).issubset(expected_values):
#         logger.error(
#             "Unexpected target labels found."
#         )
#         raise ValueError(
#             "Unexpected target labels."
#         )

#     logger.info(
#         "Target labels are valid."
#     )

#     return unique_values


# def save_validation_report(
#     duplicates,
#     missing,
#     dtypes,
#     target_values
# ):
#     """
#     Save validation summary to file.
#     """

#     logger.info(
#         "Saving validation report..."
#     )

#     with open(
#         VALIDATION_REPORT_PATH,
#         "w",
#         encoding="utf-8"
#     ) as f:

#         f.write(
#             "DATA VALIDATION REPORT\n"
#         )
#         f.write(
#             "=" * 40 + "\n\n"
#         )

#         f.write(
#             f"Duplicate rows: {duplicates}\n\n"
#         )

#         f.write(
#             "Missing Values:\n"
#         )
#         f.write(
#             str(missing)
#         )
#         f.write("\n\n")

#         f.write(
#             "Data Types:\n"
#         )
#         f.write(
#             str(dtypes)
#         )
#         f.write("\n\n")

#         f.write(
#             "Target Values:\n"
#         )
#         f.write(
#             str(target_values)
#         )

#     logger.info(
#         "Validation report saved."
#     )

# def run_validation():
#     """
#     Full validation pipeline.
#     """
#     logger.info(
#         "Starting data validation..."
#     )

#     df = load_dataset()

#     validate_target_column(df)

#     duplicates = check_duplicates(df)

#     missing = check_missing_values(df)

#     dtypes = check_data_types(df)

#     target_values = validate_target_values(df)

#     save_validation_report(
#         duplicates,
#         missing,
#         dtypes,
#         target_values
#     )

#     logger.info(
#         "Data validation completed."
#     )

# if __name__ == "__main__":
#     run_validation()


import pandas as pd
from src.config import RAW_DATA_PATH, VALIDATION_REPORT_PATH
from src.logger import logger

def validate_data():
    """
    Validates raw data and generates a report.
    Returns True if valid, False otherwise.
    """
    logger.info("Starting data validation...")
    
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        
        # كتابة التقرير
        with open(VALIDATION_REPORT_PATH, 'w') as f:
            f.write("DATA VALIDATION REPORT\n")
            f.write("========================================\n\n")
            
            duplicates = df.duplicated().sum()
            f.write(f"Duplicate rows: {duplicates}\n\n")
            
            f.write("Missing Values:\n")
            f.write(f"{df.isnull().sum()}\n\n")
            
            f.write("Data Types:\n")
            f.write(f"{df.dtypes}\n\n")
            
        logger.info(f"Validation complete. Report saved to {VALIDATION_REPORT_PATH}")
        return True # مهمة جداً عشان الـ Pipeline يعرف إن الخطوة نجحت
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return False

if __name__ == "__main__":
    validate_data()