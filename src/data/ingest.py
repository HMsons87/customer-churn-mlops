import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_PATH
from src.logger import logger


def check_file_exists(file_path):
    """
    Check if dataset file exists.
    """
    if not Path(file_path).exists():
        logger.error(f"Dataset not found: {file_path}")
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    logger.info("Dataset file found.")


def load_dataset():
    """
    Load CSV dataset into pandas DataFrame.
    """
    try:
        logger.info("Loading dataset...")

        df = pd.read_csv(RAW_DATA_PATH)

        logger.info(f"Dataset loaded successfully.")
        logger.info(f"Shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def show_basic_info(df):
    """
    Print quick dataset overview.
    """
    logger.info("Showing first 5 rows:")
    print(df.head())

    logger.info("Column names:")
    print(df.columns.tolist())

    logger.info("Missing values:")
    print(df.isnull().sum())


def run_ingestion():
    """
    Full ingestion pipeline.
    """
    logger.info("Starting data ingestion...")

    check_file_exists(RAW_DATA_PATH)

    df = load_dataset()

    show_basic_info(df)

    logger.info("Data ingestion completed.")

    return df


if __name__ == "__main__":
    run_ingestion()