import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE1 = BASE_DIR / "data" / "raw" / "maintenance_work_orders.csv"
RAW_FILE2 = BASE_DIR / "data" / "raw" / "equipment_master.csv"

def extract(file_path1, file_path2):

    try:
        logging.info("Starting work orders extraction")
        logging.info("Starting equipment master extraction")

        # Check source files
        for file_path in [file_path1, file_path2]:

            if not file_path.exists():
                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

        # Extract datasets
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)

        # Validate datasets
        if df1.empty:
            raise ValueError("Work orders dataset is empty")

        if df2.empty:
            raise ValueError("Equipment master dataset is empty")

        # Log extraction results
        logging.info(
            f"Work orders extracted: {len(df1)} records, "
            f"{len(df1.columns)} columns"
        )

        logging.info(
            f"Equipment extracted: {len(df2)} records, "
            f"{len(df2.columns)} columns"
        )

        return df1, df2

    except Exception:
        logging.exception("Data extraction failed")
        raise

if __name__ == "__main__":
    # Unpack both dataframes returned by the extract function
    df_1, df_2 = extract(RAW_FILE1, RAW_FILE2)

    print("--- Work Orders Head ---")
    print(df_1.head())
    
    print("\n--- Equipment Master Head ---")
    print(df_2.head())

    print("\nDataset 1 shape:", df_1.shape)
    print(f"Successfully extracted {len(df_1)} records")
    print("Columns:", list(df_1.columns))
    print("Number of columns:", len(df_1.columns))

    print("\nDataset 2 shape:", df_2.shape)
    print(f"Successfully extracted {len(df_2)} records")
    print("Columns:", list(df_2.columns))
    print("Number of columns:", len(df_2.columns))








   
