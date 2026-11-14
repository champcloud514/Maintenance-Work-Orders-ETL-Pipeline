import pandas as pd
from pathlib import Path
import logging


# ==================================
# 1. CONFIGURE LOGGING
# ==================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ==================================
# 2. DEFINE FILE PATHS
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent

ORDERS_VALID_FILE = BASE_DIR / "data" / "processed" / "clean_work_orders.csv"

ORDERS_REJECTED_FILE = BASE_DIR / "data" / "processed" / "rejected_work_orders.csv"

EQUIPMENT_VALID_FILE = BASE_DIR / "data" / "processed" / "clean_equipment_master.csv"


RAW_FILE1 = BASE_DIR / "data" / "raw" / "maintenance_work_orders.csv"

RAW_FILE2 = BASE_DIR / "data" / "raw" / "equipment_master.csv"


# ==================================
# 3. EXTRACT DATA
# ==================================

def extract(file_path1, file_path2):

    logging.info("Starting data extraction")

    try:
        df1 = pd.read_csv(file_path1)

        df2 = pd.read_csv(file_path2)

        logging.info(f"Work orders extracted: {len(df1)}")

        logging.info(f"Equipment records extracted: {len(df2)}")

    
    except FileNotFoundError:
        logging.error("Input file not found")
        raise
    return df1, df2


# ==================================
# 4. TRANSFORM DATA
# ==================================

def transform_data(df_1, df_2):

    logging.info("Starting data transformation")

    df_1 = df_1.copy()
    df_2 = df_2.copy()

    # Initialize rejected dataframe
    rejected_df = pd.DataFrame()

    # Standardize column names
    df_1.columns = df_1.columns.str.lower().str.strip()

    df_2.columns = df_2.columns.str.lower().str.strip()

    # Standardize text columns
    text_columns = [
        "work_order_id",
        "equipment_id",
        "status",
        "priority",
        "work_type",
        "failure_code",
        "technician"
    ]

    for column in text_columns:

        if column in df_1.columns:

            df_1[column] = (
                df_1[column]
                .astype("string")
                .str.strip()
                .str.upper()
            )

    df_2["equipment_id"] = (
        df_2["equipment_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Convert dates
    for column in ["reported_date", "completed_date"]:

        df_1[column] = pd.to_datetime(
            df_1[column],
            errors="coerce"
        )

    df_2["commissioned_date"] = pd.to_datetime(
        df_2["commissioned_date"],
        errors="coerce"
    )

    # ==================================
    # CHECK 1: DUPLICATE WORK ORDERS
    # ==================================

    df_1 = df_1.sort_values(
        by=["work_order_id", "reported_date"],
        ascending=[True, False]
    )

    duplicate_mask = df_1.duplicated(
        subset=["work_order_id"],
        keep="first"
    )

    rejected_duplicates = df_1[duplicate_mask].copy()

    rejected_duplicates["rejection_reason"] = (
        "Duplicate work_order_id"
    )

    rejected_df = pd.concat(
        [rejected_df, rejected_duplicates],
        ignore_index=True
    )

    df_1 = df_1[~duplicate_mask].copy()

    logging.info(
        f"Duplicate records rejected: {len(rejected_duplicates)}"
    )

    # ==================================
    # CHECK 2: NUMERIC CONVERSION
    # ==================================

    numeric_columns = [
        "labor_hours",
        "parts_cost_usd",
        "downtime_hours"
    ]

    for column in numeric_columns:

        if column in df_1.columns:

            df_1[column] = pd.to_numeric(
                df_1[column],
                errors="coerce"
            )

    # ==================================
    # CHECK 3: NEGATIVE VALUES
    # ==================================

    existing_numeric_cols = [
        col for col in numeric_columns
        if col in df_1.columns
    ]

    if existing_numeric_cols:

        negative_mask = (
            df_1[existing_numeric_cols] < 0
        ).any(axis=1)

        rejected_negatives = df_1[negative_mask].copy()

        rejected_negatives["rejection_reason"] = (
            "Negative numeric values"
        )

        rejected_df = pd.concat(
            [rejected_df, rejected_negatives],
            ignore_index=True
        )

        df_1 = df_1[~negative_mask].copy()

        logging.info(
            f"Negative records rejected: {len(rejected_negatives)}"
        )

    # ==================================
    # CHECK 4: MISSING REQUIRED FIELDS
    # ==================================

    required_columns = [
        "work_order_id",
        "equipment_id",
        "reported_date"
    ]

    missing_mask = df_1[
        required_columns
    ].isna().any(axis=1)

    rejected_missing = df_1[missing_mask].copy()

    rejected_missing["rejection_reason"] = (
        "Missing required fields"
    )

    rejected_df = pd.concat(
        [rejected_df, rejected_missing],
        ignore_index=True
    )

    df_1 = df_1[~missing_mask].copy()

    logging.info(
        f"Missing field records rejected: {len(rejected_missing)}"
    )

    # ==================================
    # CHECK 5: EQUIPMENT ID VALIDATION
    # ==================================

    unmatched_mask = ~df_1["equipment_id"].isin(
        df_2["equipment_id"]
    )

    rejected_unmatched = df_1[unmatched_mask].copy()

    rejected_unmatched["rejection_reason"] = (
        "Equipment ID not found in equipment master"
    )

    rejected_df = pd.concat(
        [rejected_df, rejected_unmatched],
        ignore_index=True
    )

    df_1 = df_1[~unmatched_mask].copy()

    logging.info(
        f"Unmatched equipment records rejected: "
        f"{len(rejected_unmatched)}"
    )

    # ==================================
    # CHECK 6: REMOVE EQUIPMENT DUPLICATES
    # ==================================

    df_2 = df_2.drop_duplicates(
        subset=["equipment_id"],
        keep="first"
    )

    logging.info(
        f"Valid work orders after transformation: {len(df_1)}"
    )

    logging.info(
        f"Total rejected work orders: {len(rejected_df)}"
    )

    logging.info("Transformation completed")

    return df_1, df_2, rejected_df


# ==================================
# 5. VALIDATE DATA
# ==================================

def validation(df):

    logging.info("Starting data validation")

    current_date = pd.Timestamp.today().normalize()

    # Validation 1: Reported date must not be in the future
    future_mask = (
        df["reported_date"].notna()
        & (df["reported_date"] > current_date)
    )

    # Validation 2: Completion date must not be earlier
    # than reported date
    completion_mask = (
        df["completed_date"].notna()
        & (df["completed_date"] < df["reported_date"])
    )

    # Combine both validation conditions
    invalid_mask = future_mask | completion_mask

    # Separate valid and rejected records
    valid_df = df[~invalid_mask].copy()

    rejected_df = df[invalid_mask].copy()

    # Assign rejection reasons
    rejected_df["rejection_reason"] = ""

    rejected_df.loc[
        future_mask[invalid_mask],
        "rejection_reason"
    ] = "Reported date is in the future"

    rejected_df.loc[
        completion_mask[invalid_mask],
        "rejection_reason"
    ] += " Completion date is earlier than reported date"

    logging.info(
        "Valid records: %s",
        len(valid_df)
    )

    logging.info(
        "Rejected records: %s",
        len(rejected_df)
    )

    logging.info("Validation completed")

    return valid_df, rejected_df


# ==================================
# 6. MAIN ETL PIPELINE
# ==================================

if __name__ == "__main__":

    # Extract
    work_orders_df, equipment_df = extract(
        RAW_FILE1,
        RAW_FILE2
    )

    # Transform
    transformed_work_orders, clean_equipment, rejected_transform = (
        transform_data(
            work_orders_df,
            equipment_df
        )
    )

    # Validate
    valid_df, rejected_validation = validation(
        transformed_work_orders
    )

    # Combine all rejected records
    rejected_df = pd.concat(
        [rejected_transform, rejected_validation],
        ignore_index=True
    )

    # ==================================
    # 7. LOAD PROCESSED DATA
    # ==================================

    ORDERS_VALID_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    valid_df.to_csv(
        ORDERS_VALID_FILE,
        index=False
    )

    rejected_df.to_csv(
        ORDERS_REJECTED_FILE,
        index=False
    )

    clean_equipment.to_csv(
        EQUIPMENT_VALID_FILE,
        index=False
    )

    # ==================================
    # 8. FINAL REPORT
    # ==================================

    logging.info("ETL PIPELINE COMPLETED")

    logging.info(f"Final valid work orders: {len(valid_df)}")

    logging.info(f"Final rejected work orders: {len(rejected_df)}")

    logging.info(f"Clean equipment records: {len(clean_equipment)}")

    print("\n========== ETL SUMMARY ==========")

    print("Valid work orders:", len(valid_df))

    print("Rejected work orders:", len(rejected_df))

    print("Clean equipment records:", len(clean_equipment))

    print("\nValid dataset preview:")

    print(valid_df.head())

    print("\nFinal dataset shape:", valid_df.shape)
