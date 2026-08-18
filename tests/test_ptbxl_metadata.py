from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptbxl"
    / "ptbxl_database.csv"
)


def test_ptbxl_metadata():

    print("=" * 70)
    print("PTB-XL Metadata Validation")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Check file exists
    # ---------------------------------------------------------

    print("\nMetadata file:")
    print(METADATA_FILE)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata file not found: {METADATA_FILE}"
        )

    print("[OK] Metadata file exists")

    # ---------------------------------------------------------
    # 2. Load metadata
    # ---------------------------------------------------------

    df = pd.read_csv(METADATA_FILE)

    print("\nMetadata loaded successfully.")

    # ---------------------------------------------------------
    # 3. Basic shape
    # ---------------------------------------------------------

    print("\nDataset shape:")
    print(df.shape)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # ---------------------------------------------------------
    # 4. Column names
    # ---------------------------------------------------------

    print("\nColumns:")
    print(df.columns.tolist())

    # ---------------------------------------------------------
    # 5. Required columns
    # ---------------------------------------------------------

    required_columns = [
        "ecg_id",
        "patient_id",
        "recording_date",
        "scp_codes",
        "filename_lr",
    ]

    print("\nRequired column verification:")

    for column in required_columns:

        if column in df.columns:
            print(f"  [OK] {column}")

        else:
            raise ValueError(
                f"Required column missing: {column}"
            )

    # ---------------------------------------------------------
    # 6. ECG ID uniqueness
    # ---------------------------------------------------------

    print("\nECG ID verification:")

    unique_ecg_ids = df["ecg_id"].nunique()

    print("Total ECG IDs:", len(df))
    print("Unique ECG IDs:", unique_ecg_ids)

    if unique_ecg_ids != len(df):
        raise ValueError(
            "ECG IDs are not unique."
        )

    print("[OK] ECG IDs are unique")

    # ---------------------------------------------------------
    # 7. Patient ID
    # ---------------------------------------------------------

    print("\nPatient verification:")

    unique_patients = df["patient_id"].nunique()

    print("Unique patients:", unique_patients)

    # ---------------------------------------------------------
    # 8. Missing values
    # ---------------------------------------------------------

    print("\nMissing values:")

    missing_values = df.isnull().sum()

    print(
        missing_values[
            missing_values > 0
        ]
    )

    # ---------------------------------------------------------
    # 9. Recording paths
    # ---------------------------------------------------------

    print("\nRecording path verification:")

    missing_paths = df["filename_lr"].isnull().sum()

    print("Missing filename_lr:", missing_paths)

    if missing_paths > 0:
        raise ValueError(
            "Some recordings have missing filename_lr values."
        )

    print("[OK] filename_lr is populated")

    print("\n" + "=" * 70)
    print("PTB-XL METADATA VALIDATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_ptbxl_metadata()