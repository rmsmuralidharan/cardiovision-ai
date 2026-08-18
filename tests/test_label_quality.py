from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "labels"
    / "ptbxl_mi_labels.csv"
)


def test_label_quality():

    print("=" * 70)
    print("CardioVision AI - MI Label Quality Analysis")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Check file
    # ---------------------------------------------------------

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Label file not found: {LABEL_FILE}"
        )

    print("\n[OK] Labeled metadata file exists.")

    # ---------------------------------------------------------
    # 2. Load data
    # ---------------------------------------------------------

    df = pd.read_csv(LABEL_FILE)

    print("\nDataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 3. Required columns
    # ---------------------------------------------------------

    required_columns = [
        "ecg_id",
        "patient_id",
        "scp_codes",
        "target_mi",
        "mi_codes",
        "mi_max_likelihood",
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
    # 4. Target values
    # ---------------------------------------------------------

    print("\nTarget values:")

    print(
        df["target_mi"]
        .value_counts(dropna=False)
        .sort_index()
    )

    invalid_targets = (
        ~df["target_mi"].isin([0, 1])
    ).sum()

    if invalid_targets > 0:
        raise ValueError(
            f"Found {invalid_targets} invalid target values."
        )

    print("[OK] Target contains only 0 and 1.")

    # ---------------------------------------------------------
    # 5. Missing target values
    # ---------------------------------------------------------

    missing_targets = df["target_mi"].isna().sum()

    print("\nMissing target values:")
    print(missing_targets)

    if missing_targets > 0:
        raise ValueError(
            "Missing target_mi values detected."
        )

    print("[OK] No missing target values.")

    # ---------------------------------------------------------
    # 6. Class counts
    # ---------------------------------------------------------

    non_mi_count = (
        df["target_mi"] == 0
    ).sum()

    mi_count = (
        df["target_mi"] == 1
    ).sum()

    total_count = len(df)

    print("\nClass distribution:")
    print(f"Total ECGs : {total_count}")
    print(f"Non-MI     : {non_mi_count}")
    print(f"MI         : {mi_count}")

    # ---------------------------------------------------------
    # 7. Class percentages
    # ---------------------------------------------------------

    non_mi_percentage = (
        non_mi_count / total_count
    ) * 100

    mi_percentage = (
        mi_count / total_count
    ) * 100

    print("\nClass percentages:")
    print(
        f"Non-MI : {non_mi_percentage:.2f}%"
    )
    print(
        f"MI     : {mi_percentage:.2f}%"
    )

    # ---------------------------------------------------------
    # 8. Imbalance ratio
    # ---------------------------------------------------------

    majority_count = max(
        non_mi_count,
        mi_count,
    )

    minority_count = min(
        non_mi_count,
        mi_count,
    )

    imbalance_ratio = (
        majority_count / minority_count
        if minority_count > 0
        else float("inf")
    )

    print("\nClass imbalance:")
    print(
        f"Majority / Minority ratio: "
        f"{imbalance_ratio:.2f}:1"
    )

    # ---------------------------------------------------------
    # 9. ECG ID uniqueness
    # ---------------------------------------------------------

    duplicate_ecg_ids = (
        df["ecg_id"].duplicated().sum()
    )

    print("\nDuplicate ECG IDs:")
    print(duplicate_ecg_ids)

    if duplicate_ecg_ids > 0:
        raise ValueError(
            "Duplicate ECG IDs detected."
        )

    print("[OK] ECG IDs are unique.")

    # ---------------------------------------------------------
    # 10. Patient analysis
    # ---------------------------------------------------------

    unique_patients = (
        df["patient_id"].nunique()
    )

    print("\nPatient analysis:")
    print(
        f"Unique patients: {unique_patients}"
    )

    recordings_per_patient = (
        df.groupby("patient_id")
        .size()
    )

    multiple_recording_patients = (
        recordings_per_patient > 1
    ).sum()

    print(
        "Patients with multiple ECGs: "
        f"{multiple_recording_patients}"
    )

    # ---------------------------------------------------------
    # 11. MI patients
    # ---------------------------------------------------------

    mi_patients = (
        df.loc[
            df["target_mi"] == 1,
            "patient_id"
        ]
        .nunique()
    )

    non_mi_patients = (
        df.loc[
            df["target_mi"] == 0,
            "patient_id"
        ]
        .nunique()
    )

    print("\nPatient-level class distribution:")
    print(f"MI patients     : {mi_patients}")
    print(f"Non-MI patients : {non_mi_patients}")

    # ---------------------------------------------------------
    # 12. MI likelihood
    # ---------------------------------------------------------

    mi_likelihood = (
        df.loc[
            df["target_mi"] == 1,
            "mi_max_likelihood"
        ]
    )

    print("\nMI likelihood statistics:")

    if len(mi_likelihood) > 0:
        print(
            mi_likelihood.describe()
        )

    else:
        print(
            "No MI-positive records found."
        )

    # ---------------------------------------------------------
    # 13. MI codes
    # ---------------------------------------------------------

    mi_code_counts = (
        df.loc[
            df["target_mi"] == 1,
            "mi_codes"
        ]
        .value_counts()
        .head(20)
    )

    print("\nMost common MI-code combinations:")

    print(mi_code_counts)

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL QUALITY ANALYSIS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_label_quality()