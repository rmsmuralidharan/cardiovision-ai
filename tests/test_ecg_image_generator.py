from pathlib import Path

import pandas as pd

from components.ecg_image_generator import ECGImageGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PTBXL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptbxl"
)

METADATA_FILE = (
    PTBXL_DIR
    / "ptbxl_database.csv"
)


def test_ecg_image_generator():

    print("=" * 70)
    print("ECG IMAGE GENERATION TEST")
    print("=" * 70)

    # Load metadata
    df = pd.read_csv(METADATA_FILE)

    # Take only ONE ECG
    row = df.iloc[0]

    ecg_id = int(row["ecg_id"])

    relative_record_path = Path(
        row["filename_lr"]
    )

    record_path = (
        PTBXL_DIR
        / relative_record_path
    )

    print("\nECG ID:")
    print(ecg_id)

    print("\nRecord path:")
    print(record_path)

    generator = ECGImageGenerator()

    record, signal = generator.load_recording(
        record_path
    )

    generator.validate_recording(
        record,
        signal,
    )

    print("\nSignal shape:")
    print(signal.shape)

    print("\nSampling frequency:")
    print(record.fs)

    print("\nLead names:")
    print(record.sig_name)

    output_file = generator.generate_image(
        record=record,
        signal=signal,
        ecg_id=ecg_id,
    )

    print("\nGenerated image:")
    print(output_file)

    print("\n" + "=" * 70)
    print("ECG IMAGE GENERATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_ecg_image_generator()