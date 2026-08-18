from pathlib import Path

import pandas as pd

from components.image_dataset_builder import (
    ECGImageDatasetBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_small_image_batch():

    split_file = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "splits"
        / "train.csv"
    )

    df = pd.read_csv(split_file)

    # Take only 3 ECGs for the first batch test.
    sample_df = df.head(3)

    temporary_split_file = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "splits"
        / "train_sample.csv"
    )

    sample_df.to_csv(
        temporary_split_file,
        index=False,
    )

    builder = ECGImageDatasetBuilder()

    # Build manually using the sample.
    original_split_dir = builder.split_dir

    builder.split_dir = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "splits"
    )

    # Temporarily name the sample as train_sample.
    manifest_df = builder.build_split(
        "train_sample"
    )

    print("=" * 70)
    print("SMALL ECG IMAGE BATCH TEST")
    print("=" * 70)

    print("\nGenerated images:")
    print(len(manifest_df))

    print("\nManifest:")
    print(manifest_df)

    print("\nImage existence:")

    for image_path in manifest_df[
        "image_path"
    ]:

        full_path = (
            PROJECT_ROOT
            / image_path
        )

        print(
            full_path,
            "->",
            full_path.exists(),
        )

        if not full_path.exists():
            raise FileNotFoundError(
                f"Image was not created: {full_path}"
            )

    print("\n" + "=" * 70)
    print("SMALL IMAGE BATCH TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_small_image_batch()