from pathlib import Path

from components.image_label_validator import (
    ImageLabelValidator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_image_label_validator():

    manifest_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ecg_images"
        / "train_sample_manifest.csv"
    )

    validator = ImageLabelValidator(
        manifest_file
    )

    df = validator.validate()

    print("=" * 70)
    print("IMAGE-LABEL INTEGRITY VALIDATION")
    print("=" * 70)

    print("\nRecords validated:")
    print(len(df))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(
        df["target_mi"]
        .value_counts()
        .sort_index()
    )

    print("\nImage paths:")
    print(
        df["image_path"].tolist()
    )

    print("\n" + "=" * 70)
    print("IMAGE-LABEL INTEGRITY: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_image_label_validator()