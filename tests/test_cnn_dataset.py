from pathlib import Path

import pandas as pd

from components.cnn_dataset import (
    CNNDatasetBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cnn_dataset():

    image_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ecg_images"
    )

    train_df = pd.read_csv(
        image_dir
        / "train_manifest.csv"
    )

    validation_df = pd.read_csv(
        image_dir
        / "validation_manifest.csv"
    )

    test_df = pd.read_csv(
        image_dir
        / "test_manifest.csv"
    )

    # Small samples for testing.
    train_sample = train_df.head(32)

    validation_sample = (
        validation_df.head(32)
    )

    test_sample = test_df.head(32)

    builder = CNNDatasetBuilder()

    train_dataset = (
        builder.build_dataset(
            train_sample,
            "train",
        )
    )

    validation_dataset = (
        builder.build_dataset(
            validation_sample,
            "validation",
        )
    )

    test_dataset = (
        builder.build_dataset(
            test_sample,
            "test",
        )
    )

    train_images, train_labels = next(
        iter(train_dataset)
    )

    validation_images, validation_labels = next(
        iter(validation_dataset)
    )

    test_images, test_labels = next(
        iter(test_dataset)
    )

    print("=" * 70)
    print("COMPLETE CNN DATASET PIPELINE TEST")
    print("=" * 70)

    print("\nTrain:")
    print("Images:", train_images.shape)
    print("Labels:", train_labels.shape)

    print("\nValidation:")
    print("Images:", validation_images.shape)
    print("Labels:", validation_labels.shape)

    print("\nTest:")
    print("Images:", test_images.shape)
    print("Labels:", test_labels.shape)

    # ---------------------------------------------
    # Shape validation
    # ---------------------------------------------

    expected_image_shape = (
        32,
        224,
        224,
        3,
    )

    expected_label_shape = (
        32,
    )

    assert (
        train_images.shape
        == expected_image_shape
    )

    assert (
        validation_images.shape
        == expected_image_shape
    )

    assert (
        test_images.shape
        == expected_image_shape
    )

    assert (
        train_labels.shape
        == expected_label_shape
    )

    assert (
        validation_labels.shape
        == expected_label_shape
    )

    assert (
        test_labels.shape
        == expected_label_shape
    )

    # ---------------------------------------------
    # Data type validation
    # ---------------------------------------------

    assert str(
        train_images.dtype
    ) == "<dtype: 'float32'>"

    # ---------------------------------------------
    # Normalization validation
    # ---------------------------------------------

    assert (
        float(
            train_images.numpy().min()
        ) >= 0.0
    )

    assert (
        float(
            train_images.numpy().max()
        ) <= 1.0
    )

    assert (
        float(
            validation_images.numpy().min()
        ) >= 0.0
    )

    assert (
        float(
            validation_images.numpy().max()
        ) <= 1.0
    )

    assert (
        float(
            test_images.numpy().min()
        ) >= 0.0
    )

    assert (
        float(
            test_images.numpy().max()
        ) <= 1.0
    )

    print("\n" + "=" * 70)
    print(
        "COMPLETE CNN DATASET PIPELINE: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    test_cnn_dataset()