from components.image_dataset_builder import (
    ECGImageDatasetBuilder,
)


def test_full_image_dataset():

    print("=" * 70)
    print("FULL ECG IMAGE DATASET GENERATION")
    print("=" * 70)

    builder = ECGImageDatasetBuilder()

    (
        train_manifest,
        validation_manifest,
        test_manifest,
    ) = builder.build_all_splits()

    print("\nTrain images:")
    print(len(train_manifest))

    print("\nValidation images:")
    print(len(validation_manifest))

    print("\nTest images:")
    print(len(test_manifest))

    print("\nTotal images:")
    print(
        len(train_manifest)
        + len(validation_manifest)
        + len(test_manifest)
    )

    print("\nTrain target distribution:")
    print(
        train_manifest[
            "target_mi"
        ].value_counts()
        .sort_index()
    )

    print("\nValidation target distribution:")
    print(
        validation_manifest[
            "target_mi"
        ].value_counts()
        .sort_index()
    )

    print("\nTest target distribution:")
    print(
        test_manifest[
            "target_mi"
        ].value_counts()
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("FULL ECG IMAGE DATASET GENERATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_full_image_dataset()