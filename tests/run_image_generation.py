from pipeline.image_generation_pipeline import (
    ImageGenerationPipeline,
)


def run_image_generation():

    print("=" * 70)
    print("CARDIOVISION AI - ECG IMAGE GENERATION")
    print("=" * 70)

    pipeline = ImageGenerationPipeline()

    result = pipeline.run()

    print("\n" + "=" * 70)
    print("IMAGE GENERATION COMPLETED")
    print("=" * 70)

    print(
        "\nTrain images:",
        len(result["train_manifest"])
    )

    print(
        "Validation images:",
        len(result["validation_manifest"])
    )

    print(
        "Test images:",
        len(result["test_manifest"])
    )

    print(
        "Total images:",
        (
            len(result["train_manifest"])
            + len(result["validation_manifest"])
            + len(result["test_manifest"])
        )
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_image_generation()