from pipeline.image_generation_pipeline import (
    ImageGenerationPipeline,
)


def test_image_generation_pipeline_initialization():

    pipeline = (
        ImageGenerationPipeline()
    )

    assert pipeline.preflight is not None

    assert pipeline.builder is not None

    print("=" * 70)
    print(
        "IMAGE GENERATION PIPELINE "
        "INITIALIZATION: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    test_image_generation_pipeline_initialization()