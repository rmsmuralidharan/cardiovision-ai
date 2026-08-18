from components.cnn_evaluator import (
    CNNEvaluator,
)


def test_cnn_evaluator():

    evaluator = CNNEvaluator(
        model_path=(
            "artifacts/models/"
            "cardiovision_baseline_best.keras"
        ),
        validation_manifest_path=(
            "data/processed/ecg_images/"
            "validation_manifest.csv"
        ),
        batch_size=32,
    )

    evaluator.evaluate()


if __name__ == "__main__":
    test_cnn_evaluator()