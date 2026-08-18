import pandas as pd
from pathlib import Path

from components.class_weight import (
    ClassWeightCalculator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_class_weight():

    manifest_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ecg_images"
        / "train_manifest.csv"
    )

    train_df = pd.read_csv(
        manifest_file
    )

    labels = train_df[
        "target_mi"
    ].values

    calculator = (
        ClassWeightCalculator()
    )

    # -----------------------------------------
    # Calculate baseline class weights
    # -----------------------------------------

    class_weights = (
        calculator.calculate(
            labels
        )
    )

    print("=" * 70)
    print("CLASS WEIGHT TEST")
    print("=" * 70)

    print(
        "\nTraining samples:"
    )

    print(
        len(labels)
    )

    print(
        "\nClass distribution:"
    )

    print(
        train_df[
            "target_mi"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nBaseline class weights:"
    )

    print(
        class_weights
    )

    # -----------------------------------------
    # Validate baseline weights
    # -----------------------------------------

    assert 0 in class_weights
    assert 1 in class_weights

    assert (
        class_weights[1]
        > class_weights[0]
    )

    # -----------------------------------------
    # Test MI weight override
    # -----------------------------------------

    experiment_weights = [
        1.484,
        1.187,
        0.989,
    ]

    print(
        "\nMI WEIGHT EXPERIMENTS:"
    )

    for mi_weight in experiment_weights:

        experiment_class_weights = (
            calculator.calculate(
                labels,
                mi_weight=mi_weight,
            )
        )

        print(
            f"\nMI weight: {mi_weight}"
        )

        print(
            "Class weights:"
        )

        print(
            experiment_class_weights
        )

        # -------------------------------------
        # Non-MI weight must remain unchanged
        # -------------------------------------

        assert (
            experiment_class_weights[0]
            == class_weights[0]
        )

        # -------------------------------------
        # MI weight must match requested value
        # -------------------------------------

        assert (
            experiment_class_weights[1]
            == mi_weight
        )

    print(
        "\nMI weight override tests: PASSED"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASS WEIGHT TEST: PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    test_class_weight()