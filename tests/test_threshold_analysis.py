import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


PROJECT_ROOT = r"D:\CardioVision-AI"

MODEL_PATH = (
    PROJECT_ROOT
    + r"\artifacts\models\cardiovision_baseline_best.keras"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    + r"\data\processed\ecg_images\validation_manifest.csv"
)


def load_validation_predictions():

    print("=" * 70)
    print("LOADING BASELINE CNN")
    print("=" * 70)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    validation_df = pd.read_csv(
        MANIFEST_PATH
    )

    from components.cnn_dataset import (
        CNNDatasetBuilder,
    )

    dataset_builder = CNNDatasetBuilder(
        batch_size=32
    )

    validation_dataset = (
        dataset_builder.build_dataset(
            manifest_df=validation_df,
            split_name="validation",
        )
    )

    print(
        f"\nValidation records: "
        f"{len(validation_df)}"
    )

    print(
        "\nGenerating predictions..."
    )

    probabilities = model.predict(
        validation_dataset,
        verbose=1,
    ).ravel()

    y_true = (
        validation_df[
            "target_mi"
        ]
        .astype(int)
        .to_numpy()
    )

    return y_true, probabilities


def analyze_thresholds(
    y_true,
    probabilities,
):

    thresholds = np.arange(
        0.20,
        0.81,
        0.05,
    )

    results = []

    for threshold in thresholds:

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "accuracy": round(
                    accuracy,
                    4,
                ),
                "precision": round(
                    precision,
                    4,
                ),
                "recall": round(
                    recall,
                    4,
                ),
                "f1": round(
                    f1,
                    4,
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    return results_df


def test_threshold_analysis():

    y_true, probabilities = (
        load_validation_predictions()
    )

    results_df = analyze_thresholds(
        y_true,
        probabilities,
    )

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    # -----------------------------------------
    # Best threshold by F1
    # -----------------------------------------

    best_f1_row = (
        results_df.loc[
            results_df["f1"].idxmax()
        ]
    )

    # -----------------------------------------
    # Best threshold by accuracy
    # -----------------------------------------

    best_accuracy_row = (
        results_df.loc[
            results_df["accuracy"].idxmax()
        ]
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLD BY F1")
    print("=" * 70)

    print(
        f"Threshold : "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_f1_row['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_f1_row['f1']:.4f}"
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLD BY ACCURACY")
    print("=" * 70)

    print(
        f"Threshold : "
        f"{best_accuracy_row['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_accuracy_row['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_accuracy_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_accuracy_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_accuracy_row['f1']:.4f}"
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "THRESHOLD ANALYSIS: COMPLETED"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    test_threshold_analysis()