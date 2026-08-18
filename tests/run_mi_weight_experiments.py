import numpy as np
import pandas as pd
from pathlib import Path

from components.cnn_trainer import CNNTrainer
from project_logging.logger import get_logger


logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
)

RESULT_FILE = (
    ARTIFACT_DIR
    / "mi_weight_experiments.csv"
)


EXPERIMENTS = {
    "baseline": None,
    "exp1_0.75x": 1.484,
    "exp2_0.60x": 1.187,
    "exp3_0.50x": 0.989,
}


def run_experiments():

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for experiment_name, mi_weight in EXPERIMENTS.items():

        print()
        print("=" * 70)
        print(
            f"STARTING EXPERIMENT: "
            f"{experiment_name}"
        )
        print(
            f"MI weight: {mi_weight}"
        )
        print("=" * 70)

        trainer = CNNTrainer(
            epochs=15,
            batch_size=32,
            learning_rate=0.0001,
            mi_weight=mi_weight,
            experiment_name=experiment_name,
        )

        model, history = trainer.train()

        val_auc = np.asarray(
            history.history["val_auc"]
        )

        best_epoch = int(
            np.argmax(val_auc)
        )

        results.append(
            {
                "experiment": experiment_name,
                "mi_weight": (
                    "baseline"
                    if mi_weight is None
                    else mi_weight
                ),
                "best_epoch": best_epoch + 1,
                "val_accuracy": history.history[
                    "val_accuracy"
                ][best_epoch],
                "val_auc": history.history[
                    "val_auc"
                ][best_epoch],
                "val_precision": history.history[
                    "val_precision"
                ][best_epoch],
                "val_recall": history.history[
                    "val_recall"
                ][best_epoch],
                "val_loss": history.history[
                    "val_loss"
                ][best_epoch],
                "checkpoint": (
                    f"artifacts/models/"
                    f"{experiment_name}_best.keras"
                ),
            }
        )

        print()
        print(
            f"COMPLETED: {experiment_name}"
        )
        print(
            f"Best epoch: {best_epoch + 1}"
        )
        print(
            f"Val AUC: "
            f"{val_auc[best_epoch]:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("MI WEIGHT EXPERIMENTS COMPLETED")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: {RESULT_FILE}"
    )


if __name__ == "__main__":
    run_experiments()