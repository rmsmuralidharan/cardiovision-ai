import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from components.cnn_dataset import (
    CNNDatasetBuilder,
)
from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CNNEvaluator:
    """
    Evaluates the trained CardioVision CNN
    using the validation dataset.
    """

    def __init__(
        self,
        model_path,
        validation_manifest_path,
        batch_size=32,
    ):

        self.model_path = (
            PROJECT_ROOT / model_path
        )

        self.validation_manifest_path = (
            PROJECT_ROOT
            / validation_manifest_path
        )

        self.batch_size = batch_size

    def evaluate(self):

        try:

            # -----------------------------------------
            # 1. Load trained model
            # -----------------------------------------

            logger.info(
                "Loading trained model: %s",
                self.model_path,
            )

            model = tf.keras.models.load_model(
                self.model_path
            )

            # -----------------------------------------
            # 2. Load validation manifest
            # -----------------------------------------

            logger.info(
                "Loading validation manifest: %s",
                self.validation_manifest_path,
            )

            validation_df = pd.read_csv(
                self.validation_manifest_path
            )

            logger.info(
                "Validation records: %s",
                len(validation_df),
            )

            # -----------------------------------------
            # 3. Build validation dataset
            # -----------------------------------------

            dataset_builder = (
                CNNDatasetBuilder(
                    batch_size=self.batch_size
                )
            )

            validation_dataset = (
                dataset_builder.build_dataset(
                    manifest_df=validation_df,
                    split_name="validation",
                )
            )

            # -----------------------------------------
            # 4. Generate predictions
            # -----------------------------------------

            logger.info(
                "Generating validation predictions."
            )

            probabilities = model.predict(
                validation_dataset,
                verbose=1,
            ).ravel()

            # -----------------------------------------
            # 5. True labels
            # -----------------------------------------

            y_true = (
                validation_df[
                    "target_mi"
                ]
                .astype(int)
                .to_numpy()
            )

            # -----------------------------------------
            # 6. Convert probability → class
            # -----------------------------------------

            threshold = 0.50

            y_pred = (
                probabilities >= threshold
            ).astype(int)

            # -----------------------------------------
            # 7. Calculate metrics
            # -----------------------------------------

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

            roc_auc = roc_auc_score(
                y_true,
                probabilities,
            )

            cm = confusion_matrix(
                y_true,
                y_pred,
            )

            # -----------------------------------------
            # 8. Display results
            # -----------------------------------------

            print()
            print("=" * 70)
            print(
                "CARDIOVISION BASELINE CNN "
                "VALIDATION EVALUATION"
            )
            print("=" * 70)

            print(
                f"\nValidation records: "
                f"{len(y_true)}"
            )

            print(
                f"Threshold: {threshold:.2f}"
            )

            print(
                f"\nAccuracy : {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall   : {recall:.4f}"
            )

            print(
                f"F1 Score : {f1:.4f}"
            )

            print(
                f"ROC-AUC  : {roc_auc:.4f}"
            )

            print(
                "\nConfusion Matrix:"
            )

            print(cm)

            print(
                "\nClassification Report:"
            )

            print(
                classification_report(
                    y_true,
                    y_pred,
                    target_names=[
                        "Non-MI",
                        "MI",
                    ],
                    zero_division=0,
                )
            )

            print("=" * 70)

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "confusion_matrix": cm,
                "probabilities": probabilities,
                "y_true": y_true,
                "y_pred": y_pred,
            }

        except Exception as error:

            logger.exception(
                "CNN validation evaluation failed."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error