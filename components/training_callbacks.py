from pathlib import Path

import tensorflow as tf

from project_logging.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

LOG_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "training_logs"
)


class TrainingCallbacks:
    """
    Creates callbacks used during CNN training.
    """

    def __init__(
            self,
            experiment_name = 'baseline'
    ):

        self.experiment_name = (
            experiment_name
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_callbacks(self):

        checkpoint_path = (
            MODEL_DIR
            / f"{self.experiment_name}_best.keras"
        )

        tensorboard_dir = (
            LOG_DIR
            / self.experiment_name
        )

        callbacks = [

            # -----------------------------------------
            # Early stopping
            # -----------------------------------------

            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=7,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),

            # -----------------------------------------
            # Best model checkpoint
            # -----------------------------------------

            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(
                    checkpoint_path
                ),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),

            # -----------------------------------------
            # Learning rate reduction
            # -----------------------------------------

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode="max",
                verbose=1,
            ),

            # -----------------------------------------
            # TensorBoard
            # -----------------------------------------

            tf.keras.callbacks.TensorBoard(
                log_dir=str(
                    tensorboard_dir
                )
            ),
        ]

        logger.info(
            "Training callbacks created."
        )

        logger.info(
            "Best model path: %s",
            checkpoint_path,
        )

        logger.info(
            "TensorBoard log directory: %s",
            tensorboard_dir,
        )

        return callbacks