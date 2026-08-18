import sys
from pathlib import Path

import pandas as pd

from components.class_weight import (
    ClassWeightCalculator,
)
from components.cnn_dataset import (
    CNNDatasetBuilder,
)
from components.training_callbacks import (
    TrainingCallbacks,
)
from exception.exception import (
    CardioVisionAIException,
)
from models.cnn_model import (
    CardioVisionCNN,
)
from project_logging.logger import (
    get_logger,
)


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecg_images"
)


class CNNTrainer:
    """
    Handles training of the CardioVision
    baseline CNN.
    """

    def __init__(
        self,
        epochs=10,
        batch_size=32,
        learning_rate=0.0001,
        mi_weight = None,
        experiment_name = 'baseline'
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mi_weight = mi_weight
        self.experiment_name = experiment_name

        self.dataset_builder = (
            CNNDatasetBuilder(
                batch_size=batch_size
            )
        )

        self.class_weight_calculator = (
            ClassWeightCalculator()
        )

        self.callback_builder = (
            TrainingCallbacks(experiment_name = experiment_name)
        )

        self.model_builder = (
            CardioVisionCNN(
                learning_rate=learning_rate
            )
        )

    def load_manifests(self):

        train_manifest = pd.read_csv(
            IMAGE_DIR
            / "train_manifest.csv"
        )

        validation_manifest = pd.read_csv(
            IMAGE_DIR
            / "validation_manifest.csv"
        )

        logger.info(
            "Train manifest records: %s",
            len(train_manifest),
        )

        logger.info(
            "Validation manifest records: %s",
            len(validation_manifest),
        )

        return (
            train_manifest,
            validation_manifest,
        )

    def prepare_datasets(
        self,
        train_manifest,
        validation_manifest,
    ):

        logger.info(
            "Building training dataset."
        )

        train_dataset = (
            self.dataset_builder.build_dataset(
                train_manifest,
                "train",
            )
        )

        logger.info(
            "Building validation dataset."
        )

        validation_dataset = (
            self.dataset_builder.build_dataset(
                validation_manifest,
                "validation",
            )
        )

        return (
            train_dataset,
            validation_dataset,
        )

    def calculate_class_weights(
        self,
        train_manifest,
    ):

        labels = train_manifest[
            "target_mi"
        ].values

        class_weights = (
            self.class_weight_calculator.calculate(
                labels,
                mi_weight = self.mi_weight
            )
        )

        return class_weights

    def train(self):

        try:

            logger.info(
                "Starting CardioVision baseline "
                "CNN training."
            )

            # -----------------------------------------
            # Load manifests
            # -----------------------------------------

            (
                train_manifest,
                validation_manifest,
            ) = self.load_manifests()

            # -----------------------------------------
            # Prepare datasets
            # -----------------------------------------

            (
                train_dataset,
                validation_dataset,
            ) = self.prepare_datasets(
                train_manifest,
                validation_manifest,
            )

            # -----------------------------------------
            # Calculate class weights
            # -----------------------------------------

            class_weights = (
                self.calculate_class_weights(
                    train_manifest
                )
            )

            logger.info(
                "Training class weights: %s",
                class_weights,
            )

            # -----------------------------------------
            # Build model
            # -----------------------------------------

            model = (
                self.model_builder.build_model()
            )

            # -----------------------------------------
            # Create callbacks
            # -----------------------------------------

            callbacks = (
                self.callback_builder.create_callbacks()
            )

            # -----------------------------------------
            # Train
            # -----------------------------------------

            logger.info(
                "Starting model.fit()."
            )

            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=self.epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
                shuffle=False
            )

            logger.info(
                "Baseline CNN training completed."
            )

            return (
                model,
                history,
            )

        except Exception as error:

            logger.exception(
                "Baseline CNN training failed."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error