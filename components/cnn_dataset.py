import sys
from pathlib import Path

import tensorflow as tf

from components.data_augmentation import (
    ECGDataAugmentation,
)
from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


class CNNDatasetBuilder:
    """
    Builds TensorFlow datasets for ECG image classification.

    Training:
        resize + normalize + augmentation

    Validation:
        resize + normalize

    Test:
        resize + normalize
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    ):

        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

        self.augmentation = (
            ECGDataAugmentation()
        )

    def load_image(
        self,
        image_path,
        label,
    ):

        image = tf.io.read_file(
            image_path
        )

        image = tf.image.decode_png(
            image,
            channels=3,
        )

        image = tf.image.resize(
            image,
            self.image_size,
        )

        image = tf.cast(
            image,
            tf.float32,
        ) / 255.0

        return image, label

    def augment_image(
        self,
        image,
        label,
    ):

        image = self.augmentation(
            image,
            training=True,
        )

        return image, label

    def build_dataset(
        self,
        manifest_df,
        split_name,
    ):

        try:

            if split_name not in [
                "train",
                "validation",
                "test",
            ]:

                raise ValueError(
                    "split_name must be "
                    "'train', 'validation', "
                    "or 'test'."
                )

            image_paths = [
                str(
                    PROJECT_ROOT
                    / Path(image_path)
                )
                for image_path
                in manifest_df[
                    "image_path"
                ]
            ]

            labels = (
                manifest_df[
                    "target_mi"
                ]
                .astype("float32")
                .values
            )

            dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        image_paths,
                        labels,
                    )
                )
            )

            dataset = dataset.map(
                self.load_image,
                num_parallel_calls=(
                    tf.data.AUTOTUNE
                ),
            )

            # -----------------------------------------
            # Training only
            # -----------------------------------------

            if split_name == "train":

                dataset = dataset.shuffle(
                    buffer_size=2048,
                    seed=self.seed,
                    reshuffle_each_iteration=True,
                )

                dataset = dataset.map(
                    self.augment_image,
                    num_parallel_calls=(
                        tf.data.AUTOTUNE
                    ),
                )

            # -----------------------------------------
            # Batch
            # -----------------------------------------

            dataset = dataset.batch(
                self.batch_size
            )

            # -----------------------------------------
            # Prefetch
            # -----------------------------------------

            dataset = dataset.prefetch(
                tf.data.AUTOTUNE
            )

            logger.info(
                "%s TensorFlow dataset created: "
                "records=%s, batches=%s",
                split_name,
                len(manifest_df),
                tf.data.experimental.cardinality(
                    dataset
                ).numpy(),
            )

            return dataset

        except Exception as error:

            logger.exception(
                "Failed to create %s CNN dataset.",
                split_name,
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error