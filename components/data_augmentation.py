import tensorflow as tf

from project_logging.logger import get_logger


logger = get_logger(__name__)


class ECGDataAugmentation:
    """
    Conservative augmentation pipeline for ECG images.

    Only intended for the training dataset.
    """

    def __init__(self):

        self.augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(
                    factor=5 / 360
                ),

                tf.keras.layers.RandomZoom(
                    height_factor=0.05,
                    width_factor=0.05,
                ),

                tf.keras.layers.RandomTranslation(
                    height_factor=0.02,
                    width_factor=0.02,
                ),
            ],
            name="ecg_augmentation",
        )

        logger.info(
            "ECG training augmentation pipeline initialized."
        )

    def __call__(
        self,
        images,
        training=True,
    ):

        return self.augmentation(
            images,
            training=training,
        )