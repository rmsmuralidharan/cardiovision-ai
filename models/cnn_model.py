import sys

import tensorflow as tf

from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


class CardioVisionCNN:
    """
    Baseline CNN model for ECG image-based
    myocardial infarction (MI) classification.
    """

    def __init__(
        self,
        input_shape=(224, 224, 3),
        learning_rate=0.0001,
    ):

        self.input_shape = input_shape
        self.learning_rate = learning_rate

    def build_model(self):

        try:

            inputs = tf.keras.Input(
                shape=self.input_shape,
                name="ecg_image",
            )

            # -----------------------------------------
            # Convolution Block 1
            # -----------------------------------------

            x = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False,
                name="conv1",
            )(inputs)

            x = tf.keras.layers.BatchNormalization(
                name="bn1"
            )(x)

            x = tf.keras.layers.ReLU(
                name="relu1"
            )(x)

            x = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool1",
            )(x)

            # -----------------------------------------
            # Convolution Block 2
            # -----------------------------------------

            x = tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False,
                name="conv2",
            )(x)

            x = tf.keras.layers.BatchNormalization(
                name="bn2"
            )(x)

            x = tf.keras.layers.ReLU(
                name="relu2"
            )(x)

            x = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool2",
            )(x)

            # -----------------------------------------
            # Convolution Block 3
            # -----------------------------------------

            x = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False,
                name="conv3",
            )(x)

            x = tf.keras.layers.BatchNormalization(
                name="bn3"
            )(x)

            x = tf.keras.layers.ReLU(
                name="relu3"
            )(x)

            x = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool3",
            )(x)

            # -----------------------------------------
            # Feature aggregation
            # -----------------------------------------

            x = tf.keras.layers.GlobalAveragePooling2D(
                name="global_average_pooling"
            )(x)

            # -----------------------------------------
            # Classification head
            # -----------------------------------------

            x = tf.keras.layers.Dense(
                128,
                activation="relu",
                name="dense1",
            )(x)

            x = tf.keras.layers.Dropout(
                0.4,
                name="dropout",
            )(x)

            # -----------------------------------------
            # Binary MI output
            # -----------------------------------------

            outputs = tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                name="mi_probability",
            )(x)

            model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs,
                name="CardioVision_Baseline_CNN",
            )

            # -----------------------------------------
            # Compile
            # -----------------------------------------

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate
                ),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(
                        name="accuracy"
                    ),
                    tf.keras.metrics.Precision(
                        name="precision"
                    ),
                    tf.keras.metrics.Recall(
                        name="recall"
                    ),
                    tf.keras.metrics.AUC(
                        name="auc"
                    ),
                ],
            )

            logger.info(
                "CardioVision baseline CNN "
                "successfully built."
            )

            return model

        except Exception as error:

            logger.exception(
                "Failed to build CardioVision CNN."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error