import sys

import numpy as np
from sklearn.utils.class_weight import (
    compute_class_weight,
)

from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


class ClassWeightCalculator:
    """
    Calculates class weights for binary ECG classification.

    By default, sklearn's balanced class weights are returned.

    An optional MI weight can be supplied for controlled
    class-weight experiments.
    """

    def calculate(
        self,
        labels,
        mi_weight=None,
    ):

        try:

            labels = np.asarray(
                labels
            )

            classes = np.unique(
                labels
            )

            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=labels,
            )

            class_weights = {
                int(class_value): float(weight)
                for class_value, weight
                in zip(classes, weights)
            }

            # -----------------------------------------
            # Optional MI weight override
            # -----------------------------------------

            if mi_weight is not None:

                if mi_weight <= 0:

                    raise ValueError(
                        "mi_weight must be greater than 0."
                    )

                if 1 not in class_weights:

                    raise ValueError(
                        "MI class (1) is not present "
                        "in the provided labels."
                    )

                class_weights[1] = float(
                    mi_weight
                )

                logger.info(
                    "MI class weight overridden: %s",
                    mi_weight,
                )

            logger.info(
                "Final class weights: %s",
                class_weights,
            )

            return class_weights

        except Exception as error:

            logger.exception(
                "Failed to calculate class weights."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error