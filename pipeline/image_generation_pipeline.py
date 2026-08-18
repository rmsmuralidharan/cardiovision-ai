import sys

from components.image_dataset_builder import (
    ECGImageDatasetBuilder,
)
from components.image_generation_preflight import (
    ImageGenerationPreflight,
)
from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


class ImageGenerationPipeline:
    """
    Complete ECG image generation pipeline.

    Flow:
        Preflight validation
            ↓
        ECG image generation
            ↓
        Manifest creation
    """

    def __init__(self):

        self.preflight = (
            ImageGenerationPreflight()
        )

        self.builder = (
            ECGImageDatasetBuilder()
        )

    def run(self):

        try:

            logger.info(
                "Starting CardioVision AI "
                "image generation pipeline."
            )

            # ---------------------------------------------
            # Step 1: Preflight
            # ---------------------------------------------

            logger.info(
                "Running image generation preflight."
            )

            preflight_summary = (
                self.preflight.run()
            )

            logger.info(
                "Image generation preflight passed."
            )

            # ---------------------------------------------
            # Step 2: Generate images
            # ---------------------------------------------

            logger.info(
                "Starting ECG image generation."
            )

            (
                train_manifest,
                validation_manifest,
                test_manifest,
            ) = self.builder.build_all_splits()

            # ---------------------------------------------
            # Step 3: Final summary
            # ---------------------------------------------

            total_images = (
                len(train_manifest)
                + len(validation_manifest)
                + len(test_manifest)
            )

            logger.info(
                "Image generation completed."
            )

            logger.info(
                "Total images generated/available: %s",
                total_images,
            )

            return {
                "preflight_summary": preflight_summary,
                "train_manifest": train_manifest,
                "validation_manifest": validation_manifest,
                "test_manifest": test_manifest,
            }

        except Exception as error:

            logger.exception(
                "Image generation pipeline failed."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error