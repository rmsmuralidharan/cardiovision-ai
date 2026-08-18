import sys
from pathlib import Path

import pandas as pd

from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImageLabelValidator:
    """
    Validates the relationship between ECG images and labels.
    """

    def __init__(
        self,
        manifest_file: Path,
    ):

        self.manifest_file = Path(
            manifest_file
        )

    def load_manifest(self) -> pd.DataFrame:

        try:

            if not self.manifest_file.exists():
                raise FileNotFoundError(
                    f"Manifest not found: "
                    f"{self.manifest_file}"
                )

            df = pd.read_csv(
                self.manifest_file
            )

            logger.info(
                "Loaded image manifest: %s",
                self.manifest_file,
            )

            return df

        except Exception as error:

            logger.exception(
                "Failed to load image manifest."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    def validate_required_columns(
        self,
        df: pd.DataFrame,
    ):

        required_columns = [
            "ecg_id",
            "patient_id",
            "target_mi",
            "image_path",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise CardioVisionAIException(
                "Missing manifest columns: "
                f"{missing_columns}"
            )

    def validate_missing_values(
        self,
        df: pd.DataFrame,
    ):

        required_columns = [
            "ecg_id",
            "patient_id",
            "target_mi",
            "image_path",
        ]

        missing_values = (
            df[required_columns]
            .isnull()
            .sum()
        )

        if missing_values.any():

            raise CardioVisionAIException(
                "Missing values detected:\n"
                f"{missing_values}"
            )

    def validate_target(
        self,
        df: pd.DataFrame,
    ):

        invalid_targets = (
            ~df["target_mi"].isin([0, 1])
        )

        if invalid_targets.any():

            raise CardioVisionAIException(
                "Invalid target_mi values detected."
            )

    def validate_unique_ecg_ids(
        self,
        df: pd.DataFrame,
    ):

        if df["ecg_id"].duplicated().any():

            duplicates = (
                df.loc[
                    df["ecg_id"].duplicated(),
                    "ecg_id",
                ]
                .tolist()
            )

            raise CardioVisionAIException(
                "Duplicate ECG IDs found: "
                f"{duplicates}"
            )

    def validate_unique_images(
        self,
        df: pd.DataFrame,
    ):

        if df["image_path"].duplicated().any():

            raise CardioVisionAIException(
                "Duplicate image paths found."
            )

    def validate_image_files(
        self,
        df: pd.DataFrame,
    ):

        missing_images = []

        for image_path in df["image_path"]:

            full_path = (
                PROJECT_ROOT
                / Path(image_path)
            )

            if not full_path.exists():

                missing_images.append(
                    str(full_path)
                )

        if missing_images:

            raise CardioVisionAIException(
                "Missing image files:\n"
                + "\n".join(
                    missing_images
                )
            )

    def validate_image_ecg_mapping(
        self,
        df: pd.DataFrame,
    ):

        for _, row in df.iterrows():

            ecg_id = int(
                row["ecg_id"]
            )

            image_path = Path(
                row["image_path"]
            )

            expected_name = (
                f"ecg_{ecg_id}.png"
            )

            if image_path.name != expected_name:

                raise CardioVisionAIException(
                    f"Image/ECG mismatch: "
                    f"ECG {ecg_id} points to "
                    f"{image_path.name}"
                )

    def validate(
        self,
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting image-label integrity validation."
            )

            df = self.load_manifest()

            self.validate_required_columns(
                df
            )

            self.validate_missing_values(
                df
            )

            self.validate_target(
                df
            )

            self.validate_unique_ecg_ids(
                df
            )

            self.validate_unique_images(
                df
            )

            self.validate_image_files(
                df
            )

            self.validate_image_ecg_mapping(
                df
            )

            logger.info(
                "Image-label integrity validation passed."
            )

            return df

        except CardioVisionAIException:

            raise

        except Exception as error:

            logger.exception(
                "Image-label validation failed."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error