import sys
from pathlib import Path

import pandas as pd

from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PTBXL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptbxl"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "splits"
)


class ImageGenerationPreflight:
    """
    Performs validation checks before full ECG image generation.
    """

    REQUIRED_SPLITS = [
        "train",
        "validation",
        "test",
    ]

    REQUIRED_COLUMNS = [
        "ecg_id",
        "patient_id",
        "target_mi",
        "filename_lr",
    ]

    def __init__(
        self,
        ptbxl_dir: Path = PTBXL_DIR,
        split_dir: Path = SPLIT_DIR,
    ):

        self.ptbxl_dir = Path(ptbxl_dir)
        self.split_dir = Path(split_dir)

    def load_split(
        self,
        split_name: str,
    ) -> pd.DataFrame:

        split_file = (
            self.split_dir
            / f"{split_name}.csv"
        )

        if not split_file.exists():

            raise CardioVisionAIException(
                f"Required split file does not exist: "
                f"{split_file}",
                sys.exc_info(),
            )

        df = pd.read_csv(
            split_file
        )

        logger.info(
            "Loaded %s split: %s records.",
            split_name,
            len(df),
        )

        return df

    def validate_columns(
        self,
        df: pd.DataFrame,
        split_name: str,
    ):

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:

            raise CardioVisionAIException(
                f"{split_name} split is missing "
                f"required columns: "
                f"{missing_columns}",
                sys.exc_info(),
            )

    def validate_missing_values(
        self,
        df: pd.DataFrame,
        split_name: str,
    ):

        missing_counts = (
            df[self.REQUIRED_COLUMNS]
            .isna()
            .sum()
        )

        missing_columns = (
            missing_counts[
                missing_counts > 0
            ]
        )

        if not missing_columns.empty:

            raise CardioVisionAIException(
                f"Missing values found in "
                f"{split_name} split:\n"
                f"{missing_columns}",
                sys.exc_info(),
            )

    def validate_targets(
        self,
        df: pd.DataFrame,
        split_name: str,
    ):

        invalid_targets = (
            ~df["target_mi"].isin([0, 1])
        )

        if invalid_targets.any():

            invalid_count = (
                invalid_targets.sum()
            )

            raise CardioVisionAIException(
                f"{split_name} contains "
                f"{invalid_count} invalid "
                f"target_mi values.",
                sys.exc_info(),
            )

    def validate_duplicate_ecg_ids(
        self,
        df: pd.DataFrame,
        split_name: str,
    ):

        duplicate_count = (
            df["ecg_id"]
            .duplicated()
            .sum()
        )

        if duplicate_count > 0:

            raise CardioVisionAIException(
                f"{split_name} contains "
                f"{duplicate_count} duplicate ECG IDs.",
                sys.exc_info(),
            )

    def validate_recording_files(
        self,
        df: pd.DataFrame,
        split_name: str,
    ):

        missing_header_files = []
        missing_signal_files = []

        for _, row in df.iterrows():

            relative_path = Path(
                str(row["filename_lr"])
            )

            record_path = (
                self.ptbxl_dir
                / relative_path
            )

            header_file = Path(
                f"{record_path}.hea"
            )

            signal_file = Path(
                f"{record_path}.dat"
            )

            if not header_file.exists():

                missing_header_files.append(
                    str(header_file)
                )

            if not signal_file.exists():

                missing_signal_files.append(
                    str(signal_file)
                )

        if missing_header_files:

            raise CardioVisionAIException(
                f"{split_name}: "
                f"{len(missing_header_files)} "
                f"missing .hea files.\n"
                f"Examples:\n"
                + "\n".join(
                    missing_header_files[:10]
                ),
                sys.exc_info(),
            )

        if missing_signal_files:

            raise CardioVisionAIException(
                f"{split_name}: "
                f"{len(missing_signal_files)} "
                f"missing .dat files.\n"
                f"Examples:\n"
                + "\n".join(
                    missing_signal_files[:10]
                ),
                sys.exc_info(),
            )

    def validate_split(
        self,
        split_name: str,
    ) -> dict:

        logger.info(
            "Starting preflight validation for %s.",
            split_name,
        )

        df = self.load_split(
            split_name
        )

        self.validate_columns(
            df,
            split_name,
        )

        self.validate_missing_values(
            df,
            split_name,
        )

        self.validate_targets(
            df,
            split_name,
        )

        self.validate_duplicate_ecg_ids(
            df,
            split_name,
        )

        self.validate_recording_files(
            df,
            split_name,
        )

        result = {
            "split": split_name,
            "ecg_count": len(df),
            "patient_count": df[
                "patient_id"
            ].nunique(),
            "non_mi_count": int(
                (df["target_mi"] == 0).sum()
            ),
            "mi_count": int(
                (df["target_mi"] == 1).sum()
            ),
        }

        logger.info(
            "%s preflight validation passed.",
            split_name,
        )

        return result

    def run(self):

        try:

            logger.info(
                "Starting complete image-generation "
                "preflight validation."
            )

            results = []

            for split_name in self.REQUIRED_SPLITS:

                result = self.validate_split(
                    split_name
                )

                results.append(
                    result
                )

            summary_df = pd.DataFrame(
                results
            )

            logger.info(
                "Complete image-generation "
                "preflight validation passed."
            )

            return summary_df

        except CardioVisionAIException:

            raise

        except Exception as error:

            logger.exception(
                "Image-generation preflight failed."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error