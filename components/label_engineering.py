import ast
import sys
from pathlib import Path

import pandas as pd

from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ptbxl"

METADATA_FILE = RAW_DATA_DIR / "ptbxl_database.csv"

SCP_FILE = RAW_DATA_DIR / "scp_statements.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "labels"
)

OUTPUT_FILE = OUTPUT_DIR / "ptbxl_mi_labels.csv"


class MILabelEngineer:
    """
    Creates MI/non-MI labels from PTB-XL diagnostic SCP codes.
    """

    def __init__(
        self,
        metadata_file: Path = METADATA_FILE,
        scp_file: Path = SCP_FILE,
        output_file: Path = OUTPUT_FILE,
    ):

        self.metadata_file = Path(metadata_file)
        self.scp_file = Path(scp_file)
        self.output_file = Path(output_file)

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def parse_scp_codes(value):
        """
        Convert the scp_codes string into a Python dictionary.
        """

        if pd.isna(value):
            return {}

        if isinstance(value, dict):
            return value

        try:
            parsed_value = ast.literal_eval(str(value))

            if not isinstance(parsed_value, dict):
                raise ValueError(
                    "scp_codes did not contain a dictionary."
                )

            return parsed_value

        except Exception as error:

            raise CardioVisionAIException(
                f"Failed to parse scp_codes: {value}",
                sys.exc_info(),
            ) from error

    def load_data(self):
        """
        Load PTB-XL metadata and SCP statements.
        """

        try:

            logger.info("Loading PTB-XL metadata.")

            metadata_df = pd.read_csv(
                self.metadata_file
            )

            logger.info(
                "Loaded PTB-XL metadata: %s rows.",
                len(metadata_df),
            )

            logger.info("Loading SCP statements.")

            scp_df = pd.read_csv(
                self.scp_file,
                index_col=0,
            )

            logger.info(
                "Loaded SCP statements: %s rows.",
                len(scp_df),
            )

            return metadata_df, scp_df

        except Exception as error:

            logger.exception(
                "Failed to load PTB-XL label-engineering data."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    def get_mi_codes(self, scp_df):
        """
        Identify SCP codes belonging to the MI diagnostic class.
        """

        try:

            diagnostic_df = scp_df[
                scp_df["diagnostic"] == 1
            ]

            mi_df = diagnostic_df[
                diagnostic_df["diagnostic_class"] == "MI"
            ]

            mi_codes = set(
                mi_df.index.astype(str)
            )

            logger.info(
                "Identified %s MI-related SCP codes.",
                len(mi_codes),
            )

            logger.info(
                "MI codes: %s",
                sorted(mi_codes),
            )

            return mi_codes

        except Exception as error:

            logger.exception(
                "Failed to identify MI SCP codes."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    def create_labels(
        self,
        metadata_df,
        mi_codes,
    ):
        """
        Create MI labels from scp_codes.
        """

        try:

            logger.info(
                "Creating MI labels for PTB-XL recordings."
            )

            metadata_df = metadata_df.copy()

            metadata_df["parsed_scp_codes"] = (
                metadata_df["scp_codes"]
                .apply(self.parse_scp_codes)
            )

            metadata_df["mi_codes"] = (
                metadata_df["parsed_scp_codes"]
                .apply(
                    lambda codes: sorted(
                        set(codes.keys()) & mi_codes
                    )
                )
            )

            metadata_df["target_mi"] = (
                metadata_df["mi_codes"]
                .apply(
                    lambda codes: int(len(codes) > 0)
                )
            )

            metadata_df["mi_max_likelihood"] = (
                metadata_df.apply(
                    lambda row: max(
                        [
                            float(
                                row["parsed_scp_codes"][code]
                            )
                            for code in row["mi_codes"]
                        ],
                        default=0.0,
                    ),
                    axis=1,
                )
            )

            logger.info(
                "MI-positive recordings: %s",
                int(metadata_df["target_mi"].sum()),
            )

            logger.info(
                "Non-MI recordings: %s",
                int(
                    (metadata_df["target_mi"] == 0).sum()
                ),
            )

            return metadata_df

        except Exception as error:

            logger.exception(
                "Failed to create MI labels."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    def save_labels(self, labeled_df):
        """
        Save the labeled metadata to the interim directory.
        """

        try:

            logger.info(
                "Saving MI-labeled metadata to: %s",
                self.output_file,
            )

            output_df = labeled_df.drop(
                columns=["parsed_scp_codes"],
                errors="ignore",
            )

            output_df.to_csv(
                self.output_file,
                index=False,
            )

            logger.info(
                "MI-labeled metadata saved successfully."
            )

        except Exception as error:

            logger.exception(
                "Failed to save MI-labeled metadata."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    def initiate_label_engineering(self):
        """
        Execute the complete MI label-engineering process.
        """

        logger.info(
            "Starting MI label engineering."
        )

        metadata_df, scp_df = self.load_data()

        mi_codes = self.get_mi_codes(
            scp_df
        )

        labeled_df = self.create_labels(
            metadata_df,
            mi_codes,
        )

        self.save_labels(
            labeled_df
        )

        logger.info(
            "MI label engineering completed."
        )

        return labeled_df