import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "labels"
    / "ptbxl_mi_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "splits"
)

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


class PatientLevelDataSplitter:
    """
    Splits PTB-XL data at the patient level.
    """

    def __init__(
        self,
        label_file: Path = LABEL_FILE,
        output_dir: Path = OUTPUT_DIR,
        random_state: int = RANDOM_STATE,
    ):

        self.label_file = Path(label_file)
        self.output_dir = Path(output_dir)
        self.random_state = random_state

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_data(self) -> pd.DataFrame:

        try:

            logger.info(
                "Loading labeled metadata from: %s",
                self.label_file,
            )

            df = pd.read_csv(self.label_file)

            logger.info(
                "Loaded %s ECG records.",
                len(df),
            )

            return df

        except Exception as error:

            logger.exception(
                "Failed to load labeled metadata."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    @staticmethod
    def create_patient_table(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        patient_df = (
            df.groupby("patient_id", as_index=False)
            .agg(
                patient_mi=("target_mi", "max")
            )
        )

        return patient_df

    def split_patients(
        self,
        patient_df: pd.DataFrame,
    ):

        try:

            logger.info(
                "Starting patient-level splitting."
            )

            train_patients, temp_patients = (
                train_test_split(
                    patient_df,
                    test_size=0.30,
                    random_state=self.random_state,
                    stratify=patient_df["patient_mi"],
                )
            )

            validation_patients, test_patients = (
                train_test_split(
                    temp_patients,
                    test_size=0.50,
                    random_state=self.random_state,
                    stratify=temp_patients["patient_mi"],
                )
            )

            logger.info(
                "Train patients: %s",
                len(train_patients),
            )

            logger.info(
                "Validation patients: %s",
                len(validation_patients),
            )

            logger.info(
                "Test patients: %s",
                len(test_patients),
            )

            return (
                train_patients,
                validation_patients,
                test_patients,
            )

        except Exception as error:

            logger.exception(
                "Patient-level splitting failed."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    @staticmethod
    def assign_split(
        df: pd.DataFrame,
        train_patients: pd.DataFrame,
        validation_patients: pd.DataFrame,
        test_patients: pd.DataFrame,
    ) -> pd.DataFrame:

        train_ids = set(
            train_patients["patient_id"]
        )

        validation_ids = set(
            validation_patients["patient_id"]
        )

        test_ids = set(
            test_patients["patient_id"]
        )

        result = df.copy()

        result["split"] = result["patient_id"].apply(
            lambda patient_id: (
                "train"
                if patient_id in train_ids
                else "validation"
                if patient_id in validation_ids
                else "test"
                if patient_id in test_ids
                else "unknown"
            )
        )

        return result

    @staticmethod
    def validate_no_patient_overlap(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        train_ids = set(
            train_df["patient_id"]
        )

        validation_ids = set(
            validation_df["patient_id"]
        )

        test_ids = set(
            test_df["patient_id"]
        )

        train_validation_overlap = (
            train_ids & validation_ids
        )

        train_test_overlap = (
            train_ids & test_ids
        )

        validation_test_overlap = (
            validation_ids & test_ids
        )

        if train_validation_overlap:
            raise ValueError(
                "Patient overlap between train and validation."
            )

        if train_test_overlap:
            raise ValueError(
                "Patient overlap between train and test."
            )

        if validation_test_overlap:
            raise ValueError(
                "Patient overlap between validation and test."
            )

        logger.info(
            "Patient overlap validation passed."
        )

    def save_splits(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):

        train_file = (
            self.output_dir
            / "train.csv"
        )

        validation_file = (
            self.output_dir
            / "validation.csv"
        )

        test_file = (
            self.output_dir
            / "test.csv"
        )

        train_df.to_csv(
            train_file,
            index=False,
        )

        validation_df.to_csv(
            validation_file,
            index=False,
        )

        test_df.to_csv(
            test_file,
            index=False,
        )

        logger.info(
            "Train split saved to: %s",
            train_file,
        )

        logger.info(
            "Validation split saved to: %s",
            validation_file,
        )

        logger.info(
            "Test split saved to: %s",
            test_file,
        )

    def initiate_data_split(self):

        logger.info(
            "Starting patient-level data splitting."
        )

        df = self.load_data()

        patient_df = self.create_patient_table(
            df
        )

        (
            train_patients,
            validation_patients,
            test_patients,
        ) = self.split_patients(
            patient_df
        )

        split_df = self.assign_split(
            df,
            train_patients,
            validation_patients,
            test_patients,
        )

        if (
            split_df["split"] == "unknown"
        ).any():

            raise CardioVisionAIException(
                "Some patients were not assigned to a split."
            )

        train_df = split_df[
            split_df["split"] == "train"
        ].copy()

        validation_df = split_df[
            split_df["split"] == "validation"
        ].copy()

        test_df = split_df[
            split_df["split"] == "test"
        ].copy()

        self.validate_no_patient_overlap(
            train_df,
            validation_df,
            test_df,
        )

        self.save_splits(
            train_df,
            validation_df,
            test_df,
        )

        logger.info(
            "Patient-level data splitting completed."
        )

        return (
            train_df,
            validation_df,
            test_df,
        )