import sys
from pathlib import Path

import pandas as pd

from components.ecg_image_generator import ECGImageGenerator
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

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecg_images"
)


class ECGImageDatasetBuilder:
    """
    Builds ECG image datasets from train,
    validation, and test split metadata.
    """

    def __init__(
        self,
        ptbxl_dir: Path = PTBXL_DIR,
        split_dir: Path = SPLIT_DIR,
        image_dir: Path = IMAGE_DIR,
    ):

        self.ptbxl_dir = Path(ptbxl_dir)
        self.split_dir = Path(split_dir)
        self.image_dir = Path(image_dir)

        self.image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def build_split(
        self,
        split_name: str,
    ) -> pd.DataFrame:

        try:

            # -------------------------------------------------
            # 1. Locate split file
            # -------------------------------------------------

            split_file = (
                self.split_dir
                / f"{split_name}.csv"
            )

            if not split_file.exists():

                raise FileNotFoundError(
                    f"Split file not found: "
                    f"{split_file}"
                )

            logger.info(
                "Loading %s split from: %s",
                split_name,
                split_file,
            )

            df = pd.read_csv(
                split_file
            )

            logger.info(
                "%s split contains %s ECG records.",
                split_name,
                len(df),
            )

            # -------------------------------------------------
            # 2. Create split-specific image directory
            # -------------------------------------------------

            split_image_dir = (
                self.image_dir
                / split_name
            )

            split_image_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # -------------------------------------------------
            # 3. Create ECG image generator
            # -------------------------------------------------

            generator = ECGImageGenerator(
                output_dir=split_image_dir
            )

            logger.info(
            "Image output directory: %s",
            split_image_dir,
            )

            results = []

            generated_count = 0
            skipped_count = 0
            failed_count = 0

            failed_records = []


            # -------------------------------------------------
            # 4. Process every ECG
            # -------------------------------------------------

            for index, row in df.iterrows():

                ecg_id = int(
                    row["ecg_id"]
                )

                patient_id = int(
                    row["patient_id"]
                )

                target_mi = int(
                    row["target_mi"]
                )

                # -------------------------------------------------
                # Expected image path
                # -------------------------------------------------

                expected_image_path = (
                    split_image_dir
                    / f"ecg_{ecg_id}.png"
                )

                # -------------------------------------------------
                # Skip existing valid images
                # -------------------------------------------------

                if (
                    expected_image_path.exists()
                    and expected_image_path.stat().st_size > 0
                ):

                    logger.info(
                        "Image already exists for ECG %s. Skipping.",
                        ecg_id,
                    )

                    skipped_count += 1

                    results.append(
                        {
                            "ecg_id": ecg_id,
                            "patient_id": patient_id,
                            "target_mi": target_mi,
                            "image_path": str(
                                expected_image_path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),
                        }
                    )

                    continue

                # -------------------------------------------------
                # Locate ECG recording
                # -------------------------------------------------

                relative_record_path = Path(
                    row["filename_lr"]
                )

                record_path = (
                    self.ptbxl_dir
                    / relative_record_path
                )

                try:

                    logger.info(
                        "Processing %s ECG %s (%s/%s)",
                        split_name,
                        ecg_id,
                        index + 1,
                        len(df),
                    )

                    # -------------------------------------------------
                    # Load ECG
                    # -------------------------------------------------

                    record, signal = (
                        generator.load_recording(
                            record_path
                        )
                    )

                    # -------------------------------------------------
                    # Validate ECG
                    # -------------------------------------------------

                    generator.validate_recording(
                        record,
                        signal,
                    )

                    # -------------------------------------------------
                    # Generate image
                    # -------------------------------------------------

                    image_path = (
                        generator.generate_image(
                            record=record,
                            signal=signal,
                            ecg_id=ecg_id,
                        )
                    )

                    generated_count += 1

                    # -------------------------------------------------
                    # Store metadata
                    # -------------------------------------------------

                    results.append(
                        {
                            "ecg_id": ecg_id,
                            "patient_id": patient_id,
                            "target_mi": target_mi,
                            "image_path": str(
                                image_path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),
                        }
                    )
                except Exception as error:
                    failed_count += 1

                    failed_records.append(
                        {
                            "ecg_id": ecg_id,
                            "patient_id": patient_id,
                            "target_mi": target_mi,
                            'record_path': str(
                                record_path
                            ),
                            'error': str(error),
                        }
                    )

                    logger.exception(
                        "Failed to process ECG %s in %s split.",
                        ecg_id,
                        split_name,
                    )

                    continue

            # -------------------------------------------------
            # 5. Create manifest
            # -------------------------------------------------

            manifest_df = pd.DataFrame(
                results
            )

            manifest_file = (
                self.image_dir
                / f"{split_name}_manifest.csv"
            )

            manifest_df.to_csv(
                manifest_file,
                index=False,
            )

            if failed_records:
                failed_df = pd.DataFrame(
                    failed_records
                )

                failed_file = (
                    self.image_dir
                    / f"{split_name}_failed.csv"
                )

                failed_df.to_csv(
                    failed_file,
                    index=False,
                )

                logger.warning(
                    "%s ECG records failed to process. "
                    "See %s for details.",
                    len(failed_records),
                    failed_file,
                )
            else:
                logger.info(
                    "No ECG failures in %s split.",
                    split_name
                )

            logger.info(
                "%s image manifest saved to: %s",
                split_name,
                manifest_file,
            )

            logger.info(
                "%s generation summary: "
                "generated=%s, skipped=%s, failed=%s, total=%s",
                split_name,
                generated_count,
                skipped_count,
                failed_count,
                len(df)
            )

            expected_count = len(df)

            successful_count = generated_count + skipped_count

            if successful_count + failed_count != expected_count:
                raise CardioVisionAIException(
                    f"Generation accounting mismatch for "
                    f"{split_name} : "
                    f"generated={generated_count}, "
                    f"skipped={skipped_count}, "
                    f"failed={failed_count}, "
                    f"expected={expected_count}"
                )

            return manifest_df

        except Exception as error:

            logger.exception(
                "Failed to build %s image dataset.",
                split_name,
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    def build_all_splits(self):

        logger.info(
            "Starting complete ECG image dataset construction."
        )

        train_manifest = self.build_split(
            "train"
        )

        validation_manifest = self.build_split(
            "validation"
        )

        test_manifest = self.build_split(
            "test"
        )

        logger.info(
            "Complete ECG image dataset construction completed."
        )

        return (
            train_manifest,
            validation_manifest,
            test_manifest,
        )