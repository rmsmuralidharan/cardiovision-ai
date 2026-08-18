import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptbxl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecg_images"
)


class ECGImageGenerator:
    """
    Convert PTB-XL 12-lead ECG recordings into standardized images.
    """

    EXPECTED_LEADS = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_recording(
        self,
        record_path: Path,
    ):
        """
        Load one WFDB ECG recording.
        """

        try:

            logger.info(
                "Loading ECG recording: %s",
                record_path,
            )

            record = wfdb.rdrecord(
                str(record_path)
            )

            if record.p_signal is None:
                raise ValueError(
                    "ECG recording contains no physical signal."
                )

            signal = record.p_signal

            logger.info(
                "ECG loaded. Shape: %s",
                signal.shape,
            )

            return record, signal

        except Exception as error:

            logger.exception(
                "Failed to load ECG recording."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error

    def validate_recording(
        self,
        record,
        signal: np.ndarray,
    ):
        """
        Validate the ECG before generating an image.
        """

        if signal.ndim != 2:
            raise CardioVisionAIException(
                f"Expected a 2D ECG signal, got shape {signal.shape}."
            )

        if signal.shape[1] != 12:
            raise CardioVisionAIException(
                f"Expected 12 ECG leads, got {signal.shape[1]}."
            )

        if np.isnan(signal).any():
            raise CardioVisionAIException(
                "ECG contains NaN values."
            )

        if np.isinf(signal).any():
            raise CardioVisionAIException(
                "ECG contains infinite values."
            )

        # Normalize lead names for comparison.
        actual_leads = {
            lead.strip().upper(): lead
            for lead in record.sig_name
        }

        expected_leads = {
            lead.upper()
            for lead in self.EXPECTED_LEADS
        }

        missing_leads = (
            expected_leads - actual_leads.keys()
        )

        if missing_leads:
            raise CardioVisionAIException(
                f"Missing expected ECG leads: "
                f"{sorted(missing_leads)}"
            )

        logger.info(
            "ECG recording validation passed."
        )

    def generate_image(
        self,
        record,
        signal: np.ndarray,
        ecg_id: int,
    ) -> Path:
        """
        Generate one standardized 12-lead ECG image.
        """

        try:

            logger.info(
                "Generating ECG image for ECG ID: %s",
                ecg_id,
            )

            # -----------------------------------------------------
            # 1. Normalize lead names and determine column order
            # -----------------------------------------------------

            lead_index_map = {
                lead.strip().upper(): index
                for index, lead in enumerate(record.sig_name)
            }

            lead_indices = [
                lead_index_map[lead.upper()]
                for lead in self.EXPECTED_LEADS
            ]

            signal = signal[:, lead_indices]

            # -----------------------------------------------------
            # 2. Time axis
            # -----------------------------------------------------

            sampling_frequency = float(record.fs)

            time = (
                np.arange(signal.shape[0])
                / sampling_frequency
            )

            # -----------------------------------------------------
            # 3. Determine common amplitude range
            # -----------------------------------------------------

            amplitude_min = np.min(signal)
            amplitude_max = np.max(signal)

            amplitude_range = (
                amplitude_max - amplitude_min
            )

            padding = amplitude_range * 0.05

            y_min = amplitude_min - padding
            y_max = amplitude_max + padding

            # -----------------------------------------------------
            # 4. Create figure
            # -----------------------------------------------------

            fig, axes = plt.subplots(
                nrows=4,
                ncols=3,
                figsize=(15, 10),
                sharex=True,
            )

            axes = axes.flatten()

            # -----------------------------------------------------
            # 5. Plot each lead
            # -----------------------------------------------------

            for lead_index, lead_name in enumerate(
                self.EXPECTED_LEADS
            ):

                ax = axes[lead_index]

                ax.plot(
                    time,
                    signal[:, lead_index],
                    linewidth=0.8,
                )

                ax.set_title(
                    lead_name,
                    fontsize=10,
                    loc="left",
                )

                ax.set_xlim(
                    time[0],
                    time[-1],
                )

                ax.set_ylim(
                    y_min,
                    y_max,
                )

                # Major grid
                ax.grid(
                    which="major",
                    linewidth=0.5,
                    alpha=0.5,
                )

                # Minor grid
                ax.minorticks_on()

                ax.grid(
                    which="minor",
                    linewidth=0.2,
                    alpha=0.25,
                )

                ax.tick_params(
                    labelsize=7
                )

                ax.set_xlabel("")

                ax.set_ylabel("")

            # -----------------------------------------------------
            # 6. Remove unnecessary bottom labels
            # -----------------------------------------------------

            for ax in axes:
                ax.label_outer()

            # -----------------------------------------------------
            # 7. Layout
            # -----------------------------------------------------

            fig.suptitle(
                "12-Lead ECG",
                fontsize=14,
            )

            plt.tight_layout(
                rect=[0, 0, 1, 0.97]
            )

            # -----------------------------------------------------
            # 8. Save image
            # -----------------------------------------------------

            output_file = (
                self.output_dir
                / f"ecg_{ecg_id}.png"
            )

            fig.savefig(
                output_file,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

            logger.info(
                "ECG image saved to: %s",
                output_file,
            )

            return output_file

        except Exception as error:

            plt.close("all")

            logger.exception(
                "Failed to generate ECG image."
            )

            raise CardioVisionAIException(
                str(error),
                sys.exc_info(),
            ) from error