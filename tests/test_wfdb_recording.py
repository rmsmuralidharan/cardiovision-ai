from pathlib import Path

import wfdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECORDINGS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptbxl"
    / "records100"
)


def find_first_header():
    """Find the first PTB-XL WFDB header file."""

    header_files = sorted(RECORDINGS_DIR.rglob("*.hea"))

    if not header_files:
        raise FileNotFoundError(
            f"No .hea files found in {RECORDINGS_DIR}"
        )

    return header_files[0]


def test_first_ecg_recording():

    header_file = find_first_header()

    # WFDB expects the record path without .hea
    record_path = header_file.with_suffix("")

    print("\nFirst ECG header:")
    print(header_file)

    print("\nWFDB record:")
    print(record_path)

    record = wfdb.rdrecord(str(record_path))

    print("\nECG verification")
    print("----------------------------")

    print("Sampling frequency:", record.fs)
    print("Number of samples:", record.sig_len)
    print("Number of leads:", record.n_sig)

    print("\nLead names:")
    print(record.sig_name)

    print("\nSignal shape:")
    print(record.p_signal.shape)


if __name__ == "__main__":
    test_first_ecg_recording()