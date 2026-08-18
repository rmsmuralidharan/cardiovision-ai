from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCP_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptbxl"
    / "scp_statements.csv"
)


def test_mi_mapping():

    print("=" * 70)
    print("PTB-XL MI Diagnostic Mapping")
    print("=" * 70)

    scp_df = pd.read_csv(
        SCP_FILE,
        index_col=0
    )

    print("\nSCP statements shape:")
    print(scp_df.shape)

    print("\nColumns:")
    print(scp_df.columns.tolist())

    # Show diagnostic statements
    diagnostic_df = scp_df[
        scp_df["diagnostic"] == 1
    ]

    print("\nNumber of diagnostic statements:")
    print(len(diagnostic_df))

    print("\nDiagnostic classes:")
    print(
        diagnostic_df["diagnostic_class"]
        .value_counts(dropna=False)
    )

    # Show MI statements
    mi_df = diagnostic_df[
        diagnostic_df["diagnostic_class"] == "MI"
    ]

    print("\nMI-related SCP statements:")
    print(mi_df)

    print("\nMI statement codes:")
    print(mi_df.index.tolist())


if __name__ == "__main__":
    test_mi_mapping()