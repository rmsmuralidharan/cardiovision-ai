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


def test_scp_statements():

    print("=" * 70)
    print("PTB-XL SCP Statements Validation")
    print("=" * 70)

    if not SCP_FILE.exists():
        raise FileNotFoundError(
            f"SCP statements file not found: {SCP_FILE}"
        )

    print("\n[OK] scp_statements.csv exists")

    df = pd.read_csv(
        SCP_FILE,
        index_col=0
    )

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nIndex examples:")
    print(df.index.tolist()[:10])

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\n" + "=" * 70)
    print("SCP STATEMENTS VALIDATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_scp_statements()