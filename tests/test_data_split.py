from components.data_split import PatientLevelDataSplitter


def test_data_split():

    splitter = PatientLevelDataSplitter()

    (
        train_df,
        validation_df,
        test_df,
    ) = splitter.initiate_data_split()

    print("=" * 70)
    print("PATIENT-LEVEL DATA SPLIT")
    print("=" * 70)

    print("\nTrain:")
    print("ECGs:", len(train_df))
    print(
        "Patients:",
        train_df["patient_id"].nunique(),
    )

    print("\nValidation:")
    print("ECGs:", len(validation_df))
    print(
        "Patients:",
        validation_df["patient_id"].nunique(),
    )

    print("\nTest:")
    print("ECGs:", len(test_df))
    print(
        "Patients:",
        test_df["patient_id"].nunique(),
    )

    print("\nTarget distribution:")

    print("\nTrain:")
    print(
        train_df["target_mi"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nValidation:")
    print(
        validation_df["target_mi"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nTest:")
    print(
        test_df["target_mi"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL SPLIT: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_data_split()