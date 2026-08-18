from components.label_engineering import MILabelEngineer


def test_label_engineering():

    label_engineer = MILabelEngineer()

    labeled_df = (
        label_engineer.initiate_label_engineering()
    )

    print("=" * 70)
    print("MI LABEL ENGINEERING RESULT")
    print("=" * 70)

    print("\nShape:")
    print(labeled_df.shape)

    print("\nTarget distribution:")
    print(
        labeled_df["target_mi"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget percentages:")
    print(
        labeled_df["target_mi"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print("\nMI likelihood statistics:")
    print(
        labeled_df[
            labeled_df["target_mi"] == 1
        ]["mi_max_likelihood"]
        .describe()
    )

    print("\nSample labeled records:")
    print(
        labeled_df[
            [
                "ecg_id",
                "patient_id",
                "target_mi",
                "mi_codes",
                "mi_max_likelihood",
            ]
        ].head(10)
    )

    print("\n" + "=" * 70)
    print("MI LABEL ENGINEERING: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_label_engineering()