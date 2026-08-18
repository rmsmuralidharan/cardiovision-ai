from components.image_generation_preflight import (
    ImageGenerationPreflight,
)


def test_image_generation_preflight():

    print("=" * 70)
    print("IMAGE GENERATION PREFLIGHT VALIDATION")
    print("=" * 70)

    preflight = (
        ImageGenerationPreflight()
    )

    summary_df = preflight.run()

    print("\nPreflight summary:")
    print(summary_df.to_string(index=False))

    print("\nTotal ECGs:")
    print(
        summary_df["ecg_count"].sum()
    )

    print("\nTotal patients:")
    print(
        summary_df["patient_count"].sum()
    )

    print("\nTotal Non-MI:")
    print(
        summary_df["non_mi_count"].sum()
    )

    print("\nTotal MI:")
    print(
        summary_df["mi_count"].sum()
    )

    print("\n" + "=" * 70)
    print("IMAGE GENERATION PREFLIGHT: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_image_generation_preflight()