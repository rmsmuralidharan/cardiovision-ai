from components.training_callbacks import (
    TrainingCallbacks,
)


def test_training_callbacks():

    print("=" * 70)
    print("TRAINING CALLBACK TEST")
    print("=" * 70)

    callback_builder = (
        TrainingCallbacks()
    )

    callbacks = (
        callback_builder.create_callbacks()
    )

    print("\nNumber of callbacks:")
    print(len(callbacks))

    print("\nCallback types:")

    for callback in callbacks:

        print(
            type(callback).__name__
        )

    assert len(callbacks) == 4

    callback_names = [
        type(callback).__name__
        for callback in callbacks
    ]

    assert (
        "EarlyStopping"
        in callback_names
    )

    assert (
        "ModelCheckpoint"
        in callback_names
    )

    assert (
        "ReduceLROnPlateau"
        in callback_names
    )

    assert (
        "TensorBoard"
        in callback_names
    )

    print("\n" + "=" * 70)
    print("TRAINING CALLBACK TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_training_callbacks()