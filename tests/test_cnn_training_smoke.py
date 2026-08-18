from pathlib import Path

import pandas as pd

from components.cnn_trainer import CNNTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecg_images"
)


def test_cnn_training_smoke():

    print("=" * 70)
    print("CNN TRAINING SMOKE TEST")
    print("=" * 70)

    train_df = pd.read_csv(
        IMAGE_DIR
        / "train_manifest.csv"
    )

    validation_df = pd.read_csv(
        IMAGE_DIR
        / "validation_manifest.csv"
    )

    # ---------------------------------------------
    # Tiny dataset
    # ---------------------------------------------

    train_sample = train_df.head(64)

    validation_sample = (
        validation_df.head(32)
    )

    print("\nTrain sample:")
    print(len(train_sample))

    print("\nValidation sample:")
    print(len(validation_sample))

    # ---------------------------------------------
    # Create trainer
    # ---------------------------------------------

    trainer = CNNTrainer(
        epochs=1,
        batch_size=32,
    )

    # ---------------------------------------------
    # Build datasets
    # ---------------------------------------------

    train_dataset, validation_dataset = (
        trainer.prepare_datasets(
            train_sample,
            validation_sample,
        )
    )

    # ---------------------------------------------
    # Class weights
    # ---------------------------------------------

    class_weights = (
        trainer.calculate_class_weights(
            train_sample
        )
    )

    print("\nClass weights:")
    print(class_weights)

    # ---------------------------------------------
    # Build model
    # ---------------------------------------------

    model = (
        trainer.model_builder.build_model()
    )

    callbacks = (
        trainer.callback_builder
        .create_callbacks()
    )

    # ---------------------------------------------
    # One epoch only
    # ---------------------------------------------

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=1,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        shuffle=False
    )

    # ---------------------------------------------
    # Validate history
    # ---------------------------------------------

    assert history is not None

    assert (
        len(history.history["loss"])
        == 1
    )

    assert (
        "val_auc"
        in history.history
    )

    print("\nTraining history keys:")

    print(
        list(
            history.history.keys()
        )
    )

    print("\n" + "=" * 70)
    print("CNN TRAINING SMOKE TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_cnn_training_smoke()