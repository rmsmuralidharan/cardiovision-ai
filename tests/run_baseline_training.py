from components.cnn_trainer import CNNTrainer


def run_baseline_training():

    print("=" * 70)
    print("CARDIOVISION AI — BASELINE CNN TRAINING")
    print("=" * 70)

    trainer = CNNTrainer(
        epochs=30,
        batch_size=32,
        learning_rate=0.0001,
    )

    model, history = trainer.train()

    print("\n" + "=" * 70)
    print("BASELINE CNN TRAINING COMPLETED")
    print("=" * 70)

    print("\nFinal training metrics:")

    final_epoch = len(
        history.history["loss"]
    )

    print(
        "Epochs completed:",
        final_epoch,
    )

    print(
        "Training loss:",
        history.history["loss"][-1],
    )

    print(
        "Training AUC:",
        history.history["auc"][-1],
    )

    print(
        "Training recall:",
        history.history["recall"][-1],
    )

    print(
        "Validation loss:",
        history.history["val_loss"][-1],
    )

    print(
        "Validation AUC:",
        history.history["val_auc"][-1],
    )

    print(
        "Validation recall:",
        history.history["val_recall"][-1],
    )

    print("\n" + "=" * 70)
    print("BEST MODEL:")
    print(
        "artifacts/models/"
        "cardiovision_baseline_best.keras"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_baseline_training()