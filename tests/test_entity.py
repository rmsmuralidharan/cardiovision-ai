from pathlib import Path

from entity.config_entity import (
    DataIngestionConfig,
    ImageGenerationConfig,
    ModelTrainingConfig,
)

from artifacts.artifact_entity import (
    DataIngestionArtifact,
    ImageGenerationArtifact,
    ModelTrainerArtifact,
)


def test_config_entities():

    data_config = DataIngestionConfig(
        raw_data_dir=Path("data/raw/ptbxl"),
        metadata_file=Path("data/raw/ptbxl/ptbxl_database.csv"),
        recordings_dir=Path("data/raw/ptbxl/records100"),
    )

    image_config = ImageGenerationConfig(
        output_dir=Path("data/processed/ecg_images"),
        image_width=1600,
        image_height=1200,
        dpi=100,
    )

    model_config = ModelTrainingConfig(
        model_dir=Path("models"),
        image_size=224,
        batch_size=32,
        epochs=20,
        learning_rate=0.001,
    )

    print("\nData Configuration:")
    print(data_config)

    print("\nImage Configuration:")
    print(image_config)

    print("\nModel Configuration:")
    print(model_config)


def test_artifact_entities():

    data_artifact = DataIngestionArtifact(
        metadata_file=Path("data/raw/ptbxl/ptbxl_database.csv"),
        recordings_dir=Path("data/raw/ptbxl/records100"),
    )

    image_artifact = ImageGenerationArtifact(
        image_directory=Path("data/processed/ecg_images"),
        image_count=0,
    )

    model_artifact = ModelTrainerArtifact(
        model_path=Path("models/baseline_cnn.keras"),
        training_history_path=Path(
            "artifacts/models/training_history.json"
        ),
    )

    print("\nData Artifact:")
    print(data_artifact)

    print("\nImage Artifact:")
    print(image_artifact)

    print("\nModel Artifact:")
    print(model_artifact)


if __name__ == "__main__":
    test_config_entities()
    test_artifact_entities()