from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    raw_data_dir: Path
    metadata_file: Path
    recordings_dir: Path


@dataclass
class ImageGenerationConfig:
    output_dir: Path
    image_width: int
    image_height: int
    dpi: int


@dataclass
class ModelTrainingConfig:
    model_dir: Path
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float