import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from exception.exception import CardioVisionAIException


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"


load_dotenv(ENV_FILE)


class ConfigurationManager:
    """
    Handles CardioVision AI project configuration.
    """

    def __init__(self):

        self.project_name = os.getenv("PROJECT_NAME")
        self.environment = os.getenv("ENVIRONMENT")

        if not self.project_name:
            raise CardioVisionAIException(
                "PROJECT_NAME is missing from .env"
            )

        if not self.environment:
            raise CardioVisionAIException(
                "ENVIRONMENT is missing from .env"
            )

    @staticmethod
    def read_yaml() -> dict:
        """
        Read and return config.yaml contents.
        """

        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            if config is None:
                raise ValueError("config.yaml is empty.")

            return config

        except Exception as error:

            raise CardioVisionAIException(
                str(error),
                sys.exc_info()
            )