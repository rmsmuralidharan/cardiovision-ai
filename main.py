import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from configuration.configuration import ConfigurationManager
from exception.exception import CardioVisionAIException
from project_logging.logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger(__name__)


def verify_project() -> None:
    """Perform complete CardioVision AI foundation verification."""

    logger.info("Starting CardioVision AI project verification.")

    print("=" * 70)
    print("CardioVision AI - Complete Project Verification")
    print("=" * 70)

    verification_passed = True

    # ---------------------------------------------------------
    # 1. Python
    # ---------------------------------------------------------

    print(f"\nPython version : {sys.version.split()[0]}")
    print(f"Project root   : {PROJECT_ROOT}")

    # ---------------------------------------------------------
    # 2. Environment variables
    # ---------------------------------------------------------

    project_name = os.getenv("PROJECT_NAME")
    environment = os.getenv("ENVIRONMENT")

    print(f"Project name   : {project_name}")
    print(f"Environment    : {environment}")

    if not project_name:
        verification_passed = False
        print("  [FAIL] PROJECT_NAME")

    else:
        print("  [OK] PROJECT_NAME")

    if not environment:
        verification_passed = False
        print("  [FAIL] ENVIRONMENT")

    else:
        print("  [OK] ENVIRONMENT")

    # ---------------------------------------------------------
    # 3. Required directories
    # ---------------------------------------------------------

    required_directories = [
        "data",
        "data/raw",
        "data/raw/ptbxl",
        "data/interim",
        "data/interim/labels",
        "data/interim/splits",
        "data/processed",
        "data/processed/ecg_images",
        "notebooks",
        "components",
        "entity",
        "configuration",
        "constants",
        "exception",
        "project_logging",
        "pipeline",
        "models",
        "artifacts",
        "app",
        "app/pages",
        "app/utils",
        "tests",
        "config",
        "logs",
    ]

    print("\nDirectory verification:")

    for directory in required_directories:

        path = PROJECT_ROOT / directory

        if path.exists() and path.is_dir():
            print(f"  [OK] {directory}")

        else:
            print(f"  [MISSING] {directory}")
            verification_passed = False

    # ---------------------------------------------------------
    # 4. Required files
    # ---------------------------------------------------------

    required_files = [
        "setup.py",
        "requirements.txt",
        ".gitignore",
        ".env",
        "README.md",
        "config/config.yaml",
        "project_logging/logger.py",
        "exception/exception.py",
        "configuration/configuration.py",
        "entity/config_entity.py",
        "entity/artifact_entity.py",
    ]

    print("\nFile verification:")

    for file in required_files:

        path = PROJECT_ROOT / file

        if path.exists() and path.is_file():
            print(f"  [OK] {file}")

        else:
            print(f"  [MISSING] {file}")
            verification_passed = False

    # ---------------------------------------------------------
    # 5. Python package verification
    # ---------------------------------------------------------

    print("\nPackage verification:")

    packages = [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Matplotlib", "matplotlib"),
        ("Scikit-learn", "sklearn"),
        ("OpenCV", "cv2"),
        ("Pillow", "PIL"),
        ("WFDB", "wfdb"),
        ("TensorFlow", "tensorflow"),
        ("PyYAML", "yaml"),
        ("dotenv", "dotenv"),
    ]

    for package_name, module_name in packages:

        try:

            module = __import__(module_name)

            version = getattr(module, "__version__", "installed")

            print(f"  [OK] {package_name}: {version}")

        except ImportError:

            print(f"  [FAIL] {package_name}")
            verification_passed = False

    # ---------------------------------------------------------
    # 6. Project package verification
    # ---------------------------------------------------------

    print("\nProject package verification:")

    project_packages = [
        "components",
        "entity",
        "configuration",
        "constants",
        "exception",
        "project_logging",
        "pipeline",
    ]

    for package_name in project_packages:

        try:

            __import__(package_name)

            print(f"  [OK] {package_name}")

        except ImportError:

            print(f"  [FAIL] {package_name}")
            verification_passed = False

    # ---------------------------------------------------------
    # 7. Configuration verification
    # ---------------------------------------------------------

    print("\nConfiguration verification:")

    try:

        configuration = ConfigurationManager()

        config = configuration.read_yaml()

        print("  [OK] Environment configuration")
        print("  [OK] YAML configuration")

        print(f"  [OK] Project: {configuration.project_name}")
        print(f"  [OK] Environment: {configuration.environment}")

        if not isinstance(config, dict):
            raise TypeError(
                "config.yaml did not return a dictionary."
            )

        print("  [OK] YAML structure")

    except Exception as error:

        verification_passed = False

        logger.exception("Configuration verification failed.")

        print(
            CardioVisionAIException(
                str(error),
                sys.exc_info()
            )
        )

    # ---------------------------------------------------------
    # 8. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    if verification_passed:

        logger.info(
            "CardioVision AI project verification PASSED."
        )

        print("PROJECT VERIFICATION: PASSED")

    else:

        logger.error(
            "CardioVision AI project verification FAILED."
        )

        print("PROJECT VERIFICATION: FAILED")
        print("Fix the issues above before continuing.")

    print("=" * 70)


if __name__ == "__main__":
    verify_project()