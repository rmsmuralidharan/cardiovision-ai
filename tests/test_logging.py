from project_logging.logger import get_logger


logger = get_logger(__name__)


def test_logging():
    logger.info("CardioVision AI logging test started.")
    logger.info("Logging system is working correctly.")


if __name__ == "__main__":
    test_logging()