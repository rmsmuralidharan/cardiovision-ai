import sys

from exception.exception import CardioVisionAIException


def divide_numbers():

    try:
        return 10 / 0

    except Exception as error:
        raise CardioVisionAIException(
            str(error),
            sys.exc_info()
        )


if __name__ == "__main__":
    divide_numbers()