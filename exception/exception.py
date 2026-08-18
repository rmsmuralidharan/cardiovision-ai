import sys


class CardioVisionAIException(Exception):
    """
    Custom exception class for CardioVision AI.
    """

    def __init__(
        self,
        error_message: str,
        error_detail=None
    ):
        super().__init__(error_message)

        self.error_message = error_message

        if error_detail is not None:
            _, _, traceback = error_detail

            if traceback is not None:
                self.file_name = traceback.tb_frame.f_code.co_filename
                self.line_number = traceback.tb_lineno
            else:
                self.file_name = __file__
                self.line_number = 0

        else:
            # No existing exception is being wrapped.
            # Identify the caller that raised this custom exception.
            frame = sys._getframe(1)

            self.file_name = frame.f_code.co_filename
            self.line_number = frame.f_lineno

    def __str__(self) -> str:
        return (
            f"Error occurred in file "
            f"[{self.file_name}] "
            f"at line [{self.line_number}]: "
            f"{self.error_message}"
        )