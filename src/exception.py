import sys

from src.logger import logging


def return_error_message(error, detail: sys) -> str:
    _, _, execution_info = detail.exc_info()
    filename: str = execution_info.tb_frame.f_code.co_filename
    line_number: int = execution_info.tb_lineno
    message = f"{str(error)} --> line {line_number} from '{filename}'"
    return message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message: str = return_error_message(error_message, error_detail)

    def __str__(self):
        return self.error_message
