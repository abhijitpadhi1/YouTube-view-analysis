import sys
from training.logging.logger import Logger

class CustomException(Exception):
    def __init__(self, message: str, error_detail) -> None:
        super().__init__(message)
        self.message = message
        _,  _, exc_tb = error_detail.exc_info()
        self.line_number = exc_tb.tb_lineno if exc_tb else 'Unknown'
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else 'Unknown'

    def __str__(self) -> str:
        return f"Error occurred in script: [{self.file_name}] at line number: [{self.line_number}] with message: [{self.message}]"
    
    def log_exception(self) -> None:
        """Log the exception details using the Logger class."""
        logger = Logger()
        logger.log(str(self.__str__()), 'error')


## Example execution
if __name__ == '__main__':
    try:
        logger = Logger()
        logger.log("This is an info message.")
        # Simulate an error
        a = 1 / 0
        print("Error occurred but not printed.")
    except Exception as e:
        Logger().log("An exception occurred. Logging the details.", 'error')
        custom_exception = CustomException("An error occurred during execution.", sys)
        custom_exception.log_exception()
        raise custom_exception
