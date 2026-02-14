import os
import logging
from datetime import datetime

# Define log file name with timestamp and log directory
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs")

class Logger:
    def __init__(self) -> None:
        # Create logs directory if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_file_path = os.path.join(LOG_DIR, LOG_FILE)
        logging.basicConfig(
            filename=self.log_file_path,
            format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    # Method to log messages
    def log(self, message: str, level: str = 'info') -> None:
        """Log a message to the log file."""
        if level == 'info':
            logging.info(msg=message)
        elif level == 'error':
            logging.error(msg=message, exc_info=True)
        elif level == 'warning':
            logging.warning(msg=message, exc_info=True)

    def get_log_file_path(self) -> str:
        """Get the path of the log file."""
        return self.log_file_path
    
