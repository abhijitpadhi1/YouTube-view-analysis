import os, sys
import yaml, pickle

# Importing custom modules and classes
from training.logging.logger import Logger
from training.exception.exception import CustomException

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    
    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    try:
        Logger().log(f"Reading YAML file from: {file_path}")
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        err = CustomException(str(e), sys)
        Logger().log(f"Error in read_yaml_file: {err}", level="error")
        raise err


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file.

    Args:
        file_path (str): The path to the YAML file.
        content (object): The content to write to the YAML file.
        replace (bool): Whether to replace the existing file. Default is False.

    Returns:
        None
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        Logger().log(f"Writing content to YAML file at: {file_path}")
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        err = CustomException(str(e), sys)
        Logger().log(f"Error in write_yaml_file: {err}", level="error")
        raise err
    

def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to save.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        Logger().log(f"Saving object to file at: {file_path}")
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        err = CustomException(str(e), sys)
        Logger().log(f"Error in save_object: {err}", level="error")
        raise err