import datetime
import json
import logging
import os
from os.path import exists

import numpy as np

logger = logging.getLogger(__name__)

def to_json(data_dict, path):
    """Save dictionary as JSON."""
    def converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    with open(path, 'w') as file:
        logger.info(f'Save to: {path}')
        json.dump(data_dict, file, default=converter)


def open_json(path):
    """Read JSON file."""
    try:
        with open(path, 'r') as file:
            data_json = json.load(file)
            return data_json
    except:
        logger.error(f'Cannot open {path}')

def check_file_existence(path, raise_error = False):
    """
    Check file existence with given path
    """
    bool_result = exists(path)
    if bool_result:
        if raise_error:
            raise Exception("File " + str(path) + " has already existed!")
        else:
            print("File " + str(path) + " has already existed!")
    return bool_result

def create_dir(path):
    """Create directory if the input path is not found."""
    if not os.path.exists(path):
        logger.info(f'Creating directory: {path}')
        os.makedirs(path)

def get_csv_line_from_list(line_list):
    """
    Create a comma seperated line without new line character with given list
    """
    line_str = ""
    for value in line_list:
        line_str += str(value)
        line_str += ","
    return line_str

def create_empty_result_csv_file(path_save, first_line_list):

    if not check_file_existence(path_save):

        line = get_csv_line_from_list(first_line_list)

        with open(path_save, "a") as f:
            f.write(line)
            f.write("\n")
    
def append_result_to_csv_file(path_save, line_list):

    line = get_csv_line_from_list(line_list)

    with open(path_save, "a") as f:
            f.write(line)
            f.write("\n")