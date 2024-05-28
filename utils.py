import datetime
import json
import logging
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

