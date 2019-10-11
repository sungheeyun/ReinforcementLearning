from typing import Any
import json
from typing import Optional
import logging
import os
import datetime

import numpy as np


def get_now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def set_logging_basic_config(
        main_python_file_name: str,
        level: int = logging.INFO,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_dir: Optional[str] = None,
):
    """
    Sets basic logging configuration.
    """
    main_python_file_name_root = os.path.splitext(os.path.split(main_python_file_name)[1])[0]

    if log_dir is None:
        log_dir = os.path.join(os.curdir, "log")

    if os.path.exists(log_dir):
        if not os.path.isdir(log_dir):
            raise Exception(f"{log_dir} exists, but not a directory.")
    else:
        os.mkdir(log_dir)

    logging.basicConfig(
        level=level,
        format=format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{main_python_file_name_root}_{get_now_str()}.log")),
            logging.StreamHandler(),
        ],
    )


def get_pretty_json_str(json_obj: Any) -> str:
    return json.dumps(json_obj, indent=4, sort_keys=True, default=_default_numpy)


def print_action_value_fcn_dict(action_value_fcn_dict):
    # TODO change this function to return str instead of printing and make proper changes accordingly
    str_state_action_value_dict = dict()

    for state, action_value_dict in action_value_fcn_dict.items():
        str_state_action_value_dict[str(state)] = convert_keys_to_strs(action_value_dict)

    print(get_pretty_json_str(str_state_action_value_dict))


def print_state_value_fcn_dict(state_value_fcn_dict):
    # TODO change this function to return str instead of printing and make proper changes accordingly
    print(get_pretty_json_str(convert_keys_to_strs(state_value_fcn_dict)))


def convert_keys_to_strs(dict_):
    str_key_dict = dict()

    for key, value in dict_.items():
        str_key_dict[str(key)] = value

    return str_key_dict


def _default_numpy(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, np.generic):
        return val.item()
    raise TypeError("Unknown type: '%s'" % type(val))
