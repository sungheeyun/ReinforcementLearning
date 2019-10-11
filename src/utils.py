from typing import Any, Dict
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


def action_value_fcn_dict_to_pretty_str(action_value_fcn_dict: Dict[Any, Dict[Any, float]]) -> str:
    str_state_action_value_dict: Dict[str, Dict[str, float]] = dict()

    for state, action_value_dict in action_value_fcn_dict.items():
        str_state_action_value_dict[str(state)] = convert_keys_to_strs(action_value_dict)

    return get_pretty_json_str(str_state_action_value_dict)


def state_value_fcn_dict_to_str(state_value_fcn_dict: Dict[Any, float]) -> str:
    return get_pretty_json_str(convert_keys_to_strs(state_value_fcn_dict))


def convert_keys_to_strs(dict_: Dict[Any, float]) -> Dict[str, float]:
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
