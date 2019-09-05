import json

import numpy as np


def get_pretty_json_str(json_dict):
    return json.dumps(json_dict, indent=4, sort_keys=True, default=_default_numpy)


def print_action_value_fcn_dict(action_value_fcn_dict):
    str_state_action_value_dict = dict()

    for state, action_value_dict in action_value_fcn_dict.items():
        str_state_action_value_dict[str(state)] = convert_keys_to_strs(action_value_dict)

    print(get_pretty_json_str(str_state_action_value_dict))


def print_state_value_fcn_dict(state_value_fcn_dict):
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
