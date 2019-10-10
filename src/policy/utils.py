from typing import Dict, Any, List


def action_sequence_json_obj_to_action_value_fcn_dict(
        action_sequence_json_obj: dict,
        default_action_value_fcn_value: float = 1.0
) -> Dict[Any, Dict[Any, float]]:
    action_sequence: List[Dict[Any, Any]] = action_sequence_json_obj["ActionSequence"]

    action_value_fcn_dict: Dict[Any, Dict[Any, float]] = dict()

    action_state_dict: dict
    current_state: Any = action_sequence_json_obj["InitialState"]
    for action_state_dict in action_sequence:
        action = action_state_dict["Action"]
        action_value_fcn_dict[current_state] = dict([(action, default_action_value_fcn_value)])

        current_state = action_state_dict["NextState"]

    return action_value_fcn_dict


def are_equivalent_greedy_policies(action_value_fcn_dict_1, action_value_fcn_dict_2):
    greedy_state_action_pair_set_1 = get_greedy_state_action_pair_set(
        action_value_fcn_dict_1
    )
    greedy_state_action_pair_set_2 = get_greedy_state_action_pair_set(
        action_value_fcn_dict_2
    )

    return greedy_state_action_pair_set_1 == greedy_state_action_pair_set_2


def get_greedy_state_action_pair_set(action_value_fcn_dict):
    greedy_state_action_pair_set = set()

    for state, action_value_dict in action_value_fcn_dict.items():
        max_action = max(action_value_dict, key=action_value_dict.get)
        greedy_state_action_pair_set.add((state, max_action))

    return greedy_state_action_pair_set
