""" Classes for example policies for Markov decision process """

from collections import defaultdict

from policy.probabilistic_policy_base import ProbabilisticPolicyBase


class ProbabilisticPolicy(ProbabilisticPolicyBase):
    @staticmethod
    def get_deterministic_policy_from_action_value_fcn(action_value_fcn_dict):
        state_action_probability_dict_dict = defaultdict(dict)

        for state, action_value_dict in action_value_fcn_dict.items():
            state_action_probability_dict_dict[state][max(action_value_dict, key=action_value_dict.get)] = 1.0

        return ProbabilisticPolicy(state_action_probability_dict_dict)

    def __init__(self, state_action_probability_dict_dict):
        self.state_action_probability_dict_dict = state_action_probability_dict_dict

    def get_state_action_probability_dict_dict(self):
        return self.state_action_probability_dict_dict

    def get_action_probability_dict(self, state):
        return self.state_action_probability_dict_dict[state]
