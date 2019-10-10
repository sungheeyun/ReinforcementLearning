import logging

from numpy import random as nr

from policy.policy_sampler_base import PolicySamplerBase


logger = logging.getLogger()


class EpsilonGreedyPolicySampler(PolicySamplerBase):
    """
    ProbabilisticPolicy sampler sampling actions following epsilon-greedy policy
    """

    @staticmethod
    def get_action_from_state_action_dict(state_action_dict, epsilon):
        if nr.rand() < epsilon:
            return nr.choice(list(state_action_dict))
        else:
            return max(state_action_dict, key=state_action_dict.get)

    def __init__(self, epsilon, action_value_fcn_dict):
        self.epsilon = epsilon
        self.action_value_fcn_dict = action_value_fcn_dict

    def get_action(self, state):
        logger.debug(f"self.action_value_fcn_dict[{state}]: {self.action_value_fcn_dict[state]}")
        return EpsilonGreedyPolicySampler.get_action_from_state_action_dict(
            self.action_value_fcn_dict[state], self.epsilon
        )

    def get_all_states(self):
        return list(self.action_value_fcn_dict.keys())
