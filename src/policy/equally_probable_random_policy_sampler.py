from numpy import random as nr

from policy.policy_sampler_base import PolicySamplerBase


class EquallyProbableRandomPolicySampler(PolicySamplerBase):
    """
    The random_policy that chooses an action among given set of actions with the same probability for each action.

    """

    def __init__(self, values):
        self.values = values

    def get_action(self, state):
        return nr.choice(self.values)

    def get_all_states(self):
        return None
