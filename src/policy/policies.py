""" Classes for example policies for Markov decision process """

from collections import defaultdict

import numpy.random as nr

from policy.policy_base import PolicyBase, PolicySamplerBase


def get_greedy_state_action_pair_set(action_value_fcn_dict):
    greedy_state_action_pair_set = set()

    for state, action_value_dict in action_value_fcn_dict.items():
        max_action = max(action_value_dict, key=action_value_dict.get)
        greedy_state_action_pair_set.add((state, max_action))

    return greedy_state_action_pair_set

def are_equivalent_greedy_policies(action_value_fcn_dict_1, action_value_fcn_dict_2):
    greedy_state_action_pair_set_1 = get_greedy_state_action_pair_set(action_value_fcn_dict_1)
    greedy_state_action_pair_set_2 = get_greedy_state_action_pair_set(action_value_fcn_dict_2)

    return greedy_state_action_pair_set_1 == greedy_state_action_pair_set_2

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


class EpsilonGreedyPolicySampler(PolicySamplerBase):
    """
    Policy sampler sampling actions following epsilon-greedy policy
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
        return EpsilonGreedyPolicySampler.get_action_from_state_action_dict(
            self.action_value_fcn_dict[state], self.epsilon
        )
        # if nr.rand() < self.epsilon:
        #    return nr.choice(list(self.action_value_fcn_dict[state]))
        # else:
        #    return max(self.action_value_fcn_dict[state], key=self.action_value_fcn_dict[state].get)

    def get_all_states(self):
        return list(self.action_value_fcn_dict.keys())


class Policy(PolicyBase):
    @staticmethod
    def get_deterministic_policy_from_action_value_fcn(action_value_fcn_dict):
        state_action_probability_dict_dict = defaultdict(dict)

        for state, action_value_dict in action_value_fcn_dict.items():
            state_action_probability_dict_dict[state][max(action_value_dict, key=action_value_dict.get)] = 1.0

        return Policy(state_action_probability_dict_dict)

    def __init__(self, state_action_probability_dict_dict):
        self.state_action_probability_dict_dict = state_action_probability_dict_dict

    def get_state_action_probability_dict_dict(self):
        return self.state_action_probability_dict_dict

    def get_action_probability_dict(self, state):
        return self.state_action_probability_dict_dict[state]


if __name__ == "__main__":

    from rl_utils.utils import print_action_value_fcn_dict, print_state_value_fcn_dict

    # test Policy.get_deterministic_policy_from_action_value_fcn

    action_value_fcn_dict = dict()
    action_value_fcn_dict[0] = dict(a=1, b=3)
    action_value_fcn_dict[1] = dict(a=3, b=1)

    print_action_value_fcn_dict(action_value_fcn_dict)

    determ_policy = Policy.get_deterministic_policy_from_action_value_fcn(action_value_fcn_dict)

    print_action_value_fcn_dict(determ_policy.get_state_action_probability_dict_dict())

    for _ in range(10):
        print(determ_policy.get_action(0), determ_policy.get_action(1))

    # test Policy

    state_action_probability_dict_dict = dict()

    state_action_probability_dict_dict[0] = dict(a=0.3, b=0.7)
    state_action_probability_dict_dict[1] = dict(a=0.8, b=0.2)

    policy = Policy(state_action_probability_dict_dict)

    for state in state_action_probability_dict_dict:
        print(f"state: {state}")
        print(f"\taction_probability_dist: {policy.get_action_probability_dict(state)}")

    N = 1000
    for state in state_action_probability_dict_dict:
        print(f"state: {state}")
        d = defaultdict(int)
        for _ in range(N):
            d[policy.get_action(state)] += 1

        for action in d:
            d[action] /= N

        print_state_value_fcn_dict(d)

    # test EpsilonGreedyPolicySampler

    epsilon = 0.1
    action_value_fcn_dict = dict()
    action_value_fcn_dict[0] = dict(a=1, b=3, c=-1)
    action_value_fcn_dict[1] = dict(a=3, b=1)

    print_action_value_fcn_dict(action_value_fcn_dict)

    epsilon_greedy_policy_sampler = EpsilonGreedyPolicySampler(epsilon, action_value_fcn_dict)

    N = 100000
    for state in epsilon_greedy_policy_sampler.get_all_states():
        print(f"state: {state}")

        d = defaultdict(int)
        for _ in range(N):
            d[epsilon_greedy_policy_sampler.get_action(state)] += 1

        for action in d:
            d[action] /= N

        print_state_value_fcn_dict(d)
