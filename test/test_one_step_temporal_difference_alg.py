import unittest

from environment.environments import RandomWalkEnvironment
from policy.equally_probable_random_policy_sampler import EquallyProbableRandomPolicySampler
from policy.probabilistic_policy_base import ProbabilisticPolicyBase
from utils import get_pretty_json_str


class TestOneStepTemporalDifferenceAlgorithm(unittest.TestCase):
    def test_with_random_walk_environment(self):
        random_walk_environment: RandomWalkEnvironment = RandomWalkEnvironment(19)
        random_policy: Policy = EquallyProbableRandomPolicySampler(("left", "right"))

        gamma = 1.0
        diminishing_step_size = False

    if diminishing_step_size:

        def learning_rate_strategy(iter_num, episode_num):
            return 0.1 / (1.0 + iter_num * 0.01)

    else:
        learning_rate_strategy = 0.1

    # td0 = TemporalDifference0Alg(1.0, 0.1, 0.5)
    td0 = OneStepTemporalDifferenceAlg(gamma, learning_rate_strategy, 0.5)

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    max_num_episodes = 10
    max_num_transitions_per_episode = 100

    total_num_episodes = 0
    for _ in range(5):
        td0.predict(
            random_walk_environment,
            random_policy,
            max_num_episodes,
            max_num_episodes * 100,
            max_num_transitions_per_episode,
            )

        total_num_episodes += max_num_episodes

        state_value_fcn_dict = td0.get_state_value_fcn_dict()
        print(get_pretty_json_str(state_value_fcn_dict))
        random_walk_environment.draw_state_value_fcn_values(ax, state_value_fcn_dict, "o-", label=total_num_episodes)

    ax.legend()

    fig.show()



if __name__ == '__main__':
    unittest.main()
