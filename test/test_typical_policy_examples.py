from typing import Dict, Union
import unittest
from collections import defaultdict
import logging

from utils import set_logging_basic_config, get_pretty_json_str, action_value_fcn_dict_to_pretty_str
from policy.probabilistic_policy import ProbabilisticPolicy
from policy.epsilon_greedy_policy_sampler import EpsilonGreedyPolicySampler


logger = logging.getLogger()


class TestTypicalPolicyExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_policy_get_deterministic_policy_from_action_value_fcn(self):

        action_value_fcn_dict = dict()
        action_value_fcn_dict[0] = dict(a=1, b=3)
        action_value_fcn_dict[1] = dict(a=3, b=1)

        logger.info(action_value_fcn_dict_to_pretty_str(action_value_fcn_dict))

        deterministic_policy = ProbabilisticPolicy.get_deterministic_policy_from_action_value_fcn(
            action_value_fcn_dict
        )

        logger.info(
            action_value_fcn_dict_to_pretty_str(deterministic_policy.get_state_action_probability_dict_dict())
        )

        self.assertEqual(deterministic_policy.get_action(0), "b")
        self.assertEqual(deterministic_policy.get_action(1), "a")

    def test_policy_with_simple_example(self):
        # test ProbabilisticPolicy

        state_action_probability_dict_dict = dict()

        state_action_probability_dict_dict[0] = dict(a=0.3, b=0.7)
        state_action_probability_dict_dict[1] = dict(a=0.8, b=0.2)

        policy = ProbabilisticPolicy(state_action_probability_dict_dict)

        logger.info(get_pretty_json_str(state_action_probability_dict_dict))

        N = 10000

        empirical_state_action_probability_dict_dict: Dict[int, Dict[str, float]] = dict()
        for state in state_action_probability_dict_dict:
            action_probability_dict = defaultdict(int)
            for _ in range(N):
                action_probability_dict[policy.get_action(state)] += 1

            for action in action_probability_dict:
                action_probability_dict[action] /= N

            empirical_state_action_probability_dict_dict[state] = action_probability_dict

        logger.info(get_pretty_json_str(empirical_state_action_probability_dict_dict))

        state: str
        action: str
        action_probability_dict: Dict[str, float]

        for state, action_probability_dict in empirical_state_action_probability_dict_dict.items():
            for action, probability in action_probability_dict.items():
                self.assertAlmostEqual(
                    state_action_probability_dict_dict[state][action],
                    probability,
                    1
                )

    def test_epsilon_greedy_policy_sampler(self):

        epsilon: float = 0.1
        action_value_fcn_dict: Dict[int, Dict[str, Union[float, int]]] = dict()
        action_value_fcn_dict[0] = dict(a=1, b=3, c=-1)
        action_value_fcn_dict[1] = dict(a=3, b=1)

        logger.info(get_pretty_json_str(action_value_fcn_dict))

        epsilon_greedy_policy_sampler = EpsilonGreedyPolicySampler(epsilon, action_value_fcn_dict)

        N = 10000
        empirical_action_value_fcn_dict: Dict[int, Dict[str, float]] = dict()
        for state in epsilon_greedy_policy_sampler.get_all_states():
            logger.info(f"state: {state}")

            action_value_fcn_for_one_state: Dict[str, float] = defaultdict(int)
            for _ in range(N):
                action_value_fcn_for_one_state[epsilon_greedy_policy_sampler.get_action(state)] += 1.0

            for action in action_value_fcn_for_one_state:
                action_value_fcn_for_one_state[action] /= N

            empirical_action_value_fcn_dict[state] = action_value_fcn_for_one_state

        logger.info(get_pretty_json_str(empirical_action_value_fcn_dict))


if __name__ == "__main__":
    unittest.main()
