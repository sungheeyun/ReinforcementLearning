from typing import Dict, Any, List
import logging
import unittest

import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from environment.random_walk_environment import RandomWalkEnvironment
from policy.equally_probable_random_policy_sampler import EquallyProbableRandomPolicySampler
from tabular_algorithms.one_step_temporal_difference_algorithm import OneStepTemporalDifferenceAlgorithm
from utils import get_pretty_json_str, set_logging_basic_config


logger: logging.Logger = logging.getLogger()


class TestOneStepTemporalDifferenceAlgorithm(unittest.TestCase):

    diminishing_step_size: bool = True
    num_nodes: int = 19

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_with_random_walk_environment(self) -> None:

        # RandomState(MT19937(SeedSequence(123456789)))
        numpy.random.seed(760104)

        num_nodes: int = TestOneStepTemporalDifferenceAlgorithm.num_nodes

        random_walk_environment: RandomWalkEnvironment = RandomWalkEnvironment(num_nodes)
        random_policy: EquallyProbableRandomPolicySampler = EquallyProbableRandomPolicySampler(("left", "right"))

        gamma: float = 1.0

        if TestOneStepTemporalDifferenceAlgorithm.diminishing_step_size:
            def learning_rate_strategy(iter_num: int, episode_num: int) -> float:
                return 0.1 / (1.0 + iter_num * 0.001)
        else:
            learning_rate_strategy: float = 0.1

        one_step_temporal_difference_algorithm: OneStepTemporalDifferenceAlgorithm = (
            OneStepTemporalDifferenceAlgorithm(gamma, learning_rate_strategy, 0.5)
        )

        figure: Figure
        axis: Axes
        figure, axis = plt.subplots()

        max_num_episodes: int = 300
        max_num_transitions_per_episode: int = 100
        max_num_iters: int = max_num_episodes * 100

        total_num_episodes: int = 0
        for _ in range(5):
            one_step_temporal_difference_algorithm.predict(
                random_walk_environment,
                random_policy,
                max_num_episodes,
                max_num_iters,
                max_num_transitions_per_episode,
            )

            total_num_episodes += max_num_episodes

            state_value_fcn_dict: Dict[Any, float] = one_step_temporal_difference_algorithm.get_state_value_fcn_dict()

            logger.debug(get_pretty_json_str(state_value_fcn_dict))

            random_walk_environment.draw_state_value_fcn_values(
                axis, state_value_fcn_dict, "o-", label=total_num_episodes
            )

        axis.legend()

        figure.show()

        logger.debug(get_pretty_json_str(one_step_temporal_difference_algorithm.state_value_fcn_dict))

        node: int
        err_list: List[float] = list()
        for node in range(1, num_nodes + 1):
            estimated_state_value_fcn_value: float = one_step_temporal_difference_algorithm.state_value_fcn_dict[node]
            true_state_value_fcn_value: float = float(node) / (num_nodes + 1)

            logger.debug(f"{estimated_state_value_fcn_value} ~ {true_state_value_fcn_value}")

            err: float = estimated_state_value_fcn_value - true_state_value_fcn_value
            logging.debug(err)
            err_list.append(err)

            self.assertAlmostEqual(estimated_state_value_fcn_value, true_state_value_fcn_value, 1)

        logger.debug(err_list)
        max_abs_error: float = np.abs(err_list).max()
        logger.debug(max_abs_error)
        self.assertAlmostEqual(max_abs_error, 0.04792, 5)


if __name__ == "__main__":
    unittest.main()

    if '__file__' in dir():
        plt.show()
