from typing import Dict, Any
import unittest
import os
import json
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from utils import set_logging_basic_config, get_pretty_json_str
from policy.utils import action_sequence_json_obj_to_action_value_fcn_dict
from environment.deterministic_directed_graph_environment import (
    DeterministicDirectedGraphEnvironment,
    read_deterministic_directed_graph_environment_from_json,
)
from policy.epsilon_greedy_policy_sampler import EpsilonGreedyPolicySampler
from tabular_algorithms.one_step_temporal_difference_alg import OneStepTemporalDifferenceAlg
from tabular_algorithms.one_step_temporal_difference_control_als import OneStepQLearningAlg


logger = logging.getLogger()


class TestNeighborhoodMapping(unittest.TestCase):
    test_data_directory = os.path.join(os.curdir, "data")

    simple_deterministic_state_transition_graph_json_file_path = os.path.join(
        test_data_directory, "simple_deterministic_state_transition_graph.json"
    )

    simple_action_sequence_json_file_path = os.path.join(
        test_data_directory, "simple_action_sequence.json"
    )

    large_deterministic_state_transition_graph_json_file_path = os.path.join(
        test_data_directory, "larger_deterministic_state_transition_graph.json"
    )

    large_action_sequence_json_file_path = os.path.join(
        test_data_directory, "large_action_sequence.json"
    )

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__, logging.INFO)

    def test_neighborhood_mapping_with_simple_example(self) -> None:

        deterministic_directed_graph_environment: DeterministicDirectedGraphEnvironment = (
            read_deterministic_directed_graph_environment_from_json(
                TestNeighborhoodMapping.simple_deterministic_state_transition_graph_json_file_path
            )
        )

        logger.debug(f"environment start state: {deterministic_directed_graph_environment.start_state}")

        with open(TestNeighborhoodMapping.simple_action_sequence_json_file_path) as fin:
            action_sequence_json_obj: dict = json.load(fin)

        action_value_fcn_dict_from_action_sequence: Dict[Any, Dict[Any, float]] = (
            action_sequence_json_obj_to_action_value_fcn_dict(action_sequence_json_obj)
        )

        logger.debug(
            get_pretty_json_str(
                deterministic_directed_graph_environment.state_transition_graph_dict
            )
        )
        logger.debug(
            get_pretty_json_str(
                deterministic_directed_graph_environment.state_transition_reward_dict
            )
        )
        logger.debug(get_pretty_json_str(action_sequence_json_obj))
        logger.debug(get_pretty_json_str(action_value_fcn_dict_from_action_sequence))

        gamma: float = 0.9
        learning_rate: float = 0.1
        epsilon: float = 0.1
        default_state_value_fcn_value: float = 0.0

        action_sequence_policy: EpsilonGreedyPolicySampler = EpsilonGreedyPolicySampler(
            epsilon, action_value_fcn_dict_from_action_sequence
        )

        one_step_temporal_difference_alg: OneStepTemporalDifferenceAlg = OneStepTemporalDifferenceAlg(
            gamma, learning_rate, default_state_value_fcn_value
        )

        one_step_temporal_difference_alg.predict(
            deterministic_directed_graph_environment,
            action_sequence_policy,
            100,
            10000,
            100,
            True,
            True,
            True
        )

        figure: Figure
        axis: Axes

        figure, axis = plt.subplots()
        one_step_temporal_difference_alg.plot_value_fcn_history(axis)
        figure.show()

        logger.info(get_pretty_json_str(one_step_temporal_difference_alg.state_value_fcn_dict))

        default_action_value_fcn_value: float = np.array(
            list(one_step_temporal_difference_alg.state_value_fcn_dict.values()), float
        ).mean()

        one_step_q_learning_alg: OneStepQLearningAlg = OneStepQLearningAlg(
            gamma,
            learning_rate,
            0.0,
            default_action_value_fcn_value
        )

        one_step_q_learning_alg.learn(
            deterministic_directed_graph_environment,
            100,
            10000,
            100,
            10,
            True,
            False,
            False
        )

        figure, axis = plt.subplots()
        one_step_q_learning_alg.plot_value_fcn_history(axis)
        figure.show()

        logger.info(get_pretty_json_str(one_step_q_learning_alg.action_value_fcn_dict))

        self.assertTrue(True)

    def test_neighborhood_mapping_with_large_example(self) -> None:

        deterministic_directed_graph_environment: DeterministicDirectedGraphEnvironment = (
            read_deterministic_directed_graph_environment_from_json(
                TestNeighborhoodMapping.large_deterministic_state_transition_graph_json_file_path
            )
        )

        logger.debug(f"environment start state: {deterministic_directed_graph_environment.start_state}")

        with open(TestNeighborhoodMapping.large_action_sequence_json_file_path) as fin:
            action_sequence_json_obj: dict = json.load(fin)

        action_value_fcn_dict_from_action_sequence: Dict[Any, Dict[Any, float]] = (
            action_sequence_json_obj_to_action_value_fcn_dict(action_sequence_json_obj)
        )

        logger.debug(
            get_pretty_json_str(
                deterministic_directed_graph_environment.state_transition_graph_dict
            )
        )
        logger.debug(
            get_pretty_json_str(
                deterministic_directed_graph_environment.state_transition_reward_dict
            )
        )
        logger.debug(get_pretty_json_str(action_sequence_json_obj))
        logger.debug(get_pretty_json_str(action_value_fcn_dict_from_action_sequence))

        gamma: float = 0.9
        learning_rate: float = 0.1
        epsilon: float = 0.1
        default_state_value_fcn_value: float = 0.0

        max_num_episodes: int = 200

        action_sequence_policy: EpsilonGreedyPolicySampler = EpsilonGreedyPolicySampler(
            epsilon, action_value_fcn_dict_from_action_sequence
        )

        one_step_temporal_difference_alg: OneStepTemporalDifferenceAlg = OneStepTemporalDifferenceAlg(
            gamma, learning_rate, default_state_value_fcn_value
        )

        one_step_temporal_difference_alg.predict(
            deterministic_directed_graph_environment,
            action_sequence_policy,
            max_num_episodes,
            10000,
            100,
            True,
            True,
            True
        )

        figure: Figure
        axis: Axes

        figure, axis = plt.subplots()
        one_step_temporal_difference_alg.plot_value_fcn_history(axis)
        figure.show()

        logger.info(get_pretty_json_str(one_step_temporal_difference_alg.state_value_fcn_dict))

        default_action_value_fcn_value: float = np.array(
            list(one_step_temporal_difference_alg.state_value_fcn_dict.values()), float
        ).mean()

        one_step_q_learning_alg: OneStepQLearningAlg = OneStepQLearningAlg(
            gamma,
            learning_rate,
            0.0,
            default_action_value_fcn_value
        )

        one_step_q_learning_alg.learn(
            deterministic_directed_graph_environment,
            max_num_episodes,
            10000,
            100,
            10,
            True,
            False,
            False
        )

        figure, axis = plt.subplots()
        one_step_q_learning_alg.plot_value_fcn_history(axis)
        figure.show()

        logger.info(get_pretty_json_str(one_step_q_learning_alg.action_value_fcn_dict))

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
