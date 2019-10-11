import unittest
import logging

from mpl_toolkits.mplot3d import Axes3D

from utils import action_value_fcn_dict_to_pretty_str, state_value_fcn_dict_to_str
from environment.grid_world_environment import GridWorld
from environment.grid_world_with_cliff_environment import GridWorldWithCliff
from policy.probabilistic_policy import ProbabilisticPolicy
from policy.epsilon_greedy_policy_sampler import EpsilonGreedyPolicySampler
from tabular_algorithms.one_step_q_learning_algorithm import OneStepQLearningAlgorithm
from tabular_algorithms.one_step_sarsa_algorithm import OneStepSARSAAlgorithm


logger: logging.Logger = logging.getLogger()
Axes3D


class TestOneStepTemporalDifferenceControlAlgorithms(unittest.TestCase):
    def test_one_step_sarsa_algorithm(self):

        grid_width, grid_height = 10, 10

        # test = 'grid_world'
        # test = "windy_grid"
        test = "grid_world_cliff"
        reducing_learning_rate = False
        alg = "sarsa"
        # alg = "qlearning"
        deterministic_policy_for_prediction = False

        if test == "grid_world":
            # env = GridWorld(grid_width, grid_height, 0.0, -1.0, -10.0)
            env = GridWorld(grid_width, grid_height)
        elif test == "windy_grid":
            env = GridWorld(10, 7, upward_wind_list=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
            env.set_start_state((0, 3))
            env.set_terminal_states([(7, 3)])
        elif test == "grid_world_cliff":
            env = GridWorldWithCliff(10, 4, 0.0, -1.0, -10.0)
        else:
            assert False, test

        if reducing_learning_rate:

            def learning_rate_strategy(iter_num, episode_num):
                return 0.1 / (1.0 + episode_num * 0.0001)

        else:
            learning_rate_strategy = 0.1

        max_num_epidoes = 1000
        max_num_transitions_per_episode = 100
        max_num_iters = max_num_epidoes * max_num_transitions_per_episode
        max_num_episodes_with_same_greedy_policy = 10
        gamma = 1.0
        default_action_value_fcn_value = 0.0
        epsilon = 0.5
        does_record_history = True

        if alg == "qlearning":
            td_control_alg = OneStepQLearningAlgorithm(
                gamma, learning_rate_strategy, epsilon, default_action_value_fcn_value
            )
        elif alg == "sarsa":
            td_control_alg = OneStepSARSAAlgorithm(
                gamma, learning_rate_strategy, epsilon, default_action_value_fcn_value
            )
        else:
            raise ValueError(alg)

        td_control_alg.learn(
            env,
            max_num_epidoes,
            max_num_iters,
            max_num_transitions_per_episode,
            max_num_episodes_with_same_greedy_policy,
            does_record_history=does_record_history,
        )

        logger.info(action_value_fcn_dict_to_pretty_str(td_control_alg.get_action_value_fcn_dict()))

        from matplotlib import pyplot as plt

        if does_record_history:
            fig, ax = plt.subplots()
            td_control_alg.plot_value_fcn_history(ax)
            fig.show()

        fig, ax = plt.subplots()
        env.draw_boltzmann_actions(ax, td_control_alg.get_action_value_fcn_dict())
        fig.show()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca(projection="3d")

        env.draw_3d_deterministic_action_value_fcn_values(
            ax, td_control_alg.get_action_value_fcn_dict()
        )
        fig.show()

        fig, ax = plt.subplots()
        env.draw_deterministic_actions_value_fcn_values(
            ax, td_control_alg.get_action_value_fcn_dict()
        )
        fig.show()

        if deterministic_policy_for_prediction:
            optimal_policy = ProbabilisticPolicy.get_deterministic_policy_from_action_value_fcn(
                td_control_alg.get_action_value_fcn_dict()
            )
        else:
            optimal_policy = EpsilonGreedyPolicySampler(
                epsilon, td_control_alg.get_action_value_fcn_dict()
            )

        from tabular_algorithms.one_step_temporal_difference_algorithm import (
            OneStepTemporalDifferenceAlgorithm,
        )

        td0 = OneStepTemporalDifferenceAlgorithm(
            gamma, learning_rate_strategy, default_action_value_fcn_value
        )
        td0.predict(
            env,
            optimal_policy,
            max_num_epidoes,
            max_num_iters,
            max_num_transitions_per_episode,
            record_history=does_record_history,
        )

        if does_record_history:
            fig, ax = plt.subplots()
            td0.plot_value_fcn_history(ax)
            fig.show()

        logger.info(state_value_fcn_dict_to_str(td0.get_state_value_fcn_dict()))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca(projection="3d")

        env.draw_3d_state_value_fcn_values(ax, td0.get_state_value_fcn_dict())
        fig.show()

        fig, ax = plt.subplots()
        env.draw_state_value_fcn_values(ax, td0.get_state_value_fcn_dict())
        fig.show()

        logger.info(state_value_fcn_dict_to_str(td0.get_state_value_fcn_dict()))

        if "__file__" in dir():
            plt.show()


if __name__ == '__main__':
    unittest.main()
