import unittest
import logging

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import state_value_fcn_dict_to_str, action_value_fcn_dict_to_pretty_str, set_logging_basic_config
from environment.grid_world_environment import GridWorld
from environment.grid_world_with_cliff_environment import GridWorldWithCliff
from policy.probabilistic_policy import ProbabilisticPolicy
from policy.epsilon_greedy_policy_sampler import EpsilonGreedyPolicySampler
from tabular_algorithms.n_step_q_learning_algorithm import NStepQLearningAlgorithm
from tabular_algorithms.n_step_sarsa_algorithm import NStepSARSAAlgorithm


logger: logging.Logger = logging.getLogger()
Axes3D


class TestTabularNStepControlAlgorithm(unittest.TestCase):

    record_history: bool = True
    debug_mode: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_n_step_q_learning_algorithm(self):
        grid_width, grid_height = 10, 10

        # test = 'grid_world'
        test = "windy_grid"
        # test = "grid_world_cliff"
        reducing_learning_rate = False
        # alg = "sarsa"
        alg = "qlearning"
        deterministic_policy_for_prediction = True

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
                return 0.1 / (1.0 + episode_num * 0.1)

        else:
            learning_rate_strategy = 0.1

        num_steps = 24
        # max_num_epidoes = 1000
        max_num_epidoes = 100
        max_num_transitions_per_episode = 100
        max_num_iters = max_num_epidoes * max_num_transitions_per_episode
        max_num_episodes_with_same_greedy_policy = 20
        gamma = 1.0
        default_action_value_fcn_value = 0.0
        epsilon = 0.5

        if alg == "qlearning":
            td_control_alg = NStepQLearningAlgorithm(
                gamma, learning_rate_strategy, epsilon, default_action_value_fcn_value
            )
        elif alg == "sarsa":
            td_control_alg = NStepSARSAAlgorithm(
                num_steps,
                gamma,
                learning_rate_strategy,
                epsilon,
                default_action_value_fcn_value,
            )
        else:
            raise ValueError(alg)

        td_control_alg.learn(
            env,
            max_num_epidoes,
            max_num_iters,
            max_num_transitions_per_episode,
            max_num_episodes_with_same_greedy_policy,
            record_history=TestTabularNStepControlAlgorithm.record_history,
            debug_mode=TestTabularNStepControlAlgorithm.debug_mode
        )

        logger.debug(action_value_fcn_dict_to_pretty_str(td_control_alg.get_action_value_fcn_dict()))

        from matplotlib import pyplot as plt

        if TestTabularNStepControlAlgorithm.record_history:
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

        # value function prediction

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
            record_history=TestTabularNStepControlAlgorithm.record_history,
        )

        if TestTabularNStepControlAlgorithm.record_history:
            fig, ax = plt.subplots()
            td0.plot_value_fcn_history(ax)
            fig.show()

        logger.debug(state_value_fcn_dict_to_str(td0.get_state_value_fcn_dict()))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca(projection="3d")

        env.draw_3d_state_value_fcn_values(ax, td0.get_state_value_fcn_dict())
        fig.show()

        fig, ax = plt.subplots()
        env.draw_state_value_fcn_values(ax, td0.get_state_value_fcn_dict())
        fig.show()


if __name__ == '__main__':
    unittest.main()

    if "__file__" in dir():
        plt.show()
