""" Classes for (tabular) temporal differencing learning with lambda = 0, i.e., TD(0) algorithm. """

from copy import deepcopy

from policy.policies import are_equivalent_greedy_policies
from tabular_algorithms.tabular_rl_algorithms_base import ModelFreeTabularOneStepControlAlgBase


class OneStepSARSAAlg(ModelFreeTabularOneStepControlAlgBase):
    """
    SARSA, the on-policy_sampler temporal difference (TD) control algorithm.
    """

    def __init__(self, gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value):
        super(OneStepSARSAAlg, self).__init__(gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value)

    def learn(
        self,
        env,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
        max_num_episodes_with_same_greedy_policy=float('inf'),
        does_record_history=False,
        verbose_mode=False,
        debug_mode=False
    ):

        num_episodes_with_same_greedy_policy = 0
        previous_action_value_fcn_dict = self.action_value_fcn_dict.copy()

        iter_num = 0

        for episode_num in range(max_num_episodes):
            state, _ = env.reset()

            action_value_dict = self.action_value_fcn_dict[state]
            available_actions = env.get_all_available_actions(state)
            if len(action_value_dict) < len(available_actions):
                for action_ in available_actions:
                    action_value_dict[action_]

            action = self.get_action(action_value_dict, iter_num, episode_num)

            for tran_num in range(max_num_transitions_per_episode):
                current_action_value_fcn_value = action_value_dict[action]

                next_state, reward, is_terminal_state, _ = env.apply_action(action)

                if is_terminal_state:
                    next_action_value_fcn_value = 0.0
                else:
                    # state value function for a terminal state is zero by definition.
                    next_action_value_dict = self.action_value_fcn_dict[next_state]
                    next_available_actions = env.get_all_available_actions(next_state)
                    if len(next_action_value_dict) < len(next_available_actions):
                        for next_action_ in next_available_actions:
                            next_action_value_dict[next_action_]

                    next_action = self.get_action(next_action_value_dict, iter_num, episode_num)
                    next_action_value_fcn_value = next_action_value_dict[next_action]

                learning_rate = self.get_learning_rate(iter_num, episode_num)
                action_value_dict[action] += learning_rate * (
                    reward + self.gamma * next_action_value_fcn_value - current_action_value_fcn_value
                )

                if is_terminal_state:
                    break

                iter_num += 1

                if iter_num > max_num_iters:
                    break

                action_value_dict = next_action_value_dict
                action = next_action

            if does_record_history:
                self.record_history(episode_num)

            if iter_num > max_num_iters:
                break

            if are_equivalent_greedy_policies(previous_action_value_fcn_dict, self.action_value_fcn_dict):
                num_episodes_with_same_greedy_policy += 1
            else:
                num_episodes_with_same_greedy_policy = 0

            if num_episodes_with_same_greedy_policy >= max_num_episodes_with_same_greedy_policy:
                break

            previous_action_value_fcn_dict = deepcopy(self.action_value_fcn_dict)


class OneStepQLearningAlg(ModelFreeTabularOneStepControlAlgBase):
    """
    Q-learning, the off-policy_sampler temporal difference (TD) control algorithm.
    """

    def __init__(self, gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value):
        super(OneStepQLearningAlg, self).__init__(gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value)

    def learn(
        self,
        env,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
        max_num_episodes_with_same_greedy_policy=float('inf'),
        does_record_history=False,
        verbose_mode=False,
        debug_mode=False
    ):

        iter_num = 0

        for episode_num in range(max_num_episodes):
            state, _ = env.reset()

            state_q_dict = self.action_value_fcn_dict[state]
            available_actions = env.get_all_available_actions(state)
            if len(state_q_dict) < len(available_actions):
                for action_ in available_actions:
                    state_q_dict[action_]

            if verbose_mode:
                print(f"FROM {state}")
                qs_str = f"Q[{state}]"

            for tran_num in range(max_num_transitions_per_episode):

                action = self.get_action(state_q_dict, iter_num, episode_num)
                if verbose_mode:
                    print(f"take {action}")
                current_action_value_fcn_value = state_q_dict[action]

                if verbose_mode:
                    qsa_str = f"{qs_str}[{action}]"
                    print(f"{qsa_str} = {current_action_value_fcn_value}")

                next_state, reward, is_terminal_state, _ = env.apply_action(action)

                if verbose_mode:
                    print(f"next state: {next_state} with reward: {reward}")

                if is_terminal_state:
                    # all action value functions for a terminal state are zero by definition.
                    max_next_action_value_fcn_value = 0.0
                else:
                    next_state_q_dict = self.action_value_fcn_dict[next_state]
                    next_available_actions = env.get_all_available_actions(next_state)
                    if len(next_state_q_dict) < len(next_available_actions):
                        for next_action_ in next_available_actions:
                            next_state_q_dict[next_action_]

                    max_next_action_value_fcn_value = max(next_state_q_dict.values())

                if verbose_mode:
                    print(f"max(Q[{next_state}]) = {max_next_action_value_fcn_value}")

                learning_rate = self.get_learning_rate(iter_num, episode_num)
                state_q_dict[action] += learning_rate * (
                    reward + self.gamma * max_next_action_value_fcn_value - current_action_value_fcn_value
                )

                if verbose_mode:
                    print(f"--> {qsa_str} = {state_q_dict[action]}")

                if is_terminal_state:
                    break

                iter_num += 1

                if iter_num > max_num_iters:
                    break

                if verbose_mode:
                    print(f"FROM {next_state}")
                    qs_str = f"Q[{next_state}]"

                state_q_dict = next_state_q_dict

            if does_record_history:
                self.record_history(episode_num)

            if verbose_mode:
                print(f"tran_num: {tran_num}")
                print("---------------")

            if iter_num > max_num_iters:
                break


if __name__ == "__main__":

    from rl_utils.utils import print_action_value_fcn_dict, print_state_value_fcn_dict
    from environment.environments import GridWorld, GridWorldWithCliff
    from policy.policies import Policy, EpsilonGreedyPolicySampler

    grid_width, grid_height = 10, 10

    # test = 'grid_world'
    # test = "windy_grid"
    test = "grid_world_cliff"
    reducing_learning_rate = False
    alg = 'sarsa'
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
        td_control_alg = OneStepQLearningAlg(gamma, learning_rate_strategy, epsilon, default_action_value_fcn_value)
    elif alg == "sarsa":
        td_control_alg = OneStepSARSAAlg(gamma, learning_rate_strategy, epsilon, default_action_value_fcn_value)
    else:
        raise ValueError(alg)

    td_control_alg.learn(
        env,
        max_num_epidoes,
        max_num_iters,
        max_num_transitions_per_episode,
        max_num_episodes_with_same_greedy_policy,
        does_record_history=does_record_history
    )
    print_action_value_fcn_dict(td_control_alg.get_action_value_fcn_dict())

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if does_record_history:
        fig, ax = plt.subplots()
        td_control_alg.plot_value_fcn_history(ax)
        fig.show()

    fig, ax = plt.subplots()
    env.draw_boltzmann_actions(ax, td_control_alg.get_action_value_fcn_dict())
    fig.show()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection="3d")

    env.draw_3d_deterministic_action_value_fcn_values(ax, td_control_alg.get_action_value_fcn_dict())
    fig.show()

    fig, ax = plt.subplots()
    env.draw_deterministic_actions_value_fcn_values(ax, td_control_alg.get_action_value_fcn_dict())
    fig.show()

    if deterministic_policy_for_prediction:
        optimal_policy = Policy.get_deterministic_policy_from_action_value_fcn(
            td_control_alg.get_action_value_fcn_dict()
        )
    else:
        optimal_policy = EpsilonGreedyPolicySampler(epsilon, td_control_alg.get_action_value_fcn_dict())

    from tabular_algorithms.one_step_temporal_difference_alg import OneStepTemporalDifferenceAlg

    td0 = OneStepTemporalDifferenceAlg(gamma, learning_rate_strategy, default_action_value_fcn_value)
    td0.predict(
        env,
        optimal_policy,
        max_num_epidoes,
        max_num_iters,
        max_num_transitions_per_episode,
        does_record_history=does_record_history,
    )

    if does_record_history:
        fig, ax = plt.subplots()
        td0.plot_value_fcn_history(ax)
        fig.show()

    print_state_value_fcn_dict(td0.get_state_value_fcn_dict())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection="3d")

    env.draw_3d_state_value_fcn_values(ax, td0.get_state_value_fcn_dict())
    fig.show()

    fig, ax = plt.subplots()
    env.draw_state_value_fcn_values(ax, td0.get_state_value_fcn_dict())
    fig.show()

    print_state_value_fcn_dict(td0.get_state_value_fcn_dict())

    if "__file__" in dir():
        plt.show()
