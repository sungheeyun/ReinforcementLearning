from copy import deepcopy

import numpy as np

from policy.utils import are_equivalent_greedy_policies
from tabular_algorithms.model_free_tabular_n_step_control_algorithm_base import \
    ModelFreeTabularNStepControlAlgorithmBase


class NStepSARSAAlgorithm(ModelFreeTabularNStepControlAlgorithmBase):
    """
    n-step SARSA, the on-policy n-step temporal difference (TD) control algorithm.
    """

    def __init__(
        self,
        num_steps,
        gamma,
        learning_rate_fcn,
        behavior_policy,
        default_action_value_fcn_value,
    ):
        super(NStepSARSAAlgorithm, self).__init__(
            gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value
        )

        self.num_steps = num_steps
        self.discount_factor_power_array = None

        self.__initialize()

    def __initialize(self):
        self.discount_factor_power_array = np.power(
            self.gamma, np.arange(self.num_steps + 1)
        )

    def learn(
        self,
        env,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
        max_num_episodes_with_same_greedy_policy,
        does_record_history=False,
        verbose_mode=False,
        debug_mode=False,
    ):

        num_episodes_with_same_greedy_policy = 0
        previous_action_value_fcn_dict = self.action_value_fcn_dict.copy()

        iter_num = 0

        for episode_num in range(max_num_episodes):
            state_list = list()
            action_list = list()
            reward_list = list()

            state, _ = env.reset()

            not_terminated = not env.is_terminal_state()

            action_value_dict = self.action_value_fcn_dict[state]
            available_actions = env.get_all_available_actions(state)
            if len(action_value_dict) < len(available_actions):
                for action_ in available_actions:
                    action_value_dict[action_]

            action = self.get_action(action_value_dict, iter_num, episode_num)

            state_list.append(state)
            action_list.append(action)

            for t in range(max_num_transitions_per_episode):
                if debug_mode:
                    print(f"t = {t}")

                if not_terminated:
                    next_state, reward, is_terminal_state, _ = env.apply_action(action)

                    reward_list.append(reward)

                    if env.is_terminal_state():
                        not_terminated = False
                    else:
                        next_action_value_dict = self.action_value_fcn_dict[next_state]
                        next_available_actions = env.get_all_available_actions(
                            next_state
                        )
                        if len(next_action_value_dict) < len(next_available_actions):
                            for next_action_ in next_available_actions:
                                next_action_value_dict[next_action_]

                        next_action = self.get_action(
                            next_action_value_dict, iter_num, episode_num
                        )

                        state_list.append(next_state)
                        action_list.append(next_action)

                        action = next_action

                    if debug_mode:
                        print(f"state_list: {state_list}")
                        print(f"action_list: {action_list}")
                        print(f"reward_list: {reward_list}")

                if t >= self.num_steps - 1:
                    n_reward_array = np.array(
                        reward_list[t - self.num_steps + 1:t + 1]
                    )

                    if n_reward_array.size == 0:
                        break

                    G = (
                        n_reward_array
                        * self.discount_factor_power_array[: n_reward_array.size]
                    ).sum()

                    tail_state_str = ""
                    if t + 1 < len(state_list):
                        tail_idx = t + 1
                        tail_state = state_list[tail_idx]
                        tail_action = action_list[tail_idx]

                        tail_action_value_dict = self.action_value_fcn_dict[tail_state]
                        tail_action_value_fcn_value = tail_action_value_dict[
                            tail_action
                        ]

                        G += (
                            self.discount_factor_power_array[-1]
                            * tail_action_value_fcn_value
                        )

                        tail_state_str = (
                            f", tail state/action: {str(tail_state)}/{tail_action}"
                        )

                    updated_idx = t - self.num_steps + 1
                    state_updated = state_list[updated_idx]
                    action_updated = action_list[updated_idx]

                    if debug_mode:
                        print(
                            f"\tupdated state/action: {state_updated}/{action_updated}, G: {G}{tail_state_str}"
                        )

                    action_value_dict_updated = self.action_value_fcn_dict[
                        state_updated
                    ]
                    current_action_value_fcn_value = action_value_dict_updated[
                        action_updated
                    ]

                    learning_rate = self.get_learning_rate(iter_num, episode_num)

                    action_value_dict_updated[action_updated] += learning_rate * (
                        G - current_action_value_fcn_value
                    )

                    if debug_mode:
                        print(
                            f"\tprev. val: {current_action_value_fcn_value:.2f}",
                            end=", ",
                        )
                        print(f"G: {G}", end=", ")
                        print(
                            f"updated val: {action_value_dict_updated[action_updated]:.2f}"
                        )

                    iter_num += 1

            if does_record_history:
                self.record_history(episode_num)

            if iter_num > max_num_iters:
                break

            if are_equivalent_greedy_policies(
                previous_action_value_fcn_dict, self.action_value_fcn_dict
            ):
                num_episodes_with_same_greedy_policy += 1
            else:
                num_episodes_with_same_greedy_policy = 0

            if (
                num_episodes_with_same_greedy_policy
                >= max_num_episodes_with_same_greedy_policy
            ):
                break

            previous_action_value_fcn_dict = deepcopy(self.action_value_fcn_dict)
