from copy import deepcopy

from policy.utils import are_equivalent_greedy_policies
from tabular_algorithms.model_free_tabular_one_step_control_algorithm_base import \
    ModelFreeTabularOneStepControlAlgorithmBase


class OneStepSARSAAlgorithm(ModelFreeTabularOneStepControlAlgorithmBase):
    """
    SARSA, the on-policy_sampler temporal difference (TD) control algorithm.
    """

    def __init__(
        self, gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value
    ):
        super(OneStepSARSAAlgorithm, self).__init__(
            gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value
        )

    def learn(
        self,
        env,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
        max_num_episodes_with_same_greedy_policy=float("inf"),
        does_record_history=False,
        verbose_mode=False,
        debug_mode=False,
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

                    next_action = self.get_action(
                        next_action_value_dict, iter_num, episode_num
                    )
                    next_action_value_fcn_value = next_action_value_dict[next_action]

                learning_rate = self.get_learning_rate(iter_num, episode_num)
                action_value_dict[action] += learning_rate * (
                    reward
                    + self.gamma * next_action_value_fcn_value
                    - current_action_value_fcn_value
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
