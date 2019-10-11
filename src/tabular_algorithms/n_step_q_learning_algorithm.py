from tabular_algorithms.model_free_tabular_n_step_control_algorithm_base import \
    ModelFreeTabularNStepControlAlgorithmBase


class NStepQLearningAlgorithm(ModelFreeTabularNStepControlAlgorithmBase):
    """
    Q-learning, the off-policy_sampler temporal difference (TD) control algorithm.
    """

    def __init__(
        self, gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value
    ):
        super(NStepQLearningAlgorithm, self).__init__(
            gamma, learning_rate_fcn, behavior_policy, default_action_value_fcn_value
        )

    def learn(
        self,
        env,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
        max_num_episodes_with_same_greedy_policy,
        record_history=False,
        verbose_mode=False,
        debug_mode=False,
    ):

        iter_num = 0

        for episode_num in range(max_num_episodes):
            state, _ = env.reset()

            state_q_dict = self.action_value_fcn_dict[state]
            if len(state_q_dict) < len(env.get_all_available_actions()):
                for action_ in env.get_all_available_actions():
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
                    if len(next_state_q_dict) < len(env.get_all_available_actions()):
                        for next_action_ in env.get_all_available_actions():
                            next_state_q_dict[next_action_]

                    max_next_action_value_fcn_value = max(next_state_q_dict.values())

                if verbose_mode:
                    print(f"max(Q[{next_state}]) = {max_next_action_value_fcn_value}")

                learning_rate = self.get_learning_rate(iter_num, episode_num)
                state_q_dict[action] += learning_rate * (
                    reward
                    + self.gamma * max_next_action_value_fcn_value
                    - current_action_value_fcn_value
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

            if record_history:
                self.record_history(episode_num)

            if verbose_mode:
                print(f"tran_num: {tran_num}")
                print("---------------")

            if iter_num > max_num_iters:
                break
