""" Classes for tabular temporal differencing prediction methods """

import logging

from tabular_algorithms.tabular_rl_algorithms_base import (
    ModelFreeTabularPredictionAlgBase,
)


logger = logging.getLogger()


class OneStepTemporalDifferenceAlg(ModelFreeTabularPredictionAlgBase):
    """
    Implements temporal difference prediction algorithms.
    """

    def __init__(self, gamma, learning_rate_fcn, defaulit_state_value_fcn_value):
        super(OneStepTemporalDifferenceAlg, self).__init__(
            gamma, learning_rate_fcn, defaulit_state_value_fcn_value
        )

    def predict(
        self,
        env,
        policy_sampler,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
        record_history=False,
        verbose_mode=False,
        debug_mode=False,
    ):
        iter_num = 0

        for episode_num in range(max_num_episodes):
            state, _ = env.reset()

            if debug_mode:
                logger.debug(f"Starting an episode from {state}")

            for _ in range(max_num_transitions_per_episode):
                action = policy_sampler.get_action(state)
                if debug_mode:
                    logging.debug(f"Taking an action, {action}, at state, {state}")
                next_state, reward, is_terminal_state, _ = env.apply_action(action)
                if debug_mode:
                    logging.debug(f"Next state: {next_state}")
                learning_rate = self.get_learning_rate(iter_num, episode_num)

                current_state_value_fcn_value = self.state_value_fcn_dict[state]

                if is_terminal_state:
                    next_state_value_fcn_value = 0.0
                else:
                    next_state_value_fcn_value = self.state_value_fcn_dict[next_state]

                self.state_value_fcn_dict[state] += learning_rate * (
                    reward
                    + self.gamma * next_state_value_fcn_value
                    - current_state_value_fcn_value
                )

                if is_terminal_state:
                    break

                iter_num += 1

                if iter_num > max_num_iters:
                    break

                state = next_state

            if record_history:
                self.record_history(episode_num)

            if iter_num > max_num_iters:
                break
