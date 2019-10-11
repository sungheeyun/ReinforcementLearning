""" Classes for tabular temporal differencing prediction methods """

from typing import Any
import logging

from policy.policy_sampler_base import PolicySamplerBase
from environment.environment_base import EnvironmentBase
from tabular_algorithms.tabular_rl_algorithms_base import ModelFreeTabularPredictionAlgorithmBase


logger = logging.getLogger()


class OneStepTemporalDifferenceAlgorithm(ModelFreeTabularPredictionAlgorithmBase):
    """
    Implements temporal difference prediction algorithms.
    """

    def __init__(self, gamma, learning_rate_fcn, defaulit_state_value_fcn_value):
        super(OneStepTemporalDifferenceAlgorithm, self).__init__(
            gamma, learning_rate_fcn, defaulit_state_value_fcn_value
        )

    def predict(
        self,
        env: EnvironmentBase,
        policy_sampler: PolicySamplerBase,
        max_num_episodes: int,
        max_num_iters: int,
        max_num_transitions_per_episode: int,
        record_history: bool = False,
        verbose_mode: bool = False,
        debug_mode: bool = False
    ) -> None:
        iter_num: int = 0

        episode_num: int
        for episode_num in range(max_num_episodes):
            state: Any
            state, _ = env.reset()

            if debug_mode:
                logger.debug(f"Starting an episode from {state}")

            for _ in range(max_num_transitions_per_episode):
                action: Any = policy_sampler.get_action(state)
                if debug_mode:
                    logging.debug(f"Taking an action, {action}, at state, {state}")

                next_state: Any
                reward: float
                is_terminal_state: bool
                next_state, reward, is_terminal_state, _ = env.apply_action(action)

                if debug_mode:
                    logging.debug(f"Next state: {next_state}")
                learning_rate: float = self.get_learning_rate(iter_num, episode_num)

                current_state_value_fcn_value: float = self.state_value_fcn_dict[state]

                next_state_value_fcn_value: float
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
