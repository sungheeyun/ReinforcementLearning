from abc import abstractmethod
from collections import defaultdict
from typing import Optional, Dict, Any

from environment.environment_base import EnvironmentBase
from policy.policy_sampler_base import PolicySamplerBase
from tabular_algorithms.tabular_rl_algorithms_base import TabularRLAlgorithmBase


class ModelFreeTabularPredictionAlgorithmBase(TabularRLAlgorithmBase):
    """
    Defines the interface that all tabular model-free prediction algorithms in this package
    should implement.
    """

    def __init__(self, gamma, learning_rate_fcn, default_state_value_fcn_value):
        super(ModelFreeTabularPredictionAlgorithmBase, self).__init__(gamma, learning_rate_fcn)

        self.default_state_value_fcn_value = float(default_state_value_fcn_value)
        self.state_value_fcn_dict: Optional[Dict[Any, float]] = None

        self.reset_state_value_fcn_dict()

    @abstractmethod
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
        """
        Learns the model using a tabular reinforcement learning method.

        Parameters
        ----------
        env: EnvironmentBase
            The environment instance by interacting with which the RL agent learns.
        policy_sampler: ProbabilisticPolicy
            The random_policy for which the algorithm predicts the state-value functions.
        max_num_episodes: int
            The maximum number of episodes as a stopping criterion.
        max_num_iters: int
            The maximum number of iterations as a stopping criterion.
        max_num_transitions_per_episode: int
            The maximum number of state transitions per episode.
        record_history: bool
            If True, it records history of value function values.
        verbose_mode: bool
            If True, it prints lots of information via logging.Logger.info
        debug_mode: bool
            If True, it prints debug message via logging.Logger.debug
        """
        pass

    def reset_state_value_fcn_dict(self) -> None:
        self.state_value_fcn_dict = defaultdict(lambda: self.default_state_value_fcn_value)

    def get_state_value_fcn_dict(self) -> Optional[Dict[Any, float]]:
        """
        Returns a dict the key and value of which are state and state function value.

        Returns
        -------
        state_value_fcn_dict: Optional[Dict[Any, float]]
            Dict containing state - state function value pairs.
        """
        return self.state_value_fcn_dict

    def record_history(self, episode_num):
        if self.value_fcn_history_dict is None:
            self.value_fcn_history_dict = defaultdict(list)

        for state, value in self.state_value_fcn_dict.items():
            if episode_num > 0 and len(self.value_fcn_history_dict[state]) == 0:
                self.value_fcn_history_dict[state] = [self.default_state_value_fcn_value] * episode_num
            self.value_fcn_history_dict[state].append(value)

    def plot_value_fcn_history(self, ax, *args, **kwargs):
        history_value_list_list = list()

        for state, value_list in self.value_fcn_history_dict.items():
            history_value_list_list.append(value_list)

        self._plot_value_fcn_history(ax, history_value_list_list, *args, **kwargs)
        ax.set_ylabel("State Value Function Values")
