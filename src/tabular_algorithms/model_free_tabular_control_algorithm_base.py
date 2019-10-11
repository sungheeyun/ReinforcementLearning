from abc import abstractmethod
from collections import defaultdict

from policy.epsilon_greedy_policy_sampler import EpsilonGreedyPolicySampler
from tabular_algorithms.tabular_rl_algorithms_base import TabularRLAlgorithmBase


class ModelFreeTabularControlAlgorithmBase(TabularRLAlgorithmBase):
    """
    Defines the interface that all tabular model-free reinforcement learning algorithms in this package
    should implement.
    """

    # @staticmethod
    # def default_action_selection_fcn(epsilon, action_value_dict, iter_num, episode_num):
    #    if nr.rand() < epsilon:
    #        return nr.choice(list(action_value_dict.keys()))
    #    else:
    #        return max(action_value_dict, key=action_value_dict.get)

    def __init__(self, gamma, learning_rate_fcn, action_selection_fcn, default_action_value_fcn_value):
        """
        Parameters
        ----------
        gamma: float
            The discount rate used when defining the discounted return.
        learning_rate_fcn: float or function
            Learning rate strategy. If it is a float, the learning rate strategy is a constant learning rate.
        action_selection_fcn: float or function
            Bahavior_policy_sampler_generator, e.g., epsilon-greedy policy sampler.
        default_action_value_fcn_value: float
            The default value for action value functions, i.e., Q-function.
        """

        super(ModelFreeTabularControlAlgorithmBase, self).__init__(gamma, learning_rate_fcn)

        self.learning_rate_strategy = learning_rate_fcn
        self.action_selection_fcn = action_selection_fcn
        self.default_action_value_fcn_value = default_action_value_fcn_value

        self.action_value_fcn_dict = defaultdict(
            lambda: defaultdict(lambda: float(self.default_action_value_fcn_value))
        )

        self.action_selection_strategy_fcn = None

        self._initialize()

    def _initialize(self):
        if isinstance(self.action_selection_fcn, (float, int)):
            fixed_epsilon = float(self.action_selection_fcn)
            self.action_selection_fcn = lambda action_value_dict, iter_num, episode_num: \
                EpsilonGreedyPolicySampler.get_action_from_state_action_dict(
                    action_value_dict, fixed_epsilon
                )

            # self.action_selection_fcn = lambda action_value_dict, iter_num, episode_num:\
            #    ModelFreeTabularControlAlgorithmBase.default_action_selection_fcn(
            #        fixed_epsilon, action_value_dict, iter_num, episode_num
            #    )

    def get_action(self, action_value_dict, iter_num, episode_num):
        """
        Returns the index of the action to be taken.

        Parameters
        ----------
        action_value_dict: dict
            Dict of action-value pairs. iter_num: int
            Iteration number
        episode_num: int
            Episode number

        Returns
        -------
        action:
            The action to be taken.
        """
        return self.action_selection_fcn(action_value_dict, iter_num, episode_num)

    @abstractmethod
    def learn(
            self,
            env,
            max_num_episodes,
            max_num_iters,
            max_num_transitions_per_episode,
            max_num_episodes_with_same_greedy_policy,
            does_record_history,
            verbose_mode,
            debug_mode
    ):
        """
        Learns the model using a tabular reinforcement learning method.

        Parameters
        ----------
        env: EnvironmentBase
            The environment instance by interacting with which the RL agent learns.
        max_num_episodes: int
            The maximum number of episodes as a stopping criterion.
        max_num_iters: int
            The maximum number of iterations as a stopping criterion.
        max_num_transitions_per_episode: int
            The maximum number of state transitions per episode.
        does_record_history: bool
            If True, it records history of value function values.
        verbose_mode: bool
            If True, it prints information related to inner workings of the learning algorithm.
        debug_mode: bool
            If True, it prints information for debugging.
        """
        pass

    def get_action_value_fcn_dict(self):
        """
        Returns a dict containing action value function values.

        Returns
        -------
        action_value_fcn_dict: dict[object, dict]
            Dict containing state - dict pairs where the latter dict is a dict with action - value pairs.
        """
        return self.action_value_fcn_dict

    def record_history(self, episode_num):
        if self.value_fcn_history_dict is None:
            self.value_fcn_history_dict = defaultdict(lambda: defaultdict(list))

        for state, action_value_dict in self.action_value_fcn_dict.items():
            for action, value in action_value_dict.items():
                if episode_num > 0 and len(self.value_fcn_history_dict[state][action]) == 0:
                    self.value_fcn_history_dict[state][action] = [self.default_action_value_fcn_value] * episode_num
                self.value_fcn_history_dict[state][action].append(value)

    def plot_value_fcn_history(self, ax, *args, **kwargs):
        history_value_list_list = list()

        for state, action_value_list_dict in self.value_fcn_history_dict.items():
            for action, value_list in action_value_list_dict.items():
                history_value_list_list.append(value_list)

        self._plot_value_fcn_history(ax, history_value_list_list, *args, **kwargs)
        ax.set_ylabel("Action Value Function Values")
