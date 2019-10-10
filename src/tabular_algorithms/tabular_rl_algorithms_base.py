""" Base classes for tabular reinforcement learning algorithms in this package """

from abc import abstractmethod, ABC
from collections import defaultdict

import numpy as np

from policy.epsilon_greedy_policy_sampler import EpsilonGreedyPolicySampler


class TabularAlgBase(ABC):
    """ Defines the interface that all tabular algorithms in this package should implement """

    def __init__(self, gamma, learning_rate_fcn):
        """
        Parameters
        ----------
        gamma : float
            The discount rate used when defining the discounted return.
        learning_rate_fcn: float or function
            Learning rate strategy. If it is a float, the learning rate strategy is a constant learning rate.
        """
        self.gamma = gamma
        self.learning_rate_fcn = learning_rate_fcn
        self.value_fcn_history_dict = None

        self.__initialize()

    def __initialize(self):
        if isinstance(self.learning_rate_fcn, (float, int)):
            fixed_learning_rate = float(self.learning_rate_fcn)
            self.learning_rate_fcn = lambda iter_num, episode_num: fixed_learning_rate

    def get_learning_rate(self, iter_num, episode_num):
        """
        Returns the learning rate.

        Parameters
        ----------
        iter_num: int
            Iteration number. If it's the nth iteration, the value of iter_num should be n-1.
        episode_num: int
            Episode number. If it's the nth episode, the value of episode_num should be n-1.

        Returns
        -------
        learning_rate: float
            Learning rate for reinforcement learning update.
        """
        return self.learning_rate_fcn(iter_num, episode_num)

    @abstractmethod
    def record_history(self, episode_num):
        """
        Records history of value functions.

        Parameters
        ----------
        episode_num: int
            Episode number.
        """

    @abstractmethod
    def plot_value_fcn_history(self, ax, *pargs, **kargs):
        """
        Plots value function history.

        Parameters
        ----------
        ax:
            Subplot Axes.
        *pargs: tuple
            Argument list for plotting.
        **kargs: dict
            keyword parameters for plotting.
        :return:
        """

    def _plot_value_fcn_history(self, ax, history_value_list_list, *pargs, **kargs):
        value_2d_array = np.array(history_value_list_list).T

        ax.plot(value_2d_array, *pargs, **kargs)
        ax.set_xlabel("episode num")


class ModelFreeTabularPredictionAlgBase(TabularAlgBase):
    """
    Defines the interface that all tabular model-free prediction algorithms in this package
    should implement.
    """

    def __init__(self, gamma, learning_rate_fcn, default_state_value_fcn_value):
        super(ModelFreeTabularPredictionAlgBase, self).__init__(gamma, learning_rate_fcn)

        self.default_state_value_fcn_value = float(default_state_value_fcn_value)
        self.state_value_fcn_dict = None

        self.reset_state_value_fcn_dict()

    @abstractmethod
    def predict(
        self,
        env,
        policy_sampler,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
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
        policy_sampler: ProbabilisticPolicy
            The random_policy for which the algorithm predicts the state-value functions.
        max_num_episodes: int
            The maximum number of episodes as a stopping criterion.
        max_num_iters: int
            The maximum number of iterations as a stopping criterion.
        max_num_transitions_per_episode: int
            The maximum number of state transitions per episode.
        does_record_history: bool
            If True, it records history of value function values.
        """
        pass

    def reset_state_value_fcn_dict(self):
        self.state_value_fcn_dict = defaultdict(lambda: self.default_state_value_fcn_value)

    def get_state_value_fcn_dict(self):
        """
        Returns a dict the key and value of which are state and state function value.

        Returns
        -------
        state_value_fcn_dict: dict
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

    def plot_value_fcn_history(self, ax, *pargs, **kargs):
        history_value_list_list = list()

        for state, value_list in self.value_fcn_history_dict.items():
            history_value_list_list.append(value_list)

        self._plot_value_fcn_history(ax, history_value_list_list, *pargs, **kargs)
        ax.set_ylabel("State Value Function Values")


class ModelFreeTabularControlAlgBase(TabularAlgBase):
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

        super(ModelFreeTabularControlAlgBase, self).__init__(gamma, learning_rate_fcn)

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
            #    ModelFreeTabularControlAlgBase.default_action_selection_fcn(
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

    def plot_value_fcn_history(self, ax, *pargs, **kargs):
        history_value_list_list = list()

        for state, action_value_list_dict in self.value_fcn_history_dict.items():
            for action, value_list in action_value_list_dict.items():
                history_value_list_list.append(value_list)

        self._plot_value_fcn_history(ax, history_value_list_list, *pargs, **kargs)
        ax.set_ylabel("Action Value Function Values")


class ModelFreeTabularOneStepControlAlgBase(ModelFreeTabularControlAlgBase):
    pass


class ModelFreeTabularNStepControlAlgBase(ModelFreeTabularControlAlgBase):
    pass


class ModelFreeTabularTDLambdaControlAlgBase(ModelFreeTabularControlAlgBase):
    pass
