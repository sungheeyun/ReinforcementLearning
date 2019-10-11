""" Base classes for tabular reinforcement learning algorithms in this package """

from abc import abstractmethod, ABC

import numpy as np


class TabularRLAlgorithmBase(ABC):
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

    def get_learning_rate(self, iter_num: int, episode_num: int) -> float:
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
    def plot_value_fcn_history(self, ax, *args, **kwargs):
        """
        Plots value function history.

        Parameters
        ----------
        ax:
            Subplot Axes.
        *args: tuple
            Argument list for plotting.
        **kwargs: dict
            keyword parameters for plotting.
        :return:
        """

    def _plot_value_fcn_history(self, ax, history_value_list_list, *args, **kwargs):
        value_2d_array = np.array(history_value_list_list).T

        ax.plot(value_2d_array, *args, **kwargs)
        ax.set_xlabel("episode num")
