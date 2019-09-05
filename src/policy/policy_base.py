""" Base class for policy and policy sampler for Markov decision process in this package """

from abc import ABC, abstractmethod

import numpy as np


class PolicySamplerBase(ABC):

    @abstractmethod
    def get_action(self, state):
        """
        Returns action sample from the state.

        Parameters
        ----------
        state: object
            The state at which an action is chosen to be taken.

        Returns
        -------
        action: object
            The action chosen to be taken.
        """
        pass

    @abstractmethod
    def get_all_states(self):
        """
        Returns all the states.

        Returns
        -------
        alL_states:
            1-dimensional of all the states
        """
        pass


class PolicyBase(PolicySamplerBase):

    @abstractmethod
    def get_state_action_probability_dict_dict(self):
        """
        Returns state - action probability distribution dict.

        Returns
        -------
        action: dict
            Dict with state - action probability dictribution dict.
        """
        pass

    @abstractmethod
    def get_action_probability_dict(self, state):
        """
        Returns action probability distribution from the state.

        Parameters
        ----------
        state: object
            The state at which an action is chosen to be taken.

        Returns
        -------
        action: dict
            Dict with action - probability pairs.
        """
        pass

    def get_action(self, state):
        action_probability_dict = self.get_action_probability_dict(state)
        action_probability_list = [(action, probability) for action, probability in action_probability_dict.items()]

        return action_probability_list[np.random.multinomial(1, [x[1] for x in action_probability_list]).argmax()][0]

    def get_all_states(self):
        return list(self.get_state_action_probability_dict_dict().keys())
