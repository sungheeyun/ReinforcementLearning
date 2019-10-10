from abc import ABC, abstractmethod


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
