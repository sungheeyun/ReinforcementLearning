from typing import Any, List
from abc import ABC, abstractmethod


class PolicySamplerBase(ABC):

    @abstractmethod
    def get_action(self, state: Any) -> Any:
        """
        Returns action sample from the state.

        Parameters
        ----------
        state: Any
            The state at which an action is chosen to be taken.

        Returns
        -------
        action: Any
            The action chosen to be taken.
        """
        pass

    @abstractmethod
    def get_all_states(self) -> List[Any]:
        """
        Returns all the states.

        Returns
        -------
        alL_states:
            1-dimensional of all the states
        """
        pass
