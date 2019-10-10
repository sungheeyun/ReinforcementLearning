from typing import Tuple, Any, List
from abc import ABC, abstractmethod


class EnvironmentBase(ABC):
    """ Defines the interface that all environment classes in this package should implement """

    def __init__(self):
        self.drawing_index = 0

    @abstractmethod
    def reset(self) -> Tuple[Any, Any]:
        """
        Bring the environment back to the initial state.

        Returns
        -------
        state :
            initial state
        info:
            Extra information about environment, e.g., information regarding inner state, etc.
        """
        pass

    @abstractmethod
    def get_current_state(self) -> Tuple[Any, Any]:
        """
        Returns the current state of the environment.

        Returns
        -------
        state :
            The current state of the environment.
        info:
            Extra information about environment, e.g., information regarding inner state, etc.
        """
        pass

    @abstractmethod
    def set_state(self, state) -> None:
        """
        Forcefully set the state of the environment.

        Parameters
        ----------
        state :
            The state the environment is set to.
        info:
            Extra information about environment, e.g., information regarding inner state, etc.
        """
        pass

    @abstractmethod
    def apply_action(self, action) -> Tuple[Any, Any, bool, Any]:
        """
        Applies an action to the environment and returns the next state, reward, whether or not it is at a terminal
        state (after applying the action), and extra information (if available).

        Parameters
        ----------
        action :
            The action applied to the environment.

        Returns
        -------
        next_state :
            The next state the applied action causes the environment to move to.
        reward :
            The reward the state transition incurs.
        is_terminal_state : bool
            True if and only if the agent reaches the terminal state.
        info:
            Extra information about environment, e.g., information regarding inner state, etc.
        """
        pass

    @abstractmethod
    def get_all_available_actions(self, state) -> List[Any]:
        """
        Returns available actions at the current state.

        The return value is not defined when the environment is at a terminal state.

        Parameters
        ----------
        state: object
            State

        Returns
        -------
        action_list :
            List of available actions at the current state.
        """
        pass

    @abstractmethod
    def is_terminal_state(self) -> bool:
        """
        Returns true if the environment is in a terminal state, and false otherwise.
        """
        pass

    def draw_state_value_fcn_values(self, ax, state_value_fcn_dict, *pargs, **kargs):
        """
        Draws the values of the state value functions.
        """
        pass

    def draw_actions_value_fcn_values(self, ax, actions_value_fcn_dict, *pargs, **kargs):
        """
        Draws the values of the action value functions.
        """
        pass

    def draw_deterministic_actions_value_fcn_values(self, ax, actions_value_fcn_dict, *pargs, **kargs):
        """
        Draws the values of the action value functions.
        """
        pass

    def draw_deterministic_actions(self, ax, actions_value_fcn_dict, *pargs, **kargs):
        """
        Draws best actions.
        """
        pass

    def draw_boltzmann_actions(self, ax, actions_value_fcn_dict, *pargs, **kargs):
        """
        Draws all actions based on Boltzmann distribution.
        """
        pass

    def draw_3d_state_value_fcn_values(self, ax, state_value_fcn_dict, *pargs, **kargs):
        """
        Draws the values of the state value functions.
        """
        pass

    def draw_3d_deterministic_action_value_fcn_values(self, ax, actions_value_fcn_dict, *pargs, **kargs):
        """
        Draws the values of the action value functions.
        """
        pass
