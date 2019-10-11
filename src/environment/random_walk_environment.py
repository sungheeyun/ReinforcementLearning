from collections import defaultdict
from typing import Tuple, Union, Any, Dict

import numpy as np
from matplotlib.axes import Axes

from environment.deterministic_environment import DeterministicEnvironment


class RandomWalkEnvironment(DeterministicEnvironment):
    """
    Random work Markov reward process (MRP) as described in Example 6.2 of Sutton & Barto, 2018.
    """

    all_action_tuple: Tuple[str] = ("left", "right")

    def __init__(
        self,
        num_nodes: int,
        leftmost_reward: Union[float, int] = 0.0,
        rightmost_reward: Union[float, int] = 1.0,
    ):
        super(RandomWalkEnvironment, self).__init__()
        self.num_nodes: int = num_nodes
        self.leftmost_reward: float = float(leftmost_reward)
        self.rightmost_reward: float = float(rightmost_reward)

        self.current_state: int = None

    def reset(self) -> Tuple[int, Any]:
        self.current_state = int((self.num_nodes + 1) / 2)
        return self.current_state, None

    def get_current_state(self) -> Tuple[int, Any]:
        return self.current_state, None

    def set_state(self, state: int) -> None:
        self.current_state = state, None

    def apply_action(self, action: object) -> object:

        if action == "left":
            self.current_state -= 1
        elif action == "right":
            self.current_state += 1
        else:
            raise ValueError(
                f"action should be either 'left' or 'right'; action: {action}"
            )

        if self.current_state == self.num_nodes + 1:
            reward = self.rightmost_reward
        elif self.current_state == 0:
            reward = self.leftmost_reward
        else:
            reward = 0.0

        return self.current_state, reward, self.is_terminal_state(), None

    def get_all_available_actions(self, state: int) -> Tuple[str]:
        return RandomWalkEnvironment.all_action_tuple

    def is_terminal_state(self) -> bool:
        return self.current_state == 0 or self.current_state == self.num_nodes + 1

    def get_num_nodes(self) -> int:
        return self.num_nodes

    def get_leftmost_reward(self) -> float:
        return self.leftmost_reward

    def get_rightmost_reward(self) -> float:
        return self.rightmost_reward

    def draw_state_value_fcn_values(
        self, ax: Axes, state_value_fcn_dict: Dict[Any, float], *args, **kwargs
    ) -> None:
        state_list = list(state_value_fcn_dict.keys())
        state_list.sort()

        value_list = [state_value_fcn_dict[state] for state in state_list]

        x_array = np.r_[0, state_list, self.num_nodes + 1]
        y_array = np.r_[self.leftmost_reward, value_list, self.rightmost_reward]

        ax.plot(x_array, y_array, *args, **kwargs)

    def draw_actions_value_fcn_values(
        self,
        ax: Axes,
        actions_value_fcn_dict: Dict[Any, Dict[Any, float]],
        *args,
        **kwargs,
    ) -> None:
        state_list = list(actions_value_fcn_dict.keys())
        state_list.sort()

        action_values_dict = defaultdict(lambda: np.zeros(len(state_list)))

        for idx, state in enumerate(state_list):
            for action, value in actions_value_fcn_dict[state].items():
                action_values_dict[action][idx] = value

        self.drawing_index += 1
        drawing_label = kwargs.pop("label", f"{self.drawing_index}")

        x_array = np.r_[0, state_list, self.num_nodes + 1]
        for action, value_array in action_values_dict.items():
            y_array = np.r_[self.leftmost_reward, value_array, self.rightmost_reward]
            ax.plot(
                x_array, y_array, label=f"{drawing_label}_{action}", *args, **kwargs
            )
