from typing import Union, Any, Tuple, Dict
from collections import defaultdict

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib import cm
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

    def apply_action(self, action: str) -> Tuple[int, float, bool, Any]:

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


class GridWorld(DeterministicEnvironment):
    """
    Grid world Markov decision process (MDP) as described in Sutton & Barto, 2018.
    """

    ARROW_LENGTH = 0.7
    ALL_ACTIONS_TUPLE = ("left", "right", "up", "down")
    ACTION_DELTA_XY_DICT = dict(left=(-1, 0), right=(1, 0), down=(0, -1), up=(0, 1))

    def __init__(
        self,
        width,
        height,
        goal_reward=1.0,
        normal_reward=0.0,
        hit_wall_reward=-1.0,
        upward_wind_list=None,
        rightward_wind_list=None,
    ):
        super(GridWorld, self).__init__()

        self.width = width
        self.height = height

        self.goal_reward = goal_reward
        self.normal_reward = normal_reward
        self.hit_wall_reward = hit_wall_reward

        self.upward_wind_list = upward_wind_list
        self.rightward_wind_list = rightward_wind_list

        self.start_state = None
        self.terminal_states_set = None
        self.current_state = None
        self.drawing_index = None

        self.__initialize()

    def __initialize(self):
        self.set_start_state((0, 0))
        self.set_terminal_states([(self.width - 1, self.height - 1)])

    def get_start_state(self):
        return self.start_state

    def get_terminal_states(self):
        return self.terminal_states_set

    def set_start_state(self, start_state):
        self.start_state = start_state

    def set_terminal_states(self, terminal_states):
        self.terminal_states_set = set(terminal_states)

    def reset(self):
        self.current_state = self.start_state
        return self.current_state, None

    def get_current_state(self):
        return self.current_state, None

    def set_state(self, state):
        self.current_state = state, None

    def apply_action(self, action):

        reward = self.normal_reward

        x, y = self.current_state

        if self.upward_wind_list:
            y += self.upward_wind_list[x]

        if self.rightward_wind_list:
            x += self.rightward_wind_list[y]

        if action == "left":
            x -= 1
        elif action == "right":
            x += 1
        elif action == "down":
            y -= 1
        elif action == "up":
            y += 1
        else:
            raise ValueError(
                f"'action' should be either 'left', 'right', 'up', or 'down'; {action} given"
            )

        if x < 0:
            x = 0
            reward = self.hit_wall_reward

        if x >= self.width:
            x = self.width - 1
            reward = self.hit_wall_reward

        if y < 0:
            y = 0
            reward = self.hit_wall_reward

        if y >= self.height:
            y = self.height - 1
            reward = self.hit_wall_reward

        self.current_state = x, y

        if self.is_terminal_state():
            reward = self.goal_reward

        return self.current_state, reward, self.is_terminal_state(), None

    def get_all_available_actions(self, state):
        return GridWorld.ALL_ACTIONS_TUPLE

    def is_terminal_state(self):
        return self.current_state in self.terminal_states_set

    def _draw_grid_world(self, ax):

        draw_kwargs = dict(linestyle="-", color="k", alpha=0.5)

        for col in range(self.width):
            ax.plot((col - 0.5) * np.ones(2), [-0.5, self.height - 0.5], **draw_kwargs)
        ax.plot(
            (self.width - 0.5) * np.ones(2), [-0.5, self.height - 0.5], **draw_kwargs
        )

        for row in range(self.height):
            ax.plot([-0.5, self.width - 0.5], (row - 0.5) * np.ones(2), **draw_kwargs)
        ax.plot(
            [-0.5, self.width - 0.5], (self.height - 0.5) * np.ones(2), **draw_kwargs
        )

        ax.add_artist(
            plt.Circle(
                self.get_start_state(),
                radius=GridWorld.ARROW_LENGTH * 0.5 * 1.1,
                color="b",
                fill=False,
            )
        )

        for terminal_state in self.get_terminal_states():
            ax.add_artist(
                plt.Circle(
                    terminal_state,
                    radius=GridWorld.ARROW_LENGTH * 0.5 * 1.1,
                    color="k",
                    fill=True,
                )
            )

        xlim = [-1.0, self.width]
        ylim = [-1.0, self.height]

        if self.upward_wind_list:
            for col in range(self.width):
                ax.text(
                    col,
                    -1.0,
                    f"{self.upward_wind_list[col]:.1f}",
                    ha="center",
                    va="top",
                )
                ylim[0] = -1.5

        if self.rightward_wind_list:
            for row in range(self.height):
                ax.text(-1.0, row, f"{self.rightward_wind_list[row]:.1f}")
                xlim[0] = -1.5

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.axis("equal")
        ax.axis("off")

    def draw_boltzmann_actions(self, ax, actions_value_fcn_dict, *args, **kwargs):
        arrow_half_length = GridWorld.ARROW_LENGTH / 2.0

        for state, actions_value_dict in actions_value_fcn_dict.items():
            x, y = state
            max_value = max(actions_value_dict.values())
            max_action = max(actions_value_dict, key=actions_value_dict.get)
            for action, value in actions_value_dict.items():
                delx, dely = GridWorld.ACTION_DELTA_XY_DICT[action]
                length = arrow_half_length * np.exp(value - max_value)

                kwargs_copied = kwargs.copy()

                if action == max_action:
                    kwargs_copied.update(color="r")

                ax.arrow(
                    x,
                    y,
                    length * delx,
                    length * dely,
                    head_width=0.3 * length,
                    length_includes_head=True,
                    *args,
                    **kwargs_copied,
                )

        self._draw_grid_world(ax)

    def draw_deterministic_actions(self, ax, actions_value_fcn_dict, *args, **kwargs):
        arrow_length = GridWorld.ARROW_LENGTH
        arrow_half_length = arrow_length / 2.0
        for state, actions_value_dict in actions_value_fcn_dict.items():
            x, y = state
            best_action = max(actions_value_dict, key=actions_value_dict.get)
            delx, dely = GridWorld.ACTION_DELTA_XY_DICT[best_action]
            ax.arrow(
                x - arrow_half_length * delx,
                y - arrow_half_length * dely,
                arrow_length * delx,
                arrow_length * dely,
                head_width=0.1 * arrow_length,
                length_includes_head=True,
                *args,
                **kwargs,
            )

        self._draw_grid_world(ax)

    def _draw_state_value_fcn_values(self, ax, state_value_fcn_dict, *args, **kwargs):

        for state, value in state_value_fcn_dict.items():
            x, y = state
            ax.text(x, y, f"{value:.2f}", ha="center", va="center")

        self._draw_grid_world(ax)

    def draw_state_value_fcn_values(self, ax, state_value_fcn_dict, *args, **kwargs):
        self._draw_state_value_fcn_values(ax, state_value_fcn_dict, *args, **kwargs)

        ax.set_title("State Value Functions")

    def draw_deterministic_actions_value_fcn_values(
        self, ax, actions_value_fcn_dict, *args, **kwargs
    ):

        state_deterministic_action_value_dict = dict()
        for state, action_value_dict in actions_value_fcn_dict.items():
            state_deterministic_action_value_dict[state] = max(
                action_value_dict.values()
            )

        self._draw_state_value_fcn_values(
            ax, state_deterministic_action_value_dict, *args, **kwargs
        )

        ax.set_title("Deterministic Action Value Functions")

    def _draw_3d_value_fcn_values(self, ax, X, Y, V, *args, **kwargs):

        x_1d_array = X.ravel()
        y_1d_array = Y.ravel()
        v_1d_array = V.ravel()

        non_nan_idx_array = np.logical_not(np.isnan(v_1d_array))

        non_nan_x_1d_array = x_1d_array[non_nan_idx_array]
        non_nan_y_1d_array = y_1d_array[non_nan_idx_array]
        non_nan_v_1d_array = v_1d_array[non_nan_idx_array]

        V_draw = V.copy()
        inds = np.where(np.isnan(V))
        V_draw[inds] = non_nan_v_1d_array.min()

        interpolated_V = interpolate.griddata(
            (non_nan_x_1d_array, non_nan_y_1d_array),
            non_nan_v_1d_array,
            (X, Y),
            method="nearest",
        )

        V_draw[inds] = interpolated_V[inds]

        ax.plot_surface(
            X,
            Y,
            V_draw,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            shade=False,
            *args,
            **kwargs,
        )
        ax.scatter(
            non_nan_x_1d_array,
            non_nan_y_1d_array,
            non_nan_v_1d_array,
            marker="o",
            c="b",
            s=100,
            depthshade=True,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    def draw_3d_deterministic_action_value_fcn_values(
        self, ax, actions_value_fcn_dict, *args, **kwargs
    ):

        x_list = range(self.width)
        y_list = range(self.height)

        X, Y = np.meshgrid(x_list, y_list)
        V = np.zeros_like(X, float)

        for row_idx in range(X.shape[0]):
            for col_idx in range(X.shape[1]):
                x = X[row_idx, col_idx]
                y = Y[row_idx, col_idx]

                state = x, y

                value_fcn_value = np.nan
                if state in actions_value_fcn_dict:
                    action_value_dict = actions_value_fcn_dict[state]
                    value_fcn_value = max(action_value_dict.values())

                V[row_idx, col_idx] = value_fcn_value

        self._draw_3d_value_fcn_values(ax, X, Y, V, *args, **kwargs)
        ax.set_zlabel("Deterministic Action Value Function")

    def draw_3d_state_value_fcn_values(self, ax, state_value_fcn_dict, *args, **kwargs):

        x_list = range(self.width)
        y_list = range(self.height)

        X, Y = np.meshgrid(x_list, y_list)
        V = np.zeros_like(X, float)

        for row_idx in range(X.shape[0]):
            for col_idx in range(X.shape[1]):
                x = X[row_idx, col_idx]
                y = Y[row_idx, col_idx]

                state = x, y
                value_fcn_value = np.nan
                if state in state_value_fcn_dict:
                    value_fcn_value = state_value_fcn_dict[state]

                V[row_idx, col_idx] = value_fcn_value

        self._draw_3d_value_fcn_values(ax, X, Y, V, *args, **kwargs)
        ax.set_zlabel("State Value Function")


class GridWorldWithCliff(GridWorld):
    """
    Grid world Markov decision process (MDP) with cliff.
    """

    def __init__(self, *args, **kwargs):
        super(GridWorldWithCliff, self).__init__(*args, **kwargs)
        self.__initialize()

    def __initialize(self):
        self.set_start_state((0, 0))
        self.set_terminal_states([(self.width - 1, 0)])

    def apply_action(self, action):
        current_state, reward, is_terminal_state, info = super(
            GridWorldWithCliff, self
        ).apply_action(action)

        x, y = self.current_state
        if not is_terminal_state and (x > 0 and x < self.width - 1 and y == 0):
            reward = -100.0
            info = "dropped at the cliff; going back to the start"
            self.reset()

        return self.current_state, reward, is_terminal_state, info


if __name__ == "__main__":

    rwe = RandomWalkEnvironment(5)

    print(rwe.reset())
    print(rwe.apply_action("right"))
    print(rwe.apply_action("right"))
    print(rwe.apply_action("right"))

    print(rwe.reset())
    print(rwe.apply_action("left"))
    print(rwe.apply_action("left"))
    print(rwe.apply_action("left"))
    print(rwe.apply_action("left"))
