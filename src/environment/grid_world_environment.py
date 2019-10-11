from typing import Optional, Any, Tuple, List, Union, Set, Iterable, Dict

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt, cm
from matplotlib.axes import Axes

from environment.deterministic_environment import DeterministicEnvironment


class GridWorld(DeterministicEnvironment):
    """
    Grid world Markov decision process (MDP) as described in Sutton & Barto, 2018.
    """

    ARROW_LENGTH = 0.7
    ALL_ACTIONS_TUPLE: Tuple[str] = ("left", "right", "up", "down")
    ACTION_DELTA_XY_DICT = dict(left=(-1, 0), right=(1, 0), down=(0, -1), up=(0, 1))

    def __init__(
        self,
        width: int,
        height: int,
        goal_reward: Union[float, int] = 1.0,
        normal_reward: Union[float, int] = 0.0,
        hit_wall_reward: Union[float, int] = -1.0,
        upward_wind_list: Optional[List[int]] = None,
        rightward_wind_list: Optional[List[int]] = None
    ):
        super(GridWorld, self).__init__()

        self.width: int = width
        self.height: int = height

        self.goal_reward: float = float(goal_reward)
        self.normal_reward: float = float(normal_reward)
        self.hit_wall_reward: float = float(hit_wall_reward)

        self.upward_wind_list: Optional[List[int]] = upward_wind_list
        self.rightward_wind_list: Optional[List[int]] = rightward_wind_list

        self.start_state: Optional[Tuple[int, int]] = None
        self.terminal_states_set: Optional[Set[Tuple[int, int]]] = None
        self.current_state: Optional[Tuple[int, int]] = None

        self.__initialize()

    def __initialize(self) -> None:
        self.set_start_state((0, 0))
        self.set_terminal_states([(self.width - 1, self.height - 1)])

    def get_start_state(self) -> Tuple[int, int]:
        return self.start_state

    def get_terminal_states(self) -> Tuple[int, int]:
        return self.terminal_states_set

    def set_start_state(self, start_state: Tuple[int, int]) -> None:
        self.start_state = start_state

    def set_terminal_states(self, terminal_states: Iterable[Tuple[int, int]]) -> None:
        self.terminal_states_set = set(terminal_states)

    def reset(self) -> Tuple[Tuple[int, int], Any]:
        self.current_state = self.start_state
        return self.current_state, None

    def get_current_state(self) -> Tuple[Tuple[int, int], Any]:
        return self.current_state, None

    def set_state(self, state: Tuple[int, int]) -> None:
        self.current_state = state, None

    def apply_action(self, action: str) -> Tuple[Tuple[int, int], float, bool, Any]:

        reward: float = self.normal_reward

        x: int
        y: int
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

    def get_all_available_actions(self, state: Optional[Tuple[int, int]] = None) -> Tuple[str]:
        return GridWorld.ALL_ACTIONS_TUPLE

    def is_terminal_state(self) -> bool:
        return self.current_state in self.terminal_states_set

    def _draw_grid_world(self, axis: Axes) -> None:

        draw_kwargs: Dict[str, Any] = dict(linestyle="-", color="k", alpha=0.5)

        col: int
        for col in range(self.width):
            axis.plot((col - 0.5) * np.ones(2), [-0.5, self.height - 0.5], **draw_kwargs)

        axis.plot((self.width - 0.5) * np.ones(2), [-0.5, self.height - 0.5], **draw_kwargs)

        row: int
        for row in range(self.height):
            axis.plot([-0.5, self.width - 0.5], (row - 0.5) * np.ones(2), **draw_kwargs)

        axis.plot([-0.5, self.width - 0.5], (self.height - 0.5) * np.ones(2), **draw_kwargs)

        axis.add_artist(
            plt.Circle(
                self.get_start_state(),
                radius=GridWorld.ARROW_LENGTH * 0.5 * 1.1,
                color="b",
                fill=False,
            )
        )

        for terminal_state in self.get_terminal_states():
            axis.add_artist(
                plt.Circle(
                    terminal_state,
                    radius=GridWorld.ARROW_LENGTH * 0.5 * 1.1,
                    color="k",
                    fill=True,
                )
            )

        xlim: Tuple[int, int] = [-1.0, self.width]
        ylim: Tuple[int, int] = [-1.0, self.height]
        # TODO adding typing from here

        if self.upward_wind_list:
            for col in range(self.width):
                axis.text(
                    col,
                    -1.0,
                    f"{self.upward_wind_list[col]:.1f}",
                    ha="center",
                    va="top",
                )
                ylim[0] = -1.5

        if self.rightward_wind_list:
            for row in range(self.height):
                axis.text(-1.0, row, f"{self.rightward_wind_list[row]:.1f}")
                xlim[0] = -1.5

        axis.set_xlim(xlim)
        axis.set_ylim(ylim)

        axis.axis("equal")
        axis.axis("off")

    def draw_boltzmann_actions(
            self,
            axis: Axes,
            actions_value_fcn_dict: Dict[Tuple[int, int], Dict[str, float]],
            *args,
            **kwargs
    ) -> None:
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

                axis.arrow(
                    x,
                    y,
                    length * delx,
                    length * dely,
                    head_width=0.3 * length,
                    length_includes_head=True,
                    *args,
                    **kwargs_copied,
                )

        self._draw_grid_world(axis)

    def draw_deterministic_actions(
            self,
            axis: Axes,
            actions_value_fcn_dict: Dict[Tuple[int, int], Dict[str, float]],
            *args,
            **kwargs
    ) -> None:
        arrow_length = GridWorld.ARROW_LENGTH
        arrow_half_length = arrow_length / 2.0
        for state, actions_value_dict in actions_value_fcn_dict.items():
            x, y = state
            best_action = max(actions_value_dict, key=actions_value_dict.get)
            delx, dely = GridWorld.ACTION_DELTA_XY_DICT[best_action]
            axis.arrow(
                x - arrow_half_length * delx,
                y - arrow_half_length * dely,
                arrow_length * delx,
                arrow_length * dely,
                head_width=0.1 * arrow_length,
                length_includes_head=True,
                *args,
                **kwargs,
            )

        self._draw_grid_world(axis)

    def _draw_state_value_fcn_values(
            self,
            axis: Axes,
            state_value_fcn_dict: Dict[Tuple[int, int], float]
    ) -> None:
        for state, value in state_value_fcn_dict.items():
            x, y = state
            axis.text(x, y, f"{value:.2f}", ha="center", va="center")

        self._draw_grid_world(axis)

    def draw_state_value_fcn_values(
            self,
            axis: Axes,
            state_value_fcn_dict: Dict[Tuple[int, int], float],
            *args,
            **kwargs
    ) -> None:
        self._draw_state_value_fcn_values(axis, state_value_fcn_dict, *args, **kwargs)

        axis.set_title("State Value Functions")

    def draw_deterministic_actions_value_fcn_values(
            self,
            axis: Axes,
            actions_value_fcn_dict: Dict[Tuple[int, int], Dict[str, float]],
            *args,
            **kwargs
    ) -> None:
        state_deterministic_action_value_dict = dict()
        for state, action_value_dict in actions_value_fcn_dict.items():
            state_deterministic_action_value_dict[state] = max(
                action_value_dict.values()
            )

        self._draw_state_value_fcn_values(
            axis, state_deterministic_action_value_dict, *args, **kwargs
        )

        axis.set_title("Deterministic Action Value Functions")

    def _draw_3d_value_fcn_values(
            self,
            ax,
            X,
            Y,
            V,
            *args,
            **kwargs
    ) -> None:

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
            self,
            axis: Axes,
            actions_value_fcn_dict: Dict[Tuple[int, int], Dict[str, float]],
            *args,
            **kwargs
    ) -> None:

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

        self._draw_3d_value_fcn_values(axis, X, Y, V, *args, **kwargs)
        axis.set_zlabel("Deterministic Action Value Function")

    def draw_3d_state_value_fcn_values(
            self,
            axis: Axes,
            state_value_fcn_dict: Dict[Tuple[int, int], float],
            *args,
            **kwargs
    ) -> None:

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

        self._draw_3d_value_fcn_values(axis, X, Y, V, *args, **kwargs)

        axis.set_zlabel("State Value Function")
