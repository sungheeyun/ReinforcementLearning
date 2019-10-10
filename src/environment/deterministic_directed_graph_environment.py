from typing import Dict, Any, Tuple, List
import json
from logging import getLogger

from environment.deterministic_environment import DeterministicEnvironment

logger = getLogger()


class DeterministicDirectedGraphEnvironment(DeterministicEnvironment):
    """
    Implements a deterministic environment for which the state transition can be represented by a directed graph.
    """

    def __init__(
        self,
        start_state: Any,
        goal_state: Any,
        state_transition_dict: Dict[Any, Dict[Any, Any]],
        reward_dict: Dict[Any, Dict[Any, float]],
    ):
        self.state_transition_graph_dict: Dict[
            Any, Dict[Any, Any]
        ] = state_transition_dict
        self.state_transition_reward_dict: Dict[Any, Dict[Any, float]] = reward_dict
        self.start_state = start_state
        self.goal_state = goal_state

        self.current_state = None

        self._check_attributes()

    def _check_attributes(self) -> None:
        if self.start_state not in self.state_transition_graph_dict:
            raise KeyError(f"self.start_state should be in self.directed_graph_dict.")

    def reset(self) -> Tuple[Any, Any]:
        self.current_state = self.start_state
        return self.current_state, None

    def get_current_state(self) -> Tuple[Any, Any]:
        return self.current_state, None

    def set_state(self, state: Any) -> None:
        self.current_state = state
        return self.current_state, None

    def apply_action(self, action: Any) -> Tuple[Any, Any, bool, Any]:
        current_state: Any = self.current_state
        next_state: Any

        next_state = self.state_transition_graph_dict[current_state][action]
        reward: float = self.state_transition_reward_dict[current_state][action]

        self.current_state = next_state
        is_terminal_state: bool = self.is_terminal_state()
        info: Any = None

        return next_state, reward, is_terminal_state, info

    def get_all_available_actions(self, state: Any = None) -> List[Any]:
        if state is None:
            state = self.current_state

        return list(self.state_transition_graph_dict[state].keys())

    def is_terminal_state(self) -> bool:
        return self.current_state == self.goal_state


def read_deterministic_directed_graph_environment_from_json(
    json_file_path: str
) -> DeterministicDirectedGraphEnvironment:
    # TODO have this function take as an input dict obtained by parsing a json file, not json file name! And of course,
    # TODO and make proper changes accordingly wherever this function is called.

    with open(json_file_path) as fin:
        deterministic_directed_graph_environment_json_dict: Dict[str, Any] = json.load(
            fin
        )

    start_state = deterministic_directed_graph_environment_json_dict["StartState"]
    goal_state = deterministic_directed_graph_environment_json_dict["GoalState"]
    state_transition_graph_list: List[
        dict
    ] = deterministic_directed_graph_environment_json_dict["StateTransitionGraph"]

    state_transition_reward_list: List[
        dict
    ] = deterministic_directed_graph_environment_json_dict["StateTransitionReward"]

    logger.debug(
        "deterministic_directed_graph_environment_json_dict['DefaultReward'].__class__: "
        f"{deterministic_directed_graph_environment_json_dict['DefaultReward'].__class__}"
    )

    default_reward: float = float(
        deterministic_directed_graph_environment_json_dict["DefaultReward"]
    )

    # read state transition reward
    given_state_transition_reward_dict: Dict[Any, Dict[Any, float]] = dict()
    for state_transition_reward_from_one_state_dict in state_transition_reward_list:
        from_state = state_transition_reward_from_one_state_dict["FromState"]
        given_state_transition_reward_dict[from_state]: Dict[Any, float] = dict()
        action_reward_dict: Dict[Any, float]
        for action_reward_dict in state_transition_reward_from_one_state_dict[
            "StateTransition"
        ]:
            action: Any = action_reward_dict["Action"]
            reward: float = float(action_reward_dict["Reward"])
            given_state_transition_reward_dict[from_state][action] = float(reward)

    # read state transition graph and assign rewards
    state_transition_reward_dict: Dict[Any, Dict[Any, float]] = dict()
    state_transition_graph_dict: Dict[Any, Dict[Any, Any]] = dict()
    for state_transition_from_one_state_dict in state_transition_graph_list:
        from_state = state_transition_from_one_state_dict["FromState"]
        state_transition_graph_dict[from_state]: dict = dict()
        state_transition_reward_dict[from_state]: Dict[Any, float] = dict()

        action_reward_dict: Dict[Any, float]
        for action_reward_dict in state_transition_from_one_state_dict[
            "StateTransition"
        ]:
            action: Any = action_reward_dict["Action"]
            next_state: Any = action_reward_dict["NextState"]

            logger.debug(
                f"from_state: {from_state}, action: {action}, next_state: {next_state}"
            )
            state_transition_graph_dict[from_state][action] = next_state

            reward: float
            try:
                reward = given_state_transition_reward_dict[from_state][action]
            except KeyError:
                reward = default_reward

            state_transition_reward_dict[from_state][action] = reward

    return DeterministicDirectedGraphEnvironment(
        start_state,
        goal_state,
        state_transition_graph_dict,
        state_transition_reward_dict,
    )
