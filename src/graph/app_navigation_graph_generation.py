from typing import Dict, Tuple, List
import logging

import numpy as np
import networkx as nx


logger = logging.getLogger()


class AppNavigationGraph:

    def __init__(
            self,
            directed_graph: nx.DiGraph,
            start_node: int,
            goal_node: int,
            pos: Dict[int, Tuple[float, float]]
    ):
        self.directed_graph: nx.DiGraph = directed_graph
        self.start_node: int = start_node
        self.goal_node: int = goal_node
        self.pos: Dict[int, Tuple[float, float]] = pos


def get_x_coor(index: int, total_num: int, margin: float = 0.0) -> float:
    return float(index) / total_num * (1.0 - 2 * margin) + margin


def generate_app_navigation_graph(
        shortest_path_length: int
) -> AppNavigationGraph:

    app_nav_graph: nx.DiGraph = nx.DiGraph()

    pos: Dict[int, Tuple[float, float]] = dict()

    start_node: int
    first_layer_node_list: List[int] = list()
    for start_node in range(shortest_path_length):
        end_node: int = start_node + 1
        app_nav_graph.add_edge(start_node, end_node)
        pos[start_node] = (get_x_coor(start_node, shortest_path_length), 0.0)

        first_layer_node_list.append(start_node)
    first_layer_node_list.append(end_node)

    pos[end_node] = (float(end_node)/shortest_path_length, 0.0)

    prev_end_node = end_node

    second_layer_length: int = int(shortest_path_length * 1.5)

    second_layer_node_list: List[int] = list()
    for idx in range(second_layer_length):
        start_node: int = prev_end_node + idx + 1
        end_node: int = start_node + 1

        app_nav_graph.add_edge(start_node, end_node)
        pos[start_node] = (get_x_coor(idx, second_layer_length), 1.0)
        second_layer_node_list.append(start_node)

    pos[end_node] = (get_x_coor(idx + 1, second_layer_length), 1.0)
    second_layer_node_list.append(end_node)

    for idx in range(2):
        node: int
        for node in first_layer_node_list[:-1]:
            app_nav_graph.add_edge(node, int(np.random.choice(second_layer_node_list)))

    start_node = 0
    goal_node = shortest_path_length

    app_nav_graph.add_edge(second_layer_node_list[-1], first_layer_node_list[-1])

    return AppNavigationGraph(app_nav_graph, start_node, goal_node, pos)


def generate_multi_layer_app_navigation_graph(
        shortest_path_length: int, num_layers: int, layer_size_reduction_rate: float = 0.9
) -> AppNavigationGraph:

    di_graph: nx.DiGraph = nx.DiGraph()

    layer_node_list_list: List[List[int]] = list()
    pos: Dict[int, Tuple[float, float]] = dict()

    num_nodes_in_layer: int = shortest_path_length + 1
    pos_margin: float = 0.0
    prev_last_node: int = -1

    layer_idx: int
    for layer_idx in range(num_layers):

        y_coor: float = float(layer_idx)

        layer_node_list: List[int] = list()

        node_idx: int
        for node_idx in range(num_nodes_in_layer-1):
            start_node: int = prev_last_node + node_idx + 1
            end_node: int = start_node + 1

            di_graph.add_edge(start_node, end_node)
            layer_node_list.append(start_node)
            pos[start_node] = (get_x_coor(node_idx, num_nodes_in_layer - 1, pos_margin), y_coor)
        layer_node_list.append(end_node)
        pos[end_node] = (get_x_coor(node_idx + 1, num_nodes_in_layer - 1, pos_margin), y_coor)
        prev_last_node = end_node

        layer_node_list_list.append(layer_node_list)

        num_nodes_in_layer = int(num_nodes_in_layer * layer_size_reduction_rate)
        pos_margin = 1.0 - (1.0 - pos_margin) * layer_size_reduction_rate

    start_node = 0
    goal_node = shortest_path_length

    layer_idx: int
    for layer_idx in range(num_layers - 1):
        lower_layer_node_list: List[int] = layer_node_list_list[layer_idx]
        upper_layer_node_list: List[int] = layer_node_list_list[layer_idx + 1]

        di_graph.add_edge(lower_layer_node_list[0], upper_layer_node_list[0])
        di_graph.add_edge(upper_layer_node_list[-1], lower_layer_node_list[-1])

        from_node: int
        for from_node in lower_layer_node_list:
            to_node: int = int(np.random.choice(upper_layer_node_list))
            di_graph.add_edge(from_node, to_node)

        from_node: int
        for from_node in upper_layer_node_list:
            to_node: int = int(np.random.choice(lower_layer_node_list))
            di_graph.add_edge(from_node, to_node)

    return AppNavigationGraph(di_graph, start_node, goal_node, pos)


def app_navigation_graph_to_json_obj(app_nav_graph: AppNavigationGraph) -> dict:

    digraph: nx.DiGraph = app_nav_graph.directed_graph

    json_obj: dict = dict()

    json_obj["StartState"] = app_nav_graph.start_node
    json_obj["GoalState"] = app_nav_graph.goal_node

    state_transition_graph: List[dict] = list()

    from_state: int
    for from_state in digraph.nodes:
        state_transition: List[Dict[str, int]] = list()

        action: int
        next_state: int
        for action, next_state in enumerate(digraph.neighbors(from_state)):
            state_transition.append(dict(Action=action, NextState=next_state))

        if state_transition:
            state_transition_graph.append(dict(
                FromState=from_state,
                StateTransition=state_transition
            ))

    json_obj["StateTransitionGraph"] = state_transition_graph
    json_obj["StateTransitionReward"] = list()
    json_obj["DefaultReward"] = -1.0

    return json_obj
