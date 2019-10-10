from typing import List
import unittest
import logging
import os

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import networkx as nx

from utils import get_pretty_json_str, set_logging_basic_config
from graph.app_nav_graph_generation import (
    AppNavGraph, generate_app_navigation_graph, app_nav_graph_to_json_obj, generate_multi_layer_app_nav_graph
)

logger = logging.getLogger()


class TestAppNavGraph(unittest.TestCase):

    test_data_directory = os.path.join(os.curdir, "data")

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_generate_app_navigation_graph(self):

        shortest_path_length = 10

        app_nav_graph: AppNavGraph = generate_app_navigation_graph(shortest_path_length)

        figure: Figure
        axis: Axes

        figure, axis = plt.subplots()
        nx.draw(app_nav_graph.app_nav_graph, axis=axis, pos=app_nav_graph.pos)
        nx.draw_networkx_labels(app_nav_graph.app_nav_graph, axis=axis, pos=app_nav_graph.pos)

        axis.axis([-.1, 1.1, -.1, 1.1])

        figure.show()

        json_obj: dict = app_nav_graph_to_json_obj(app_nav_graph)

        logger.debug("json_obj from graph")
        logger.debug(get_pretty_json_str(json_obj))

        state_transition_graph_json_file_path = os.path.join(
            TestAppNavGraph.test_data_directory,
            'large_deterministic_state_transition_graph.json'
        )

        # with open(state_transition_graph_json_file_path, 'w') as fout:
        #    fout.write(get_pretty_json_str(json_obj) + "\n")

        self.assertEqual(True, True)

    def test_generate_multi_layer_app_nav_graph(self):

        shortest_path_length = 10
        num_layers = 4

        app_nav_graph: AppNavGraph = generate_multi_layer_app_nav_graph(shortest_path_length, num_layers)

        figure: Figure
        axis: Axes

        figure, axis = plt.subplots()
        nx.draw(app_nav_graph.app_nav_graph, axis=axis, pos=app_nav_graph.pos)
        nx.draw_networkx_labels(app_nav_graph.app_nav_graph, axis=axis, pos=app_nav_graph.pos)

        axis.axis([-.1, 1.1, -.1, 3.1])

        figure.show()

        json_obj: dict = app_nav_graph_to_json_obj(app_nav_graph)

        logger.debug("json_obj from graph")
        logger.debug(get_pretty_json_str(json_obj))

        state_transition_graph_json_file_path = os.path.join(
            TestAppNavGraph.test_data_directory,
            'larger_deterministic_state_transition_graph.json'
        )

        # with open(state_transition_graph_json_file_path, 'w') as fout:
        #    fout.write(get_pretty_json_str(json_obj) + "\n")

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
