from typing import Dict, Any
import unittest
import os
import logging
import json

from utils import set_logging_basic_config
from environment.deterministic_directed_graph_environment import (
    DeterministicDirectedGraphEnvironment,
    create_deterministic_directed_graph_environment_from_json_obj,
)


class TestReadStateTransitionGraphInJsonFormat(unittest.TestCase):
    test_data_directory = os.path.join(os.curdir, "data")

    simple_deterministic_state_transition_graph_json_file_path = os.path.join(
        test_data_directory, "simple_deterministic_state_transition_graph.json"
    )

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__, logging.INFO)

    def test_read_simple_graph(self):
        deterministic_directed_graph_environment_json_dict: Dict[str, Any]
        with open(
            TestReadStateTransitionGraphInJsonFormat.simple_deterministic_state_transition_graph_json_file_path
        ) as fin:
            deterministic_directed_graph_environment_json_dict = json.load(fin)

        deterministic_directed_graph_environment: DeterministicDirectedGraphEnvironment = (
            create_deterministic_directed_graph_environment_from_json_obj(
                deterministic_directed_graph_environment_json_dict
            )
        )

        self.assertEqual(deterministic_directed_graph_environment.start_state, 0)
        self.assertEqual(deterministic_directed_graph_environment.goal_state, 3)

        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[0][0],
            1,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[0][1],
            4,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[1][0],
            2,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[1][1],
            5,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[2][0],
            3,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[2][1],
            6,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[4][0],
            1,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[4][1],
            5,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[5][0],
            2,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[5][1],
            6,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_graph_dict[6][0],
            3,
        )

        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[0][0],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[0][1],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[1][0],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[1][1],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[2][0],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[2][1],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[4][0],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[4][1],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[5][0],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[5][1],
            -1.0,
        )
        self.assertEqual(
            deterministic_directed_graph_environment.state_transition_reward_dict[6][0],
            -1.0,
        )

        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    0
                ][0],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    0
                ][1],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    1
                ][0],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    1
                ][1],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    2
                ][0],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    2
                ][1],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    4
                ][0],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    4
                ][1],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    5
                ][0],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    5
                ][1],
                float,
            )
        )
        self.assertTrue(
            isinstance(
                deterministic_directed_graph_environment.state_transition_reward_dict[
                    6
                ][0],
                float,
            )
        )


if __name__ == "__main__":
    unittest.main()
