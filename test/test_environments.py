import unittest

from environment.random_walk_environment import RandomWalkEnvironment


class TestEnvironments(unittest.TestCase):
    def test_random_walk_environment(self):

        random_walk_environment: RandomWalkEnvironment = RandomWalkEnvironment(5)

        random_walk_environment.reset()

        self.assertEqual(random_walk_environment.reset(), (3, None))
        self.assertEqual(random_walk_environment.apply_action("right"), (4, 0.0, False, None))
        self.assertEqual(random_walk_environment.apply_action("right"), (5, 0.0, False, None))
        self.assertEqual(random_walk_environment.apply_action("right"), (6, 1.0, True, None))

        self.assertEqual(random_walk_environment.reset(), (3, None))
        self.assertEqual(random_walk_environment.apply_action("left"), (2, 0.0, False, None))
        self.assertEqual(random_walk_environment.apply_action("left"), (1, 0.0, False, None))
        self.assertEqual(random_walk_environment.apply_action("left"), (0, 0.0, True, None))

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
