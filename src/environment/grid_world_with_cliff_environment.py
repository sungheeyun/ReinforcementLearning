from environment.grid_world_environment import GridWorld


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
