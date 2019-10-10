from environment.environment_base import EnvironmentBase


class DeterministicEnvironment(EnvironmentBase):
    """
    An abstract class representing a deterministic environment, i.e., with the same action and state,
     it always transits to a determined next state.

    BTW, a random environment is an environment where the same action from the same state can lead to different states,
    i.e., the nature of the environment is probabilistic.
    """
    pass
