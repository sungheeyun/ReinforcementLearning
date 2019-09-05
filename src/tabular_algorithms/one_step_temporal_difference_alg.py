""" Classes for tabular temporal differencing prediction methods """

from tabular_algorithms.tabular_rl_algorithms_base import ModelFreeTabularPredictionAlgBase


class OneStepTemporalDifferenceAlg(ModelFreeTabularPredictionAlgBase):
    """
    Implements temporal difference prediction algorithms.
    """

    def __init__(self, gamma, learning_rate_fcn, defaulit_state_value_fcn_value):
        super(OneStepTemporalDifferenceAlg, self).__init__(gamma, learning_rate_fcn, defaulit_state_value_fcn_value)

    def predict(
        self,
        env,
        policy_sampler,
        max_num_episodes,
        max_num_iters,
        max_num_transitions_per_episode,
        does_record_history=False,
        verbose_mode=False,
        debug_mode=False,
    ):
        iter_num = 0

        for episode_num in range(max_num_episodes):
            state, _ = env.reset()

            for _ in range(max_num_transitions_per_episode):
                action = policy_sampler.get_action(state)
                next_state, reward, is_terminal_state, _ = env.apply_action(action)
                learning_rate = self.get_learning_rate(iter_num, episode_num)

                current_state_value_fcn_value = self.state_value_fcn_dict[state]

                if is_terminal_state:
                    next_state_value_fcn_value = 0.0
                else:
                    next_state_value_fcn_value = self.state_value_fcn_dict[next_state]

                self.state_value_fcn_dict[state] += learning_rate * (
                    reward + self.gamma * next_state_value_fcn_value - current_state_value_fcn_value
                )

                if is_terminal_state:
                    break

                iter_num += 1

                if iter_num > max_num_iters:
                    break

                state = next_state

            if does_record_history:
                self.record_history(episode_num)

            if iter_num > max_num_iters:
                break


if __name__ == "__main__":

    from environment.environments import RandomWalkEnvironment
    from policy.policies import EquallyProbableRandomPolicySampler
    from rl_utils.utils import get_pretty_json_str

    random_walk_environment = RandomWalkEnvironment(19)
    random_policy = EquallyProbableRandomPolicySampler(("left", "right"))

    gamma = 1.0
    diminishing_step_size = False

    if diminishing_step_size:

        def learning_rate_strategy(iter_num, episode_num):
            return 0.1 / (1.0 + iter_num * 0.01)

    else:
        learning_rate_strategy = 0.1

    # td0 = TemporalDifference0Alg(1.0, 0.1, 0.5)
    td0 = OneStepTemporalDifferenceAlg(gamma, learning_rate_strategy, 0.5)

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    max_num_episodes = 10
    max_num_transitions_per_episode = 100

    total_num_episodes = 0
    for _ in range(5):
        td0.predict(
            random_walk_environment,
            random_policy,
            max_num_episodes,
            max_num_episodes * 100,
            max_num_transitions_per_episode,
        )

        total_num_episodes += max_num_episodes

        state_value_fcn_dict = td0.get_state_value_fcn_dict()
        print(get_pretty_json_str(state_value_fcn_dict))
        random_walk_environment.draw_state_value_fcn_values(ax, state_value_fcn_dict, "o-", label=total_num_episodes)

    ax.legend()

    fig.show()
