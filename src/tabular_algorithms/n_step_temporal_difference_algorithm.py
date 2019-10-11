import numpy as np

from tabular_algorithms.model_free_tabular_prediction_algorithm_base import ModelFreeTabularPredictionAlgorithmBase


class NStepTemporalDifferenceAlgorithm(ModelFreeTabularPredictionAlgorithmBase):
    """
    Defines N-step temporal difference prediction algorithm.
    """

    def __init__(
        self, num_steps, gamma, learning_rate_fcn, defaulit_state_value_fcn_value
    ):
        super(NStepTemporalDifferenceAlgorithm, self).__init__(
            gamma, learning_rate_fcn, defaulit_state_value_fcn_value
        )

        self.num_steps = num_steps
        self.discount_factor_power_array = None

        self.__initialize()

    def __initialize(self):
        self.discount_factor_power_array = np.power(
            self.gamma, np.arange(self.num_steps + 1)
        )

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
            state_list = list()
            reward_list = list()
            state, _ = env.reset()

            not_terminated = not env.is_terminal_state()

            state_list.append(state)
            for t in range(max_num_transitions_per_episode):
                if debug_mode:
                    print(f"t = {t}")

                if not_terminated:
                    action = policy_sampler.get_action(state)
                    next_state, reward, is_terminal_state, _ = env.apply_action(action)

                    reward_list.append(reward)

                    if env.is_terminal_state():
                        not_terminated = False
                    else:
                        state_list.append(next_state)
                        state = next_state

                    if debug_mode:
                        print(f"state_list: {state_list}")
                        print(f"reward_list: {reward_list}")

                if t >= self.num_steps - 1:
                    n_reward_array = np.array(
                        reward_list[t-self.num_steps+1:t+1]
                    )

                    if n_reward_array.size == 0:
                        break

                    G = (
                        n_reward_array
                        * self.discount_factor_power_array[: n_reward_array.size]
                    ).sum()

                    tail_state_str = ""
                    if t + 1 < len(state_list):
                        tail_state = state_list[t + 1]
                        G += (
                            self.discount_factor_power_array[-1]
                            * self.state_value_fcn_dict[tail_state]
                        )
                        tail_state_str = f", tail state: {str(tail_state)}"

                    state_updated = state_list[t - self.num_steps + 1]

                    if debug_mode:
                        print(
                            f"\tupdated state: {state_updated}, G: {G}{tail_state_str}"
                        )

                    learning_rate = self.get_learning_rate(iter_num, episode_num)

                    current_state_value_fcn_value = self.state_value_fcn_dict[
                        state_updated
                    ]
                    self.state_value_fcn_dict[state_updated] += learning_rate * (
                        G - current_state_value_fcn_value
                    )

                    if debug_mode:
                        print(
                            f"\tprev. val: {current_state_value_fcn_value:.2f}",
                            end=", ",
                        )
                        print(f"G: {G}", end=", ")
                        print(
                            f"updated val: {self.state_value_fcn_dict[state_updated]:.2f}"
                        )

                    iter_num += 1

            if iter_num > max_num_iters:
                break

            if does_record_history:
                self.record_history(episode_num)

            if iter_num > max_num_iters:
                break


if __name__ == "__main__":

    def get_rms_error(random_walk_env, state_value_fcn_dict):
        """
        Returns the root mean square (RMS) error between state-value functions and true solution

        Parameters
        ----------
        random_walk_env: RandomWalkEnvironment
            random walk environment
        state_value_fcn_dict: dict
            stats-value function values
        """

        num_states = random_walk_env.get_num_nodes()

        state_value_array = np.zeros(num_states)
        for state, value in state_value_fcn_dict.items():
            state_value_array[state - 1] = value

        true_state_value_array = np.linspace(
            random_walk_env.get_leftmost_reward(),
            random_walk_env.get_rightmost_reward(),
            num_states + 2,
        )

        err_array = true_state_value_array[1:-1] - state_value_array

        return np.linalg.norm(err_array) / np.sqrt(err_array.size)

    from environment.random_walk_environment import RandomWalkEnvironment
    from policy.equally_probable_random_policy_sampler import EquallyProbableRandomPolicySampler

    gamma = 1.0
    num_plays = 10
    debug_mode = True
    defaulit_state_value_fcn_value = 0.0

    random_policy = EquallyProbableRandomPolicySampler(("left", "right"))

    # td0 = TemporalDifference0Alg(1.0, 0.1, 0.5)

    # test = "rms_with_alpha_sweep"
    test = "draw_state_values"

    if test == "rms_with_alpha_sweep":
        import pickle

        num_states = 19
        num_repetitions = 200
        # num_alpha_split = 11
        num_alpha_split = 21
        max_num_episodes = 10
        max_num_transitions_per_episode = 1000
        # num_steps_list = [1, 2, 4, 8, 16]
        # num_steps_list = [1, 2, 4, 8, 16, 32, 64]
        # num_steps_list = [2, 4, 8, 16, 32, 64]
        # num_steps_list = [2, 4]
        num_steps_list = []

        random_walk_environment = RandomWalkEnvironment(num_states, -1.0, 1.0)

        alpha_array = np.linspace(0.0, 1.0, num_alpha_split)

        for num_steps in num_steps_list:
            print(f"The number of TD steps: {num_steps}")

            mean_rms_error_list = list()

            for alpha in alpha_array:
                print(f"alpha: {alpha:.2f}: ", end="")
                learning_rate_strategy = alpha
                tdn = NStepTemporalDifferenceAlgorithm(
                    num_steps,
                    gamma,
                    learning_rate_strategy,
                    defaulit_state_value_fcn_value,
                )

                rms_error_list = list()
                for repeat_num in range(num_repetitions):
                    tdn.reset_state_value_fcn_dict()
                    for episode_num in range(max_num_episodes):
                        tdn.predict(
                            random_walk_environment,
                            random_policy,
                            1,
                            1 * max_num_transitions_per_episode,
                            max_num_transitions_per_episode,
                        )

                        rms_error = get_rms_error(
                            random_walk_environment, tdn.get_state_value_fcn_dict()
                        )
                        rms_error_list.append(rms_error)

                mean_rms_error_list.append(np.array(rms_error_list).mean())
                print(f"Mean RMS error: {mean_rms_error_list[-1]:.2f}")

            save_obj = (alpha_array, mean_rms_error_list)
            save_file_name = (
                f"n_step_test_results_with_random_walk_with_n_step_{num_steps}.pkl"
            )
            with open(save_file_name, "wb") as fid:
                pickle.dump(save_obj, fid)

        import os
        import re

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()

        for file_name in os.listdir(os.curdir):
            m = re.match(
                r"^n_step_test_results_with_random_walk_with_n_step_(\d+).pkl$",
                file_name,
            )
            if m:
                print(f"reading `{file_name}`")
                num_steps = int(m.group(1))

                with open(file_name, "rb") as fid:
                    save_obj = pickle.load(fid)

                alpha_array, mean_rms_error_list = save_obj

                ax.plot(alpha_array, mean_rms_error_list, label=f"n = {num_steps}")

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"Average RMS errors")
        ax.set_ylim((0.25, 0.55))
        ax.legend()
        fig.show()

    elif test == "draw_state_values":
        num_steps = 4
        num_states = 5

        random_walk_environment = RandomWalkEnvironment(num_states, -1.0, 1.0)

        print(f"The number of TD steps: {num_steps}")

        diminishing_step_size = True

        if diminishing_step_size:

            def learning_rate_strategy(iter_num, episode_num):
                return 0.1 / (1.0 + episode_num * 0.1)

        else:
            learning_rate_strategy = 0.2

        tdn = NStepTemporalDifferenceAlgorithm(num_steps, gamma, learning_rate_strategy, 0.5)

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()

        max_num_episodes = 100
        max_num_transitions_per_episode = 100

        total_num_episodes = 0
        for idx in range(num_plays):
            tdn.predict(
                random_walk_environment,
                random_policy,
                max_num_episodes,
                max_num_episodes * max_num_transitions_per_episode * 10,
                max_num_transitions_per_episode,
                debug_mode=debug_mode,
            )

            total_num_episodes += max_num_episodes

            state_value_fcn_dict = tdn.get_state_value_fcn_dict()
            # print(get_pretty_json_str(state_value_fcn_dict))
            print(
                f"RMS error: {get_rms_error(random_walk_environment, state_value_fcn_dict):.2f}"
            )

            random_walk_environment.draw_state_value_fcn_values(
                ax,
                state_value_fcn_dict,
                "o-",
                label=total_num_episodes,
                alpha=((idx + 1.0) / num_plays) ** 2,
            )

        ax.legend()

        fig.show()
    else:
        raise ValueError(test)
