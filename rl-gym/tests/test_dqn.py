import pytest
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
import tensorflow as tf
import gym
from rlgym.dqn import (
    mlp_q_network,
    select_action_epsilon_greedily,
    EpsilonSchedule,
)


@pytest.fixture
def env() -> gym.Env:
    return gym.make("CartPole-v1", new_step_api=True)


@pytest.fixture
def sample_obs(env: gym.Env) -> np.ndarray:
    sample_obs = env.reset()
    assert isinstance(sample_obs, np.ndarray)
    return sample_obs[np.newaxis, :]


@pytest.fixture
def Q(env: gym.Env) -> tf.keras.Model:
    return mlp_q_network(env, [3, 4])


class TestMLPQNetwork:
    def test_maps_env_observations_to_valid_action_values(
        self, env: gym.Env, Q: tf.keras.Model, sample_obs: np.ndarray
    ) -> None:
        sample_output = Q(sample_obs)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert_array_equal(sample_output.shape, [1, env.action_space.n])

    def test_hidden_layers_as_expected(self, Q: tf.keras.Model) -> None:
        assert len(Q.layers) == 3
        assert Q.layers[0].units == 3
        assert Q.layers[1].units == 4

    def test_default_is_linear_model(self, env: gym.Env) -> None:
        linear_Q = mlp_q_network(env)
        assert len(linear_Q.layers) == 1


class TestSelectActionEpsilonGreedily:
    def test_output_scalar_when_input_rank_one(self) -> None:
        action_values = np.array([1.0, 2.0, 0.0])
        action = select_action_epsilon_greedily(action_values, epsilon=0.0)
        assert_array_equal(action, 1)

    def test_output_vector_when_input_rank_two(self) -> None:
        action_values = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
        actions = select_action_epsilon_greedily(action_values, epsilon=0.0)
        assert_array_equal(actions, [1, 2])

    @pytest.mark.parametrize(
        "epsilon,lower,upper",
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.65, 0.71),  # should be about 2/3
            (0.5, 0.30, 0.35),  # should be about 1/3
        ],
    )
    def test_epsilon_statistics(
        self,
        epsilon: float,
        lower: float,
        upper: float,
    ) -> None:
        # Set up a random action_values matrix
        np.random.seed(82432)  # for reproducibility
        n_actions = 3
        n_trials = 1000  # larger makes type I errors less likely but is slower
        action_values = np.random.normal(size=(n_trials, n_actions))

        # Compare epsilon greedy selected actions against best action values
        # and report how often the best action was chosen
        best_action_values = np.max(action_values, axis=-1)
        chosen_actions = select_action_epsilon_greedily(action_values, epsilon)
        chosen_action_values = np.take_along_axis(
            action_values, chosen_actions[:, np.newaxis], axis=1
        ).squeeze()

        nongreedy_action_proportion = 1.0 - np.mean(
            np.isclose(best_action_values, chosen_action_values)
        )

        assert lower <= nongreedy_action_proportion <= upper


@pytest.fixture
def epsilon() -> EpsilonSchedule:
    return EpsilonSchedule(1.0, 9000, 0.1)


class TestEpsilonSchedule:
    def test_starting_value(self, epsilon: EpsilonSchedule) -> None:
        assert_almost_equal(epsilon(0), 1.0)

    def test_long_term_value(self, epsilon: EpsilonSchedule) -> None:
        assert_almost_equal(epsilon(1000000), 0.1)

    @pytest.mark.parametrize("step,expected", [(1000, 0.9), (5000, 0.5)])
    def test_linear_rampdown_in_between(
        self, epsilon: EpsilonSchedule, step: int, expected: float
    ) -> None:
        assert_almost_equal(epsilon(step), expected)
