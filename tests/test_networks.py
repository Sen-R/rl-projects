import pytest
import numpy as np
from numpy.testing import assert_array_equal
import gym
import tensorflow as tf
from rlgym.networks import mlp_q_network


@pytest.fixture
def Q(cp_env: gym.Env) -> tf.keras.Model:
    return mlp_q_network(cp_env, [3, 4])


class TestMLPQNetwork:
    def test_maps_env_observations_to_valid_action_values(
        self, cp_env: gym.Env, Q: tf.keras.Model
    ) -> None:
        sample_obs, _ = cp_env.reset()
        sample_output = Q(sample_obs[np.newaxis, :])
        assert isinstance(cp_env.action_space, gym.spaces.Discrete)
        assert_array_equal(sample_output.shape, [1, cp_env.action_space.n])

    def test_hidden_layers_as_expected(self, Q: tf.keras.Model) -> None:
        assert len(Q.layers) == 3
        assert Q.layers[0].units == 3
        assert Q.layers[1].units == 4

    def test_default_is_linear_model(self, cp_env: gym.Env) -> None:
        linear_Q = mlp_q_network(cp_env)
        assert len(linear_Q.layers) == 1
