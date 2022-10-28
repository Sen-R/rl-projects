import typing
import gym
import tensorflow as tf


def _extract_env_obs_and_action_space_sizes(
    env: gym.Env,
) -> typing.Tuple[typing.Any, int]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("Observation space not of type `Box`")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Action space not of type `Discrete`")

    return env.observation_space.shape, env.action_space.n


def linear_q_network(env: gym.Env) -> tf.keras.Model:
    """Returns a linear representation of the action value function q(s, a)."""
    return mlp_q_network(env)


def mlp_q_network(
    env: gym.Env, hidden_layers: typing.Optional[typing.Sequence[int]] = None
) -> tf.keras.Model:
    """Returns a MLP representation of the action value function q(s, a)."""
    if hidden_layers is None:
        hidden_layers = []
    obs_shape, n_actions = _extract_env_obs_and_action_space_sizes(env)
    all_layers = [
        tf.keras.layers.Dense(n_units, activation="relu")
        for n_units in hidden_layers
    ]
    all_layers.append(tf.keras.layers.Dense(n_actions))
    model = tf.keras.Sequential(all_layers)
    model.build(input_shape=[None, *obs_shape])
    return model
