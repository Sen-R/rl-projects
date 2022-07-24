import typing
from collections import namedtuple
import gym
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tqdm import trange  # type: ignore


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
    all_layers = [tf.keras.layers.Dense(n_units) for n_units in hidden_layers]
    all_layers.append(tf.keras.layers.Dense(n_actions))
    model = tf.keras.Sequential(all_layers)
    model.build(input_shape=[None, *obs_shape])
    return model


def select_action_epsilon_greedily(
    action_values: npt.NDArray[np.float64], epsilon: float
) -> npt.NDArray[np.int64]:
    av_shape = action_values.shape
    n_actions = av_shape[-1]
    explore_shape = 1 if len(av_shape) == 1 else av_shape[0]

    explore = np.random.binomial(1, epsilon, explore_shape)
    best_action = np.argmax(action_values, axis=-1)
    random_action = np.random.choice(n_actions, size=explore_shape)

    return np.where(explore, random_action, best_action)


class EpsilonSchedule:
    def __init__(self, start: float, rampdown_length: int, end: float):
        self.start = start
        self.end = end
        self.slope = (end - start) / float(rampdown_length)

    def __call__(self, step: int) -> float:
        return max(self.start + step * self.slope, self.end)


class QLearningAgent:
    def __init__(self, q_network: tf.keras.Model):
        self.Q = q_network

    def select_action(
        self, observation: npt.NDArray, epsilon: float = 0.0
    ) -> npt.NDArray[np.int64]:
        return select_action_epsilon_greedily(
            self.Q(observation), epsilon=epsilon
        )


class TrainingProgress:
    def on_epoch_begin(self) -> None:
        self._episode_rewards_this_epoch: typing.List[float] = []
        self._episode_lengths_this_epoch: typing.List[int] = []

    def on_episode_end(self, episode_length: int, total_reward: float) -> None:
        self._episode_rewards_this_epoch.append(total_reward)
        self._episode_lengths_this_epoch.append(episode_length)

    def get_epoch_stats(self) -> typing.Dict[str, typing.Union[int, float]]:
        return {
            "num_episodes": len(self._episode_rewards_this_epoch),
            "av_episode_length": float(
                np.mean(self._episode_lengths_this_epoch)
            ),
            "av_reward_per_episode": float(
                np.mean(self._episode_rewards_this_epoch)
            ),
        }

    def epoch_stats_string(self) -> str:
        stats = self.get_epoch_stats()
        return " | ".join(
            (
                k + ": " + (f"{v:.2f}" if isinstance(v, float) else f"{v}")
                for k, v in stats.items()
            )
        )


Experience = namedtuple(
    "Experience", ["obs", "action", "reward", "next_obs", "terminated"]
)


class QAgentInEnvironment:
    def __init__(self, env: gym.Env, Q: tf.keras.Model):
        self.env = env
        self.Q = Q
        self.history = TrainingProgress()
        self.reset_env()

    def reset_env(self) -> None:
        self._obs: npt.NDArray = self.env.reset()  # type: ignore
        self._episode_step = 0
        self._episode_reward = 0.0

    def select_action(self, epsilon: float) -> int:
        # TODO: deal with observations of other shapes (e.g. 2/3D)
        action_values = tf.squeeze(self.Q(self._obs.reshape([1, -1]))).numpy()
        action = select_action_epsilon_greedily(action_values, epsilon)
        return int(action)

    def collect_experience(self, epsilon: float) -> Experience:
        action = self.select_action(epsilon)
        next_obs, reward, terminated, truncated, _ = self.env.step(
            action
        )  # type: ignore
        exp = Experience(self._obs, action, reward, next_obs, terminated)

        self._episode_step += 1
        self._episode_reward += reward

        if terminated or truncated:
            self.history.on_episode_end(
                self._episode_step, self._episode_reward
            )
            self.reset_env()
        else:
            self._obs = next_obs

        return exp

    def _td_target(
        self,
        reward: float,
        next_obs: npt.NDArray,
        terminated: bool,
        gamma: float,
    ) -> typing.Union[npt.NDArray[np.float64], float]:
        if terminated:
            td_target = reward
        else:
            td_target = reward + gamma * np.max(
                self.Q(next_obs.reshape([1, -1])).numpy()
            )
        return td_target

    def learn(
        self,
        epochs: int,
        steps_per_epoch: int,
        epsilon_schedule: typing.Callable[[int], float],
        gamma: float,
    ) -> None:
        optimizer = tf.keras.optimizers.Adam()
        total_step = 0

        for epoch in range(epochs):
            self.history.on_epoch_begin()

            with trange(steps_per_epoch, ascii=" =") as step_iter:
                step_iter.set_description(f"Epoch {epoch:2d}/{epochs:2d}")
                for step in step_iter:
                    epsilon = epsilon_schedule(total_step)
                    step_iter.set_postfix(epsilon=epsilon)
                    experience = self.collect_experience(epsilon)
                    total_step += 1

                    obs, action, reward, next_obs, truncated = experience
                    td_target = self._td_target(
                        reward, next_obs, truncated, gamma
                    )

                    with tf.GradientTape() as tape:
                        q_est = tf.gather(
                            self.Q(self._obs.reshape([1, -1])),
                            [action],
                            axis=1,
                            batch_dims=1,
                        )
                        loss = tf.square(td_target - q_est)

                    grads = tape.gradient(loss, self.Q.trainable_weights)
                    optimizer.apply_gradients(
                        zip(grads, self.Q.trainable_weights)
                    )

            print(self.history.epoch_stats_string() + "\n")
