import typing
import os
import gym
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tqdm import trange  # type: ignore
from .agents import Experience, LearningAgent
from .experiences import create_or_restore_replay_buffer


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


def _td_target(
    Q_target: typing.Callable[[npt.NDArray], tf.Tensor],
    Q_online: typing.Callable[[npt.NDArray], tf.Tensor],
    reward: npt.NDArray[np.float64],
    next_obs: npt.NDArray,
    terminated: npt.NDArray[np.bool_],
    gamma: float,
) -> npt.NDArray[np.float64]:
    greedy_next_action = np.argmax(Q_online(next_obs), axis=1, keepdims=True)
    next_action_value = np.take_along_axis(
        Q_target(next_obs).numpy(), greedy_next_action, axis=1
    )
    return reward + gamma * (~terminated) * next_action_value


def soft_update(
    target: tf.keras.Model, online: tf.keras.Model, alpha: float
) -> None:
    """Soft updates a target network's weights towards online network weights.

    `alpha` is the smoothing parameter. Zero corresponds to no update, one
    corresponds to completely replacing the original weights.
    """
    new_weights = [
        alpha * w_o + (1.0 - alpha) * w_t
        for w_o, w_t in zip(online.get_weights(), target.get_weights())
    ]
    target.set_weights(new_weights)


class QAgentInEnvironment(LearningAgent):
    def __init__(
        self,
        env: gym.Env,
        Q_builder: typing.Callable[[], tf.keras.Model],
        checkpoint_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
        epsilon: float = 0.0,
    ):
        super().__init__(env)
        self.Q = Q_builder()
        self.Q_target = Q_builder()
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_dir = checkpoint_dir
        self.epsilon = epsilon

        # Fully update Q_target to begin with (alpha=1.0)
        soft_update(target=self.Q_target, online=self.Q, alpha=1.0)
        self.reset_env()

        # Restore weights if necessary
        if self.checkpoint_dir is not None:
            self._restore_model_from_checkpoint(
                self.checkpoint_dir,
                Q=self.Q,
                Q_target=self.Q_target,
                optimizer=self.optimizer,
            )

    def select_action(self) -> int:
        # TODO: deal with observations of other shapes (e.g. 2/3D)
        action_values = tf.squeeze(self.Q(self._obs.reshape([1, -1]))).numpy()
        action = select_action_epsilon_greedily(action_values, self.epsilon)
        return int(action)

    def train_step(
        self,
        gamma: float,
        experiences: Experience,
        target_update_alpha: float,
    ) -> None:
        # Sample batch of experience
        obs, action, reward, next_obs, terminated = experiences

        # Calculate TD error (squared) and its gradient wrt Q weights
        td_target = _td_target(
            self.Q_target, self.Q, reward, next_obs, terminated, gamma
        )
        with tf.GradientTape() as tape:
            q_est = tf.gather(self.Q(obs), action, axis=1, batch_dims=1)
            td_error = td_target - q_est
            loss = tf.reduce_sum(
                tf.minimum(0.5 * tf.square(td_error), tf.abs(td_error))
            )
        grads = tape.gradient(loss, self.Q.trainable_weights)

        # Update online network using SGD
        self.optimizer.apply_gradients(zip(grads, self.Q.trainable_weights))

        # Update target network using soft update
        soft_update(
            target=self.Q_target, online=self.Q, alpha=target_update_alpha
        )

    def learn(
        self,
        epsilon_schedule: typing.Callable[[int], float],
        gamma: float,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        memory_size: int,
        target_update_alpha: float = 0.1,
    ) -> None:
        total_step = 0
        memory, replay_buffer_path = create_or_restore_replay_buffer(
            memory_size, self.checkpoint_dir
        )

        for epoch in range(1, epochs + 1):
            print()
            self.history.on_epoch_begin()
            with trange(steps_per_epoch, ascii=" =") as step_iter:
                step_iter.set_description(f"Epoch {epoch:2d}/{epochs:2d}")
                for step in step_iter:
                    self.epsilon = epsilon_schedule(total_step)
                    step_iter.set_postfix(epsilon=self.epsilon)
                    memory.add(self.collect_experience())
                    total_step += 1
                    if len(memory) > batch_size:
                        exps = memory.sample(batch_size)
                        self.train_step(gamma, exps, target_update_alpha)

            print(self.history.epoch_stats_string())
            if self.checkpoint_dir is not None:
                assert hasattr(self, "checkpoint_manager")
                self.checkpoint_manager.save()
                assert replay_buffer_path is not None
                memory.save(replay_buffer_path)
                print("Saved checkpoint and replay buffer.")


def mlp_q_agent_builder(
    env: gym.Env,
    hidden_layers: typing.Sequence[int],
    checkpoint_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
    epsilon: float = 0.0,
) -> QAgentInEnvironment:
    agent = QAgentInEnvironment(
        env,
        lambda: mlp_q_network(env, hidden_layers),
        checkpoint_dir,
        epsilon,
    )
    return agent
