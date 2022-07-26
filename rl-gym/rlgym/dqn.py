import typing
import os
from pathlib import Path
import gym
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tqdm import trange  # type: ignore
from .agents import Experience, AgentInEnvironment


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


class QLearningAgent:
    def __init__(self, q_network: tf.keras.Model):
        self.Q = q_network

    def select_action(
        self, observation: npt.NDArray, epsilon: float = 0.0
    ) -> npt.NDArray[np.int64]:
        return select_action_epsilon_greedily(
            self.Q(observation), epsilon=epsilon
        )


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.size = 0
        self.cursor = 0

    def add(self, experience: Experience) -> None:
        if self.size == 0:
            self._initialize_from_example(experience)
        else:
            self._add_another(experience)

        if self.size < self.maxlen:
            self.size += 1
        self.cursor = (self.cursor + 1) % self.maxlen

        assert self.size <= self.maxlen
        assert self.cursor <= self.size

    def sample(self, n: int) -> Experience:
        indices = np.random.choice(self.size, n, replace=False)
        return self[indices]

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: typing.Union[int, npt.NDArray[np.int64]]
    ) -> Experience:
        idx = np.array(idx)
        if (idx < 0).any() or (idx >= self.size).any():
            raise IndexError(f"Index out of bounds: {idx}")
        return Experience(
            self._o[idx],
            self._a[idx],
            self._r[idx],
            self._no[idx],
            self._t[idx],
        )

    def save(self, filename: typing.Union[str, os.PathLike]) -> None:
        np.savez_compressed(
            filename,
            maxlen=self.maxlen,
            size=self.size,
            cursor=self.cursor,
            obs=self._o,
            action=self._a,
            reward=self._r,
            next_obs=self._no,
            terminated=self._t,
        )

    @classmethod
    def restore(
        cls, filename: typing.Union[str, os.PathLike]
    ) -> "ReplayBuffer":
        npz = np.load(filename)
        buf = cls(maxlen=int(npz["maxlen"]))
        buf.size = int(npz["size"])
        buf.cursor = int(npz["cursor"])
        buf._o = npz["obs"]
        buf._a = npz["action"]
        buf._r = npz["reward"]
        buf._no = npz["next_obs"]
        buf._t = npz["terminated"]
        return buf

    def _init_one_array(self, el) -> npt.NDArray:
        return np.repeat([el], self.maxlen, axis=0)

    def _initialize_from_example(self, experience: Experience) -> None:
        assert self.size == 0  # private method, so can expect this
        obs, action, reward, next_obs, terminated = experience
        self._o = self._init_one_array(obs)
        self._a = self._init_one_array(action)
        self._r = self._init_one_array(reward)
        self._no = self._init_one_array(next_obs)
        self._t = self._init_one_array(terminated)

    def _add_another(self, experience: Experience) -> None:
        assert self.size > 0
        obs, action, reward, next_obs, terminated = experience

        self._o[self.cursor] = obs
        self._a[self.cursor] = action
        self._r[self.cursor] = reward
        self._no[self.cursor] = next_obs
        self._t[self.cursor] = terminated


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


class QAgentInEnvironment(AgentInEnvironment):
    def __init__(
        self,
        env: gym.Env,
        Q_builder: typing.Callable[[], tf.keras.Model],
        memory_size: int,
        target_update_alpha: float = 0.1,
        checkpoint_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
        epsilon: float = 0.0,
    ):
        super().__init__(env)
        self.Q = Q_builder()
        self.Q_target = Q_builder()
        self.memory = ReplayBuffer(maxlen=memory_size)
        self.alpha = target_update_alpha
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_dir = checkpoint_dir
        self.epsilon = epsilon

        # Fully update Q_target to begin with (alpha=1.0)
        soft_update(target=self.Q_target, online=self.Q, alpha=1.0)
        self.reset_env()

        # Restore weights if necessary
        if self.checkpoint_dir is not None:
            self.buffer_path = Path(self.checkpoint_dir) / "replay_buffer.npz"
            self._restore_model_from_checkpoint()
            self._restore_replay_buffer()

    def _restore_model_from_checkpoint(self) -> None:
        ckpt = tf.train.Checkpoint(
            Q=self.Q, Q_target=self.Q_target, optimizer=self.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            ckpt, self.checkpoint_dir, max_to_keep=3
        )
        if self.checkpoint_manager.latest_checkpoint:
            ckpt.restore(
                self.checkpoint_manager.latest_checkpoint
            ).expect_partial()
            print(
                "Restored model weights from checkpoint:",
                self.checkpoint_manager.latest_checkpoint,
            )
        else:
            print(
                "Initializing from scratch, no checkpoint found at dir:",
                self.checkpoint_dir,
            )

    def _restore_replay_buffer(self) -> None:
        if self.buffer_path.exists():
            self.memory.restore(self.buffer_path)
            print("Restored replay buffer from:", self.buffer_path)
        else:
            print(
                "Replay buffer empty, no saved buffer found at:",
                self.buffer_path,
            )

    def select_action(self) -> int:
        # TODO: deal with observations of other shapes (e.g. 2/3D)
        action_values = tf.squeeze(self.Q(self._obs.reshape([1, -1]))).numpy()
        action = select_action_epsilon_greedily(action_values, self.epsilon)
        return int(action)

    def train_step(self, gamma: float, batch_size: int) -> None:
        # Sample batch of experience
        obs, action, reward, next_obs, terminated = self.memory.sample(
            batch_size
        )

        # Calculate TD error (squared) and its gradient wrt Q weights
        td_target = _td_target(
            self.Q_target, self.Q, reward, next_obs, terminated, gamma
        )
        with tf.GradientTape() as tape:
            q_est = tf.gather(self.Q(obs), action, axis=1, batch_dims=1)
            loss = tf.reduce_sum(tf.square(td_target - q_est))
        grads = tape.gradient(loss, self.Q.trainable_weights)

        # Update online network using SGD
        self.optimizer.apply_gradients(zip(grads, self.Q.trainable_weights))

        # Update target network using soft update
        soft_update(target=self.Q_target, online=self.Q, alpha=self.alpha)

    def learn(
        self,
        epsilon_schedule: typing.Callable[[int], float],
        gamma: float,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
    ) -> None:
        total_step = 0

        for epoch in range(1, epochs + 1):
            self.history.on_epoch_begin()
            with trange(steps_per_epoch, ascii=" =") as step_iter:
                step_iter.set_description(f"Epoch {epoch:2d}/{epochs:2d}")
                for step in step_iter:
                    self.epsilon = epsilon_schedule(total_step)
                    step_iter.set_postfix(epsilon=self.epsilon)
                    self.memory.add(self.collect_experience())
                    total_step += 1
                    if len(self.memory) > batch_size:
                        self.train_step(gamma, batch_size)

            print(self.history.epoch_stats_string())
            if self.checkpoint_dir is not None:
                assert hasattr(self, "checkpoint_manager")
                assert hasattr(self, "buffer_path")
                self.checkpoint_manager.save()
                self.memory.save(self.buffer_path)
                print("Saved checkpoint and replay buffer.")
            print()


def mlp_q_agent_builder(
    env: gym.Env,
    hidden_layers: typing.Sequence[int],
    memory_size: int,
    target_update_alpha: float = 0.1,
    checkpoint_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
    epsilon: float = 0.0,
) -> QAgentInEnvironment:
    agent = QAgentInEnvironment(
        env,
        lambda: mlp_q_network(env, hidden_layers),
        memory_size,
        target_update_alpha,
        checkpoint_dir,
        epsilon,
    )
    return agent
