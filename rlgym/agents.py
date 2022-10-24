from abc import ABC, abstractmethod
from collections import namedtuple
import typing
import os
import numpy.typing as npt
import numpy as np
import tensorflow as tf
import gym


Experience = namedtuple(
    "Experience", ["obs", "action", "reward", "next_obs", "terminated"]
)


class TrainingProgress:
    def __init__(self):
        self.reset_epoch_stats()

    def on_epoch_begin(self) -> None:
        self.reset_epoch_stats()

    def reset_epoch_stats(self) -> None:
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


class AgentInEnvironment(ABC):
    """Base class for agents."""

    def __init__(self, env: gym.Env):
        self._env = env
        self._history = TrainingProgress()
        self.reset_env()

    def reset_env(self) -> None:
        self._obs: npt.NDArray
        self._obs, _ = self.env.reset()
        self._episode_step = 0
        self._episode_reward = 0.0

    @property
    def env(self) -> gym.Env:
        return self._env

    @property
    def history(self) -> TrainingProgress:
        return self._history

    @abstractmethod
    def select_action(self) -> int:
        raise NotImplementedError()

    def collect_experience(self) -> Experience:
        action = self.select_action()
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        experience = Experience(
            self._obs, action, reward, next_obs, terminated
        )

        self._episode_step += 1
        self._episode_reward += reward

        if terminated or truncated:
            self.history.on_episode_end(
                self._episode_step, self._episode_reward
            )
            self.reset_env()
        else:
            self._obs = next_obs

        return experience


class RandomAgentInEnvironment(AgentInEnvironment):
    """Agent that takes random actions."""

    def select_action(self) -> int:
        return self.env.action_space.sample()


class LearningAgent(AgentInEnvironment):
    """Base class for agents that learn."""

    def _restore_model_from_checkpoint(
        self,
        checkpoint_dir: typing.Union[str, os.PathLike],
        **objects_to_checkpoint,
    ) -> None:
        ckpt = tf.train.Checkpoint(**objects_to_checkpoint)
        self.checkpoint_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_dir, max_to_keep=3
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
                checkpoint_dir,
            )
