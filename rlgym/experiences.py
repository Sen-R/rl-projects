import typing
import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
from .agents import Experience


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


def create_or_restore_replay_buffer(
    memory_size: int,
    checkpoint_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
) -> typing.Tuple[ReplayBuffer, typing.Optional[Path]]:
    memory = ReplayBuffer(maxlen=memory_size)
    buffer_path = None
    if checkpoint_dir is not None:
        buffer_path = Path(checkpoint_dir) / "replay_buffer.npz"
        if buffer_path.exists():
            memory = ReplayBuffer.restore(buffer_path)
            assert len(memory) > 0
            print("Restored replay buffer from:", buffer_path)
            print(f"Contains {len(memory)} experiences.")
        else:
            print(
                "Replay buffer empty, no saved buffer found at:",
                buffer_path,
            )
    return memory, buffer_path
