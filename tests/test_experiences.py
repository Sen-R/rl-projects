import pytest
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal
from rlgym.agents import Experience
from rlgym.experiences import ReplayBuffer


class TestReplayBuffer:
    def test_initial_state(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        assert len(buf) == 0

    def test_insert_one_and_retrieve_the_same(self) -> None:
        buf = ReplayBuffer(maxlen=1)
        buf.add(Experience(1, 2, 3, 4, True))
        assert len(buf) == 1
        assert buf.cursor == 0  # maxlen is only one
        assert buf[0] == Experience(1, 2, 3, 4, True)

    def test_insert_second_and_retrieve_both(self) -> None:
        buf = ReplayBuffer(maxlen=3)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.add(Experience(5, 6, 7, 8, False))
        assert len(buf) == 2
        assert buf.cursor == 2
        assert buf[0] == Experience(1, 2, 3, 4, True)
        assert buf[1] == Experience(5, 6, 7, 8, False)

    def test_insert_over_maxlen_goes_back_to_start(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.add(Experience(5, 6, 7, 8, False))
        buf.add(Experience(9, 0, 1, 2, True))
        assert len(buf) == 2
        assert buf.cursor == 1
        assert buf[0] == Experience(9, 0, 1, 2, True)
        assert buf[1] == Experience(5, 6, 7, 8, False)

    def test_getitem_using_multiple_indices(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        buf.add(Experience([0, 1], 0, 1.0, [2, 3], False))
        buf.add(Experience([2, 3], 1, 0.0, [4, 5], True))
        o, a, r, no, t = buf[np.array([1, 0])]
        assert_array_equal(o, [[2, 3], [0, 1]])
        assert_array_equal(a, [1, 0])
        assert_array_equal(r, [0.0, 1.0])
        assert_array_equal(no, [[4, 5], [2, 3]])
        assert_array_equal(t, [True, False])

    def test_sample_returns_correct_number_of_samples(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.add(Experience(5, 6, 7, 8, False))
        o, a, r, no, t = buf.sample(1)
        assert len(o) == len(a) == len(r) == len(no) == len(t) == 1

    def test_getitem_raises_when_index_out_of_bounds(self) -> None:
        buf = ReplayBuffer(maxlen=1)
        buf.add(Experience(1, 2, 3, 4, True))
        with pytest.raises(IndexError):
            buf[1]

    def test_sample_raises_when_asking_for_too_many(self) -> None:
        buf = ReplayBuffer(maxlen=1)
        buf.add(Experience(1, 2, 3, 4, True))
        with pytest.raises(ValueError):
            buf.sample(2)

    def test_save(self, tmp_path: Path) -> None:
        buf = ReplayBuffer(maxlen=1)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.save(tmp_path / "buffer.npz")
        assert (tmp_path / "buffer.npz").exists()

    def test_restore(self, tmp_path: Path) -> None:
        buf = ReplayBuffer(maxlen=2)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.save(tmp_path / "buffer.npz")
        del buf

        restored = ReplayBuffer.restore(tmp_path / "buffer.npz")
        assert len(restored) == 1
        assert restored[0] == Experience(1, 2, 3, 4, True)
        assert restored.maxlen == 2
        assert restored.cursor == 1
