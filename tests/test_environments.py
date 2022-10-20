import pytest
from numpy.testing import assert_almost_equal
import gym
from rlgym.environments import soft_bounding_penalty


@pytest.mark.parametrize(
    "x,penalty",
    [
        (-1.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.5, 0.125),
        (3.0, 1.0),
        (4.0, 1.0),
    ],
)
def test_soft_bounding_penalty(x: float, penalty: float) -> None:
    assert_almost_equal(soft_bounding_penalty(x, 2.0, 3.0, 3.0), penalty)


class TestDenseCartPole:
    def test_functionality(self) -> None:
        env = gym.make(
            "rlgym.environments:DenseCartPole-v1",
            render_mode="human",
        )
        env.reset()
        _, r, _, _, _ = env.step(env.action_space.sample())  # type: ignore
        assert r < 1.0
