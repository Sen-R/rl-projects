import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import tensorflow as tf
from rlgym.learning_utils import EpsilonSchedule, soft_update


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


@pytest.mark.parametrize(
    "alpha,expected", [(0.0, 0.5), (1.0, 1.5), (0.2, 0.7)]
)
def test_soft_update(alpha: float, expected: float) -> None:
    # Prepare trivial (single-parameter) target and online networks
    # with prespecified weights
    target = tf.keras.Sequential(
        tf.keras.layers.Dense(1, use_bias=False, input_shape=(1,))
    )
    target.set_weights([np.array([[0.5]])])

    online = tf.keras.Sequential(
        tf.keras.layers.Dense(1, use_bias=False, input_shape=(1,))
    )
    online.set_weights([np.array([[1.5]])])

    # Perform soft-update with alpha and check online is unchanged and
    # target changes correctly
    soft_update(target, online, alpha)
    assert_almost_equal(online.get_weights()[0], 1.5)
    assert_almost_equal(target.get_weights()[0], expected)
