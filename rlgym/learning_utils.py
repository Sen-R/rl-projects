import tensorflow as tf


class EpsilonSchedule:
    def __init__(self, start: float, rampdown_length: int, end: float):
        self.start = start
        self.end = end
        self.slope = (end - start) / float(rampdown_length)

    def __call__(self, step: int) -> float:
        return max(self.start + step * self.slope, self.end)


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
