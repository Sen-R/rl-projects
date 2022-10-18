import typing
import numpy as np
import gym
from gym.envs.registration import register


class CartPoleDenseWrapper(gym.Wrapper):
    """
    Dense reward signal for CartPoleEnvironment.

    Adds penalties for moving off-centre, off-vertical or for non-zero
    translational or angular velocities.

    Args:
      reward_weights: tuple of form `[x_w, v_w, theta_w, omega_w]` that
        is dotted with the absolute value of the state vector to form
        the dense penalty signal described above.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_weights: typing.Tuple[float, float, float, float],
        new_step_api: bool,
    ) -> None:
        super().__init__(env, new_step_api=new_step_api)
        if len(reward_weights) != 4:
            raise ValueError(
                f"Reward weights should be length 4, got: {reward_weights}"
            )
        self._weights = np.array(reward_weights)

    def step(self, action: int):
        obs, reward, term, trunc, info = self.env.step(action)  # type: ignore
        reward -= np.dot(self._weights, np.abs(obs))
        return obs, reward, term, trunc, info


_def_weights = (0.1, 0.1, 0.1, 0.1)


def cartpole_dense(
    reward_weights: typing.Tuple[float, float, float, float] = _def_weights,
    new_step_api: bool = True,
    **kwargs,
) -> gym.Env:
    env = gym.make("CartPole-v1", new_step_api=new_step_api, **kwargs)
    env = CartPoleDenseWrapper(env, reward_weights, new_step_api=new_step_api)
    return env


register(
    id="DenseCartPole-v1",
    entry_point="rlgym.environments:cartpole_dense",
)
