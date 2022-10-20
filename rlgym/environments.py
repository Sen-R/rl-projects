import numpy as np
import gym
from gym.envs.registration import register


def soft_bounding_penalty(
    x: float, soft_edge: float, hard_edge: float, alpha: float
) -> float:
    """Imposes a soft bounding penalty up to `hard_edge`.

    Output is zero below `soft_edge` and polynomially increases to
    one at `hard_edge`, with the exponent set by `alpha`.
    """
    return (
        np.clip((x - soft_edge) / (hard_edge - soft_edge), 0.0, 1.0) ** alpha
    )


class CartPoleDenseWrapper(gym.Wrapper):
    """
    Dense reward signal for CartPoleEnvironment.

    Modifies the reward from original CartPole as follows:
    - Soft penalty for veering too far from the centre
    - Soft penalty for veering too far from vertical
    - Penalty on angular velocity to guide agent to apply a torque to restore
      balance (i.e. in opposite direction to angular displacement from
      vertical)
    """

    def __init__(
        self,
        env: gym.Env,
        stuck_ops_max: int = 5,
    ) -> None:
        super().__init__(env)

    def step(self, action: int):
        # environment-specific constants
        x_lim = 4.8
        theta_lim = 0.2095

        # reward shaping constants (could be parameterised in future)
        alpha = 2.0
        tip_recovery_aggressiveness = 4.0

        obs, reward, term, trunc, info = self.env.step(action)
        x, v, theta, omega = obs
        target_omega = -tip_recovery_aggressiveness * theta
        reward -= soft_bounding_penalty(abs(x), x_lim / 2.0, x_lim, alpha)
        reward -= soft_bounding_penalty(
            abs(theta), theta_lim / 2.0, theta_lim, alpha
        )
        reward -= (omega - target_omega) ** alpha

        return obs, reward, term, trunc, info


def cartpole_dense(**kwargs) -> gym.Env:
    env = gym.make("CartPole-v1", **kwargs)
    env = CartPoleDenseWrapper(env)
    return env


register(
    id="DenseCartPole-v1",
    entry_point="rlgym.environments:cartpole_dense",
)
