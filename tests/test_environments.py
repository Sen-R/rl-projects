import gym


class TestDenseCartPole:
    def test_functionality(self) -> None:
        env = gym.make(
            "rlgym.environments:DenseCartPole-v1",
            render_mode="human",
            new_step_api=True,
        )
        env.reset()
        _, r, _, _, _ = env.step(env.action_space.sample())  # type: ignore
        assert r < 1.0
