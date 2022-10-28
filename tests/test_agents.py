import gym
from rlgym.agents import RandomAgentInEnvironment, Experience


def test_experience() -> None:
    e = Experience("obs", 3, 1.0, "new_obs", False)
    assert e.obs == "obs"
    assert e.action == 3
    assert e.reward == 1.0
    assert e.next_obs == "new_obs"
    assert e.terminated is False


class TestRandomAgentInEnvironment:
    def test_functionality(self, mc_env: gym.Env) -> None:
        agent = RandomAgentInEnvironment(mc_env)
        sampled_actions = [agent.select_action() for _ in range(1000)]
        assert max(sampled_actions) == 2  # MountainCar has 3 actions
        assert min(sampled_actions) == 0

    def test_collect_experience(self, mc_env: gym.Env) -> None:
        agent = RandomAgentInEnvironment(mc_env)
        exp = agent.collect_experience()
        assert isinstance(exp, Experience)
        assert agent._episode_step == 1
        assert agent._episode_reward == exp.reward
