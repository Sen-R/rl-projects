import pytest
import gym
from rlgym.agents import RandomAgentInEnvironment
from rlgym.dqn import QAgentInEnvironment
from rlgym import agent_factory


@pytest.fixture
def env() -> gym.Env:
    return gym.make("CartPole-v1")


class TestAgentFactory:
    def test_make_random_agent(self, env: gym.Env) -> None:
        agent = agent_factory.create("random", env)
        assert isinstance(agent, RandomAgentInEnvironment)

    def test_make_q_agent(self, env: gym.Env) -> None:
        agent = agent_factory.create(
            "dqn_mlp", env, hidden_layers=[32, 32], epsilon=0.1
        )
        assert isinstance(agent, QAgentInEnvironment)
        assert len(agent.Q.layers) == 3
        assert agent.epsilon == 0.1
