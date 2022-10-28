import gym
from rlgym.agents import RandomAgentInEnvironment
from rlgym.dqn import DQNAgent
from rlgym import agent_factory


class TestAgentFactory:
    def test_make_random_agent(self, cp_env: gym.Env) -> None:
        agent = agent_factory.create("random", cp_env)
        assert isinstance(agent, RandomAgentInEnvironment)

    def test_make_q_agent(self, cp_env: gym.Env) -> None:
        agent = agent_factory.create(
            "dqn_mlp", cp_env, hidden_layers=[32, 32], epsilon=0.1
        )
        assert isinstance(agent, DQNAgent)
        assert len(agent.Q.layers) == 3
        assert agent.epsilon == 0.1
