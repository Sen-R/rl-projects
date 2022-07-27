import typing
import gym
from .agents import AgentInEnvironment, RandomAgentInEnvironment
from .dqn import mlp_q_agent_builder


_builders: typing.Dict[str, typing.Callable[..., AgentInEnvironment]] = {}


def create(agent_name: str, env: gym.Env, **kwargs) -> AgentInEnvironment:
    return _builders[agent_name](env, **kwargs)


def register_agent(
    agent_name: str,
    builder: typing.Callable[..., AgentInEnvironment],
) -> None:
    if agent_name in _builders:
        raise ValueError(f"Agent already registerd with name: {agent_name}")
    _builders[agent_name] = builder


register_agent("random", RandomAgentInEnvironment)
register_agent("dqn_mlp", mlp_q_agent_builder)
