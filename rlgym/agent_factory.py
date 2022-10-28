import typing
import os
import gym
from .agents import AgentInEnvironment, RandomAgentInEnvironment
from .dqn import QAgentInEnvironment
from .networks import mlp_q_network


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


def mlp_q_agent_builder(
    env: gym.Env,
    hidden_layers: typing.Sequence[int],
    checkpoint_dir: typing.Optional[typing.Union[str, os.PathLike]] = None,
    epsilon: float = 0.0,
) -> QAgentInEnvironment:
    agent = QAgentInEnvironment(
        env,
        lambda: mlp_q_network(env, hidden_layers),
        checkpoint_dir,
        epsilon,
    )
    return agent


register_agent("random", RandomAgentInEnvironment)
register_agent("dqn_mlp", mlp_q_agent_builder)
