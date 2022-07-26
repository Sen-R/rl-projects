import click
import json
import gym
from rlgym import agent_factory


@click.command()
@click.argument("env_name")
@click.argument("agent-name")
@click.option("--agent-config", help="JSON dict format keyword arguments.")
@click.option("--seed", type=int, default=42, help="Environment seed.")
@click.option("--steps", type=int, default=1000, help="Number of steps.")
def enjoy(
    env_name: str, agent_name: str, agent_config: str, seed: int, steps: int
) -> None:
    """Render an agent playing the given environment."""

    env = gym.make(env_name, render_mode="human", new_step_api=True)
    env.reset(seed=seed)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    if agent_config is None:
        agent_kwargs = {}
    else:
        agent_kwargs = json.loads(agent_config)
        if not isinstance(agent_kwargs, dict):
            raise ValueError(
                f"--agent-config should be passed a dict, got: {agent_kwargs}"
            )
    agent = agent_factory.create(agent_name, env, **agent_kwargs)

    for _ in range(steps):
        agent.collect_experience()

    env.close()


if __name__ == "__main__":
    enjoy()
