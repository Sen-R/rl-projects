import click
import gym


@click.command()
@click.argument("env_name")
@click.option("--seed", type=int, default=42, help="Environment seed.")
@click.option("--steps", type=int, default=1000, help="Number of steps.")
def enjoy(env_name, seed, steps):
    """Render a random agent playing the given environment."""

    env = gym.make(env_name, render_mode="human", new_step_api=True)
    env.reset(seed=seed)

    print(env.action_space)
    print(env.observation_space)

    for _ in range(steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    enjoy()
