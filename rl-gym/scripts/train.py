import typing
import os
import click
import gym

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tf noise on stderr
import tensorflow as tf  # noqa: E402
from rlgym.dqn import (  # noqa: E402
    mlp_q_network,
    QAgentInEnvironment,
    EpsilonSchedule,
)


@click.command()
@click.argument("env_name")
@click.option(
    "--hidden-layers",
    type=str,
    default="",
    help="Comma separated list of hidden layer sizes.",
)
@click.argument("epochs", type=int)
@click.argument("steps_per_epoch", type=int)
@click.argument("batch_size", type=int)
@click.argument("memory_size", type=int)
@click.option(
    "--epsilon",
    type=(float, int, float),
    required=True,
    help="Epsilon schedule in format `start_val, rampdown_length, end_val`.",
)
@click.option("--gamma", type=float, default=1.0, help="Discount rate.")
@click.option(
    "--target-update-alpha",
    type=float,
    default=0.1,
    help="Soft update parameter for updating target Q network.",
)
@click.option("--render/--no-render", help="View environment during training.")
def train(
    env_name: str,
    hidden_layers: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    memory_size: int,
    epsilon: typing.Tuple[float, int, float],
    gamma: float,
    target_update_alpha: float,
    render: bool,
):
    """Train a linear policy using vanilla Q-learning for the given
    environment."""

    render_mode = "human" if render else None
    env = gym.make(env_name, new_step_api=True, render_mode=render_mode)
    hidden_layers_list = _parse_hidden_layers(hidden_layers)

    def Q_builder() -> tf.keras.Model:
        return mlp_q_network(env, hidden_layers_list)

    Q_builder().summary()
    print()

    tf.keras.backend.clear_session()  # clear up memory before training
    agent = QAgentInEnvironment(
        env, Q_builder, memory_size, target_update_alpha
    )
    agent.learn(
        EpsilonSchedule(*epsilon), gamma, epochs, steps_per_epoch, batch_size
    )


def _parse_hidden_layers(hidden_layers: str) -> typing.List[int]:
    try:
        return [int(n.strip()) for n in hidden_layers.split(",") if n]
    except ValueError:
        raise ValueError(
            f"Error parsing --hidden-layers, got: {hidden_layers}"
        )


if __name__ == "__main__":
    train()
