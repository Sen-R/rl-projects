import typing
import os
import click
import gym

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tf noise on stderr
import tensorflow as tf  # noqa: E402
from rlgym.networks import mlp_q_network  # noqa: E402
from rlgym.dqn import DQNAgent  # noqa: E402
from rlgym.learning_utils import EpsilonSchedule  # noqa: E402


@click.command()
@click.argument("env_name")
@click.option(
    "--hidden-layers",
    type=str,
    default="",
    help="Comma separated list of hidden layer sizes.",
)
@click.option("--epochs", type=int, required=True, help="Number of epochs.")
@click.option(
    "--steps-per-epoch",
    type=int,
    required=True,
    help="Training steps per epoch.",
)
@click.option(
    "--batch-size", type=int, required=True, help="DQN training batch size."
)
@click.option(
    "--memory-size", type=int, required=True, help="Replay buffer capacity."
)
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
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    help="checkpoint directory",
)
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
    checkpoint_dir: typing.Optional[os.PathLike] = None,
):
    """Train a linear policy using vanilla Q-learning for the given
    environment."""

    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    hidden_layers_list = _parse_hidden_layers(hidden_layers)

    def Q_builder() -> tf.keras.Model:
        return mlp_q_network(env, hidden_layers_list)

    Q_builder().summary()
    print()

    tf.keras.backend.clear_session()  # clear up memory before training
    agent = DQNAgent(env, Q_builder, checkpoint_dir)
    agent.learn(
        EpsilonSchedule(*epsilon),
        gamma,
        epochs,
        steps_per_epoch,
        batch_size,
        memory_size,
        target_update_alpha,
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
