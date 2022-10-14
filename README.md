# RL gym

## Introduction

Starter projects to train basic deep RL agents on benchmark environments.

## Getting started

If you have poetry, you can get started by running `poetry install`
and then `poetry shell`. Note that you will first need
[SWIG](https://swig.org) installed on your system to be able to
install the Box2D Gym environments.

Alternatively (e.g. on Colab), install any missing dependencies in
`pyproject.toml` in a suitably prepared virtual environment.

## Examples

### Training an agent

Use deep double Q learning with a two-layer 32 unit MLP as the Q
network to solve LunarLander:

```bash
python -m scripts.train \
  LunarLander-v2 \
  --hidden-layers 32,32 \
  --epochs 60 \
  --steps-per-epoch 500 \
  --batch-size 32 \
  --memory-size 10000 \
  --epsilon 1.0 2000 0.1 \
  --gamma 0.99 \
  --target-update-alpha 0.01 \
  --checkpoint-dir <your-checkpoint-dir-here> \
  --render  # to view the simulation during training
```

Use 

```bash
python -m scripts.train --help`
```

to check out the meaning of these parameters.

# Simulating an agent interacting with the environment

Try

```bash
python -m scripts.enjoy \
  LunarLander-v2 \
  dqn_mlp \
  --agent-config '{"hidden_layers": [128, 128, 128], "checkpoint_dir": "checkpoints/LunarLander-v2"}' \
  --steps 1000
```

Again, use the `--help` option to understand the command line arguments.

# Still to do

* Additional algorithms
* Handling multidimensional inputs (e.g. screens) and other network
  architectures
* Reward clipping

