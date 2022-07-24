# RL gym

## Introduction

Starter projects to train basic deep RL agents on benchmark environments.

## Getting started

If you have poetry, you can get started by running `poetry install` and then `poetry shell`. Otherwise, install any missing dependencies in `pyproject.toml` in a suitably prepared virtual environment.

## Examples

Use deep double Q learning with a two-layer 32 unit MLP as the Q
network to solve LunarLander:

```bash
python -m scripts.train \
  LunarLander-v2 \
  --hidden-layers 32,32 \
  --epochs 30 \
  --steps-per-epoch 500 \
  --batch-size 32 \
  --memory-size 10000 \
  --epsilon 1.0 2000 0.1 \
  --gamma 0.99 \
  --target-update-alpha 0.01 \
  --render  # to view the simulation during training
```

Use 

```bash
python -m scripts.train --help`
```

to check out the meaning of these parameters.

# Still to do

* Save weights at end of training
* Update `enjoy` script to run trained agents
* Additional algorithms
* Handling multidimensional inputs (e.g. screens) and other network
  architectures
* TD-error and reward clipping

