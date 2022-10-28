import pytest
import gym


@pytest.fixture
def cp_env() -> gym.Env:
    return gym.make("CartPole-v1")


@pytest.fixture
def mc_env() -> gym.Env:
    return gym.make("MountainCar-v0")
