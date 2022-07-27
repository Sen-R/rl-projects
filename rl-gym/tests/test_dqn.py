import typing
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
import tensorflow as tf
import gym
from rlgym.dqn import (
    mlp_q_network,
    select_action_epsilon_greedily,
    EpsilonSchedule,
    Experience,
    ReplayBuffer,
    soft_update,
    QAgentInEnvironment,
)


@pytest.fixture
def env() -> gym.Env:
    return gym.make("CartPole-v1", new_step_api=True)


@pytest.fixture
def sample_obs(env: gym.Env) -> np.ndarray:
    sample_obs = env.reset()
    assert isinstance(sample_obs, np.ndarray)
    return sample_obs[np.newaxis, :]


def Q_builder(env: gym.Env) -> typing.Callable[[], tf.keras.Model]:
    return mlp_q_network(env, [3, 4])


@pytest.fixture
def Q(env: gym.Env) -> tf.keras.Model:
    return Q_builder(env)


class TestMLPQNetwork:
    def test_maps_env_observations_to_valid_action_values(
        self, env: gym.Env, Q: tf.keras.Model, sample_obs: np.ndarray
    ) -> None:
        sample_output = Q(sample_obs)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert_array_equal(sample_output.shape, [1, env.action_space.n])

    def test_hidden_layers_as_expected(self, Q: tf.keras.Model) -> None:
        assert len(Q.layers) == 3
        assert Q.layers[0].units == 3
        assert Q.layers[1].units == 4

    def test_default_is_linear_model(self, env: gym.Env) -> None:
        linear_Q = mlp_q_network(env)
        assert len(linear_Q.layers) == 1


class TestSelectActionEpsilonGreedily:
    def test_output_scalar_when_input_rank_one(self) -> None:
        action_values = np.array([1.0, 2.0, 0.0])
        action = select_action_epsilon_greedily(action_values, epsilon=0.0)
        assert_array_equal(action, 1)

    def test_output_vector_when_input_rank_two(self) -> None:
        action_values = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
        actions = select_action_epsilon_greedily(action_values, epsilon=0.0)
        assert_array_equal(actions, [1, 2])

    @pytest.mark.parametrize(
        "epsilon,lower,upper",
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.65, 0.71),  # should be about 2/3
            (0.5, 0.30, 0.35),  # should be about 1/3
        ],
    )
    def test_epsilon_statistics(
        self,
        epsilon: float,
        lower: float,
        upper: float,
    ) -> None:
        # Set up a random action_values matrix
        np.random.seed(82432)  # for reproducibility
        n_actions = 3
        n_trials = 1000  # larger makes type I errors less likely but is slower
        action_values = np.random.normal(size=(n_trials, n_actions))

        # Compare epsilon greedy selected actions against best action values
        # and report how often the best action was chosen
        best_action_values = np.max(action_values, axis=-1)
        chosen_actions = select_action_epsilon_greedily(action_values, epsilon)
        chosen_action_values = np.take_along_axis(
            action_values, chosen_actions[:, np.newaxis], axis=1
        ).squeeze()

        nongreedy_action_proportion = 1.0 - np.mean(
            np.isclose(best_action_values, chosen_action_values)
        )

        assert lower <= nongreedy_action_proportion <= upper


@pytest.fixture
def epsilon() -> EpsilonSchedule:
    return EpsilonSchedule(1.0, 9000, 0.1)


class TestEpsilonSchedule:
    def test_starting_value(self, epsilon: EpsilonSchedule) -> None:
        assert_almost_equal(epsilon(0), 1.0)

    def test_long_term_value(self, epsilon: EpsilonSchedule) -> None:
        assert_almost_equal(epsilon(1000000), 0.1)

    @pytest.mark.parametrize("step,expected", [(1000, 0.9), (5000, 0.5)])
    def test_linear_rampdown_in_between(
        self, epsilon: EpsilonSchedule, step: int, expected: float
    ) -> None:
        assert_almost_equal(epsilon(step), expected)


def test_experience() -> None:
    e = Experience("obs", 3, 1.0, "new_obs", False)
    assert e.obs == "obs"
    assert e.action == 3
    assert e.reward == 1.0
    assert e.next_obs == "new_obs"
    assert e.terminated is False


class TestReplayBuffer:
    def test_initial_state(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        assert len(buf) == 0

    def test_insert_one_and_retrieve_the_same(self) -> None:
        buf = ReplayBuffer(maxlen=1)
        buf.add(Experience(1, 2, 3, 4, True))
        assert len(buf) == 1
        assert buf.cursor == 0  # maxlen is only one
        assert buf[0] == Experience(1, 2, 3, 4, True)

    def test_insert_second_and_retrieve_both(self) -> None:
        buf = ReplayBuffer(maxlen=3)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.add(Experience(5, 6, 7, 8, False))
        assert len(buf) == 2
        assert buf.cursor == 2
        assert buf[0] == Experience(1, 2, 3, 4, True)
        assert buf[1] == Experience(5, 6, 7, 8, False)

    def test_insert_over_maxlen_goes_back_to_start(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.add(Experience(5, 6, 7, 8, False))
        buf.add(Experience(9, 0, 1, 2, True))
        assert len(buf) == 2
        assert buf.cursor == 1
        assert buf[0] == Experience(9, 0, 1, 2, True)
        assert buf[1] == Experience(5, 6, 7, 8, False)

    def test_getitem_using_multiple_indices(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        buf.add(Experience([0, 1], 0, 1.0, [2, 3], False))
        buf.add(Experience([2, 3], 1, 0.0, [4, 5], True))
        o, a, r, no, t = buf[np.array([1, 0])]
        assert_array_equal(o, [[2, 3], [0, 1]])
        assert_array_equal(a, [1, 0])
        assert_array_equal(r, [0.0, 1.0])
        assert_array_equal(no, [[4, 5], [2, 3]])
        assert_array_equal(t, [True, False])

    def test_sample_returns_correct_number_of_samples(self) -> None:
        buf = ReplayBuffer(maxlen=2)
        buf.add(Experience(1, 2, 3, 4, True))
        buf.add(Experience(5, 6, 7, 8, False))
        o, a, r, no, t = buf.sample(1)
        assert len(o) == len(a) == len(r) == len(no) == len(t) == 1

    def test_getitem_raises_when_index_out_of_bounds(self) -> None:
        buf = ReplayBuffer(maxlen=1)
        buf.add(Experience(1, 2, 3, 4, True))
        with pytest.raises(IndexError):
            buf[1]

    def test_sample_raises_when_asking_for_too_many(self) -> None:
        buf = ReplayBuffer(maxlen=1)
        buf.add(Experience(1, 2, 3, 4, True))
        with pytest.raises(ValueError):
            buf.sample(2)


@pytest.mark.parametrize(
    "alpha,expected", [(0.0, 0.5), (1.0, 1.5), (0.2, 0.7)]
)
def test_soft_update(alpha: float, expected: float) -> None:
    # Prepare trivial (single-parameter) target and online networks
    # with prespecified weights
    target = tf.keras.Sequential(
        tf.keras.layers.Dense(1, use_bias=False, input_shape=(1,))
    )
    target.set_weights([np.array([[0.5]])])

    online = tf.keras.Sequential(
        tf.keras.layers.Dense(1, use_bias=False, input_shape=(1,))
    )
    online.set_weights([np.array([[1.5]])])

    # Perform soft-update with alpha and check online is unchanged and
    # target changes correctly
    soft_update(target, online, alpha)
    assert_almost_equal(online.get_weights()[0], 1.5)
    assert_almost_equal(target.get_weights()[0], expected)


@pytest.fixture
def agent(env: gym.Env) -> QAgentInEnvironment:
    return QAgentInEnvironment(env, lambda: Q_builder(env), 10)


class TestQAgentInEnvironment:
    def test_initial_state(self, agent: QAgentInEnvironment) -> None:
        assert len(agent.memory) == 0
        for w_t, w_o in zip(
            agent.Q_target.get_weights(), agent.Q.get_weights()
        ):
            assert_array_equal(w_t, w_o)

    def test_select_action(self, agent: QAgentInEnvironment) -> None:
        action_values = agent.Q(agent._obs[np.newaxis, :]).numpy().squeeze()
        best_action = np.argmax(action_values)
        chosen_action = agent.select_action(epsilon=0.0)
        assert chosen_action == best_action

    def test_collect_experience(self, agent: QAgentInEnvironment) -> None:
        agent.collect_experience(epsilon=0.1)
        assert len(agent.memory) == 1
        exp = agent.memory[0]
        assert agent._episode_step == 1
        assert agent._episode_reward == exp.reward

    @pytest.mark.xfail
    def test_td_target(self, agent: QAgentInEnvironment) -> None:
        raise NotImplementedError

    @pytest.mark.xfail
    def test_train_step(self, agent: QAgentInEnvironment) -> None:
        raise NotImplementedError

    @pytest.mark.xfail
    def test_learn(self, agent: QAgentInEnvironment) -> None:
        raise NotImplementedError
