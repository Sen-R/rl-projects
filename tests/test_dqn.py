import pytest
from numpy.testing import assert_array_equal
import numpy as np
import gym
from rlgym.dqn import select_action_epsilon_greedily, DQNAgent
from rlgym.networks import mlp_q_network


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
def agent(cp_env: gym.Env) -> DQNAgent:
    return DQNAgent(cp_env, lambda: mlp_q_network(cp_env, [3, 4]), epsilon=0.0)


class TestDQNAgent:
    def test_initial_state(self, agent: DQNAgent) -> None:
        for w_t, w_o in zip(
            agent.Q_target.get_weights(), agent.Q.get_weights()
        ):
            assert_array_equal(w_t, w_o)

    def test_select_action(self, agent: DQNAgent) -> None:
        action_values = agent.Q(agent._obs[np.newaxis, :]).numpy().squeeze()
        best_action = np.argmax(action_values)
        chosen_action = agent.select_action()
        assert chosen_action == best_action

    @pytest.mark.xfail
    def test_td_target(self, agent: DQNAgent) -> None:
        raise NotImplementedError

    @pytest.mark.xfail
    def test_train_step(self, agent: DQNAgent) -> None:
        raise NotImplementedError

    @pytest.mark.xfail
    def test_learn(self, agent: DQNAgent) -> None:
        raise NotImplementedError
