"""
Unit tests for RL agents: Q-Learning, Thompson Sampling, and Combined Agent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import json
import numpy as np
from collections import defaultdict

from environment.state import State, ACTIONS
from rl.q_learning import QLearningAgent
from rl.thompson_sampling import ThompsonSamplingAgent
from rl.combined_agent import CombinedAgent


class TestQLearningAgent(unittest.TestCase):
    """Test QLearningAgent class."""

    def setUp(self):
        """Create agent for each test."""
        self.agent = QLearningAgent(
            actions=ACTIONS,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.alpha, 0.1)
        self.assertEqual(self.agent.gamma, 0.95)
        self.assertEqual(self.agent.epsilon, 0.1)
        self.assertEqual(self.agent.actions, ACTIONS)
        self.assertEqual(self.agent.total_updates, 0)

    def test_initialization_default_actions(self):
        """Test agent uses default actions."""
        agent = QLearningAgent()
        self.assertEqual(agent.actions, ACTIONS)

    def test_get_q_values_initial(self):
        """Test Q-values are zero initially."""
        state = State.initial()
        q_values = self.agent.get_q_values(state)

        self.assertIsInstance(q_values, dict)
        self.assertEqual(len(q_values), len(ACTIONS))
        for action in ACTIONS:
            self.assertEqual(q_values[action], 0.0)

    def test_get_q_value_single(self):
        """Test getting single Q-value."""
        state = State.initial()
        q_value = self.agent.get_q_value(state, "planner")
        self.assertEqual(q_value, 0.0)

    def test_choose_action_returns_valid_action(self):
        """Test that choose_action returns a valid action."""
        state = State.initial()
        action = self.agent.choose_action(state)
        self.assertIn(action, ACTIONS)

    def test_choose_action_with_valid_actions_filter(self):
        """Test choosing from subset of valid actions."""
        state = State.initial()
        valid = ["planner", "coder"]
        action = self.agent.choose_action(state, valid_actions=valid)
        self.assertIn(action, valid)

    def test_choose_action_epsilon_greedy_explore(self):
        """Test epsilon-greedy exploration."""
        # Set high epsilon for exploration
        agent = QLearningAgent(epsilon=1.0)
        state = State.initial()

        # With epsilon=1.0, should always explore (random)
        # Run multiple times to verify randomness
        actions = [agent.choose_action(state) for _ in range(100)]
        unique_actions = set(actions)
        # Should explore multiple actions
        self.assertGreater(len(unique_actions), 1)

    def test_choose_action_greedy(self):
        """Test greedy action selection."""
        state = State.initial()

        # Set Q-value for planner higher
        self.agent.q_table[state.to_key()]["planner"] = 10.0

        # With explore=False, should always pick planner
        for _ in range(10):
            action = self.agent.choose_action(state, explore=False)
            self.assertEqual(action, "planner")

    def test_update_increases_q_value(self):
        """Test that positive reward increases Q-value."""
        state = State.initial()
        next_state = State(has_plan=True)

        old_q = self.agent.get_q_value(state, "planner")
        self.agent.update(state, "planner", reward=1.0, next_state=next_state, done=False)
        new_q = self.agent.get_q_value(state, "planner")

        self.assertGreater(new_q, old_q)

    def test_update_decreases_q_value(self):
        """Test that negative reward decreases Q-value."""
        state = State.initial()
        next_state = State.initial()

        # Start with positive Q-value
        self.agent.q_table[state.to_key()]["planner"] = 5.0

        old_q = self.agent.get_q_value(state, "planner")
        self.agent.update(state, "planner", reward=-2.0, next_state=next_state, done=False)
        new_q = self.agent.get_q_value(state, "planner")

        self.assertLess(new_q, old_q)

    def test_update_terminal_state(self):
        """Test update for terminal state."""
        state = State(has_code=True)
        next_state = State(has_code=True, tests_pass=True)

        self.agent.update(state, "tester", reward=10.0, next_state=next_state, done=True)

        # For terminal state, Q should be approximately alpha * reward
        q_value = self.agent.get_q_value(state, "tester")
        expected = 0.1 * 10.0  # alpha * reward
        self.assertAlmostEqual(q_value, expected, places=5)

    def test_update_increments_counter(self):
        """Test that update increments total_updates."""
        state = State.initial()
        next_state = State.initial()

        self.assertEqual(self.agent.total_updates, 0)
        self.agent.update(state, "planner", 0.0, next_state, False)
        self.assertEqual(self.agent.total_updates, 1)

    def test_q_learning_formula(self):
        """Test Q-learning update formula."""
        # Q(s,a) = Q(s,a) + alpha * (r + gamma * max_Q(s') - Q(s,a))
        agent = QLearningAgent(alpha=0.5, gamma=0.9)

        state = State.initial()
        next_state = State(has_plan=True)

        # Set initial Q-values
        agent.q_table[state.to_key()]["planner"] = 1.0
        agent.q_table[next_state.to_key()]["coder"] = 2.0

        # Update
        agent.update(state, "planner", reward=0.5, next_state=next_state, done=False)

        # Expected: 1.0 + 0.5 * (0.5 + 0.9 * 2.0 - 1.0) = 1.0 + 0.5 * 1.3 = 1.65
        expected = 1.0 + 0.5 * (0.5 + 0.9 * 2.0 - 1.0)
        actual = agent.get_q_value(state, "planner")
        self.assertAlmostEqual(actual, expected, places=5)

    def test_get_policy(self):
        """Test getting learned policy."""
        state = State.initial()

        # Set Q-values to make planner best
        self.agent.q_table[state.to_key()]["planner"] = 5.0
        self.agent.q_table[state.to_key()]["coder"] = 2.0

        policy = self.agent.get_policy()
        self.assertEqual(policy[state.to_key()], "planner")

    def test_get_state_value(self):
        """Test getting state value (max Q)."""
        state = State.initial()

        self.agent.q_table[state.to_key()]["planner"] = 5.0
        self.agent.q_table[state.to_key()]["coder"] = 3.0

        value = self.agent.get_state_value(state)
        self.assertEqual(value, 5.0)

    def test_decay_epsilon(self):
        """Test epsilon decay."""
        agent = QLearningAgent(epsilon=0.5)

        agent.decay_epsilon(decay_rate=0.9)
        self.assertAlmostEqual(agent.epsilon, 0.45)

        # Test minimum epsilon
        agent.epsilon = 0.02
        agent.decay_epsilon(decay_rate=0.9, min_epsilon=0.01)
        self.assertAlmostEqual(agent.epsilon, 0.018)

        agent.decay_epsilon(decay_rate=0.1, min_epsilon=0.01)
        self.assertEqual(agent.epsilon, 0.01)

    def test_save_and_load(self):
        """Test saving and loading agent."""
        state = State.initial()
        self.agent.q_table[state.to_key()]["planner"] = 5.0
        self.agent.total_updates = 100

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # Save
            self.agent.save(filepath)

            # Create new agent and load
            new_agent = QLearningAgent()
            new_agent.load(filepath)

            # Verify
            self.assertEqual(new_agent.alpha, self.agent.alpha)
            self.assertEqual(new_agent.gamma, self.agent.gamma)
            self.assertEqual(new_agent.total_updates, 100)
            self.assertEqual(
                new_agent.get_q_value(state, "planner"),
                5.0
            )
        finally:
            os.unlink(filepath)


class TestThompsonSamplingAgent(unittest.TestCase):
    """Test ThompsonSamplingAgent class."""

    def setUp(self):
        """Create agent for each test."""
        self.agent = ThompsonSamplingAgent(
            actions=ACTIONS,
            prior_alpha=1.0,
            prior_beta=1.0
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.prior_alpha, 1.0)
        self.assertEqual(self.agent.prior_beta, 1.0)
        self.assertEqual(self.agent.actions, ACTIONS)
        self.assertEqual(self.agent.total_updates, 0)

    def test_sample_returns_valid_range(self):
        """Test that samples are in [0, 1] range."""
        state = State.initial()

        for _ in range(100):
            samples = self.agent.sample(state)
            for action, sample in samples.items():
                self.assertGreaterEqual(sample, 0.0)
                self.assertLessEqual(sample, 1.0)

    def test_sample_all_actions(self):
        """Test that sample returns value for all actions."""
        state = State.initial()
        samples = self.agent.sample(state)

        self.assertEqual(len(samples), len(ACTIONS))
        for action in ACTIONS:
            self.assertIn(action, samples)

    def test_get_mean_initial(self):
        """Test initial mean values."""
        state = State.initial()
        means = self.agent.get_mean(state)

        # With alpha=1, beta=1, mean = 1/(1+1) = 0.5
        for action in ACTIONS:
            self.assertAlmostEqual(means[action], 0.5)

    def test_get_uncertainty_initial(self):
        """Test initial uncertainty."""
        state = State.initial()

        for action in ACTIONS:
            uncertainty = self.agent.get_uncertainty(state, action)
            # With alpha=1, beta=1, variance = 1*1 / (4 * 3) = 1/12 ≈ 0.0833
            self.assertGreater(uncertainty, 0)
            self.assertLess(uncertainty, 1)

    def test_choose_action_returns_valid_action(self):
        """Test that choose_action returns a valid action."""
        state = State.initial()
        action = self.agent.choose_action(state)
        self.assertIn(action, ACTIONS)

    def test_choose_action_with_valid_actions_filter(self):
        """Test choosing from subset of valid actions."""
        state = State.initial()
        valid = ["planner", "coder"]
        action = self.agent.choose_action(state, valid_actions=valid)
        self.assertIn(action, valid)

    def test_choose_action_is_stochastic(self):
        """Test that action selection is stochastic."""
        state = State.initial()

        actions = [self.agent.choose_action(state) for _ in range(100)]
        unique_actions = set(actions)
        # Should choose multiple different actions due to sampling
        self.assertGreater(len(unique_actions), 1)

    def test_update_positive_reward_increases_alpha(self):
        """Test that positive reward increases alpha (success count)."""
        state = State.initial()
        action = "planner"

        old_alpha = self.agent.alpha[state.to_key()][action]
        self.agent.update(state, action, reward=5.0)  # Positive reward
        new_alpha = self.agent.alpha[state.to_key()][action]

        self.assertGreater(new_alpha, old_alpha)

    def test_update_negative_reward_increases_beta(self):
        """Test that negative reward increases beta (failure count)."""
        state = State.initial()
        action = "planner"

        old_beta = self.agent.beta[state.to_key()][action]
        self.agent.update(state, action, reward=-2.0)  # Negative reward
        new_beta = self.agent.beta[state.to_key()][action]

        self.assertGreater(new_beta, old_beta)

    def test_update_changes_mean(self):
        """Test that updates change the mean."""
        state = State.initial()

        # Multiple positive updates
        for _ in range(10):
            self.agent.update(state, "planner", reward=5.0)

        means = self.agent.get_mean(state)
        # Planner should have higher mean now
        self.assertGreater(means["planner"], means["coder"])

    def test_update_reduces_uncertainty(self):
        """Test that more updates reduce uncertainty."""
        state = State.initial()

        initial_uncertainty = self.agent.get_uncertainty(state, "planner")

        # Multiple updates
        for _ in range(20):
            self.agent.update(state, "planner", reward=3.0)

        final_uncertainty = self.agent.get_uncertainty(state, "planner")
        self.assertLess(final_uncertainty, initial_uncertainty)

    def test_get_exploration_bonus(self):
        """Test exploration bonus calculation."""
        state = State.initial()

        bonus = self.agent.get_exploration_bonus(state, "planner")
        self.assertGreater(bonus, 0)

    def test_save_and_load(self):
        """Test saving and loading agent."""
        state = State.initial()

        # Do some updates
        for _ in range(10):
            self.agent.update(state, "planner", 5.0)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # Save
            self.agent.save(filepath)

            # Create new agent and load
            new_agent = ThompsonSamplingAgent()
            new_agent.load(filepath)

            # Verify
            self.assertEqual(new_agent.total_updates, self.agent.total_updates)

            # Means should be similar
            old_mean = self.agent.get_mean(state)["planner"]
            new_mean = new_agent.get_mean(state)["planner"]
            self.assertAlmostEqual(old_mean, new_mean, places=5)
        finally:
            os.unlink(filepath)


class TestCombinedAgent(unittest.TestCase):
    """Test CombinedAgent class."""

    def setUp(self):
        """Create agent for each test."""
        self.agent = CombinedAgent(
            actions=ACTIONS,
            alpha=0.1,
            gamma=0.95,
            exploration_scale=2.0
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.alpha, 0.1)
        self.assertEqual(self.agent.gamma, 0.95)
        self.assertEqual(self.agent.exploration_scale, 2.0)
        self.assertEqual(self.agent.actions, ACTIONS)
        self.assertIsInstance(self.agent.q_agent, QLearningAgent)
        self.assertIsInstance(self.agent.ts_agent, ThompsonSamplingAgent)

    def test_get_q_values(self):
        """Test getting Q-values."""
        state = State.initial()
        q_values = self.agent.get_q_values(state)

        self.assertIsInstance(q_values, dict)
        self.assertEqual(len(q_values), len(ACTIONS))

    def test_get_sampled_values(self):
        """Test getting sampled values."""
        state = State.initial()
        sampled = self.agent.get_sampled_values(state)

        self.assertIsInstance(sampled, dict)
        self.assertEqual(len(sampled), len(ACTIONS))

    def test_sampled_values_vary(self):
        """Test that sampled values vary between calls."""
        state = State.initial()

        samples = [self.agent.get_sampled_values(state) for _ in range(10)]

        # Should have some variation
        planner_values = [s["planner"] for s in samples]
        self.assertGreater(max(planner_values) - min(planner_values), 0)

    def test_choose_action_returns_valid(self):
        """Test that choose_action returns valid action."""
        state = State.initial()
        action = self.agent.choose_action(state)
        self.assertIn(action, ACTIONS)

    def test_choose_action_with_valid_actions(self):
        """Test choosing from subset of actions."""
        state = State.initial()
        valid = ["planner", "coder"]
        action = self.agent.choose_action(state, valid_actions=valid)
        self.assertIn(action, valid)

    def test_choose_action_explore_vs_exploit(self):
        """Test difference between explore and exploit modes."""
        state = State.initial()

        # Set up clear preference
        self.agent.q_agent.q_table[state.to_key()]["planner"] = 10.0

        # Exploit should consistently pick planner
        exploit_actions = [
            self.agent.choose_action(state, explore=False)
            for _ in range(20)
        ]
        self.assertTrue(all(a == "planner" for a in exploit_actions))

        # Explore might occasionally pick other actions
        # (though not guaranteed with high Q-value)

    def test_update_both_components(self):
        """Test that update affects both Q-learning and Thompson Sampling."""
        state = State.initial()
        next_state = State(has_plan=True)

        # Update
        self.agent.update(state, "planner", reward=5.0, next_state=next_state, done=False)

        # Q-value should change
        q_value = self.agent.get_q_values(state)["planner"]
        self.assertNotEqual(q_value, 0.0)

        # Thompson Sampling should update
        mean = self.agent.ts_agent.get_mean(state)["planner"]
        self.assertGreater(mean, 0.5)  # Should be > 0.5 after positive reward

    def test_update_increments_counter(self):
        """Test that update increments total_updates."""
        state = State.initial()
        next_state = State.initial()

        self.assertEqual(self.agent.total_updates, 0)
        self.agent.update(state, "planner", 0.0, next_state, False)
        self.assertEqual(self.agent.total_updates, 1)

    def test_get_policy(self):
        """Test getting policy from combined agent."""
        state = State.initial()

        # Set up Q-values
        self.agent.q_agent.q_table[state.to_key()]["planner"] = 5.0

        policy = self.agent.get_policy()
        self.assertEqual(policy[state.to_key()], "planner")

    def test_get_uncertainty(self):
        """Test getting uncertainty for all actions."""
        state = State.initial()

        uncertainty = self.agent.get_uncertainty(state)

        self.assertIsInstance(uncertainty, dict)
        self.assertEqual(len(uncertainty), len(ACTIONS))
        for action in ACTIONS:
            self.assertGreater(uncertainty[action], 0)

    def test_save_and_load(self):
        """Test saving and loading combined agent."""
        state = State.initial()

        # Do some training
        next_state = State(has_plan=True)
        for _ in range(10):
            self.agent.update(state, "planner", 5.0, next_state, False)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # Save
            self.agent.save(filepath)

            # Create new agent and load
            new_agent = CombinedAgent()
            new_agent.load(filepath)

            # Verify
            self.assertEqual(new_agent.alpha, self.agent.alpha)
            self.assertEqual(new_agent.gamma, self.agent.gamma)
            self.assertEqual(new_agent.total_updates, self.agent.total_updates)

            # Q-values should match
            old_q = self.agent.get_q_values(state)["planner"]
            new_q = new_agent.get_q_values(state)["planner"]
            self.assertAlmostEqual(old_q, new_q, places=5)
        finally:
            os.unlink(filepath)


class TestRLAgentLearning(unittest.TestCase):
    """Integration tests for RL agent learning."""

    def test_q_learning_converges(self):
        """Test that Q-learning learns in simple environment."""
        agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

        # Simple training: reward planner in initial state
        state = State.initial()
        next_state = State(has_plan=True)

        for _ in range(100):
            agent.update(state, "planner", reward=1.0, next_state=next_state, done=False)

        # Planner should have highest Q-value
        q_values = agent.get_q_values(state)
        self.assertEqual(max(q_values, key=q_values.get), "planner")

    def test_thompson_sampling_learns_preference(self):
        """Test that Thompson Sampling learns action preferences."""
        agent = ThompsonSamplingAgent()
        state = State.initial()

        # Give positive rewards to planner, negative to coder
        for _ in range(50):
            agent.update(state, "planner", reward=5.0)
            agent.update(state, "coder", reward=-2.0)

        means = agent.get_mean(state)
        self.assertGreater(means["planner"], means["coder"])

    def test_combined_agent_balances_explore_exploit(self):
        """Test that combined agent explores but prefers good actions."""
        agent = CombinedAgent(exploration_scale=1.0)
        state = State.initial()
        next_state = State(has_plan=True)

        # Train heavily on planner
        for _ in range(50):
            agent.update(state, "planner", reward=5.0, next_state=next_state, done=False)
            agent.update(state, "coder", reward=-1.0, next_state=State.initial(), done=False)

        # Should mostly choose planner when exploiting
        exploit_choices = [
            agent.choose_action(state, explore=False)
            for _ in range(50)
        ]
        planner_ratio = exploit_choices.count("planner") / 50
        self.assertGreater(planner_ratio, 0.8)


class TestRLAgentEdgeCases(unittest.TestCase):
    """Test edge cases for RL agents."""

    def test_q_learning_handles_all_states(self):
        """Test Q-learning handles all 64 states."""
        agent = QLearningAgent()

        for i in range(64):
            state = State.from_index(i)
            q_values = agent.get_q_values(state)
            self.assertEqual(len(q_values), 4)

    def test_thompson_sampling_handles_extreme_rewards(self):
        """Test Thompson Sampling handles extreme rewards."""
        agent = ThompsonSamplingAgent()
        state = State.initial()

        # Very large reward
        agent.update(state, "planner", reward=100.0)
        mean = agent.get_mean(state)["planner"]
        self.assertGreater(mean, 0.5)

        # Very negative reward
        agent.update(state, "coder", reward=-100.0)
        mean = agent.get_mean(state)["coder"]
        self.assertLess(mean, 0.5)

    def test_combined_agent_zero_exploration_scale(self):
        """Test combined agent with zero exploration."""
        agent = CombinedAgent(exploration_scale=0.0)
        state = State.initial()

        # Set clear Q-value preference
        agent.q_agent.q_table[state.to_key()]["planner"] = 10.0

        # With zero exploration, sampled values should equal Q-values
        q_values = agent.get_q_values(state)
        sampled = agent.get_sampled_values(state)

        # Planner Q-value should dominate
        self.assertEqual(sampled["planner"], q_values["planner"])


if __name__ == "__main__":
    unittest.main()
