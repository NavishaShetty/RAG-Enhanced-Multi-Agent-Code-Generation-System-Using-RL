"""
Unit tests for the RL environment: State, Rewards, SimulatedEnv.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from environment.state import State, ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION, get_valid_actions
from environment.rewards import RewardFunction, REWARDS


class TestState(unittest.TestCase):
    """Test State dataclass."""

    def test_initial_state(self):
        """Test creating initial state."""
        state = State.initial()
        self.assertFalse(state.has_plan)
        self.assertFalse(state.has_code)
        self.assertFalse(state.has_error)
        self.assertFalse(state.tests_pass)
        self.assertEqual(state.iteration, 0)

    def test_state_with_values(self):
        """Test creating state with specific values."""
        state = State(
            has_plan=True,
            has_code=True,
            has_error=False,
            tests_pass=False,
            iteration=2
        )
        self.assertTrue(state.has_plan)
        self.assertTrue(state.has_code)
        self.assertFalse(state.has_error)
        self.assertEqual(state.iteration, 2)

    def test_iteration_bucket(self):
        """Test iteration bucketing for tabular RL."""
        state = State(iteration=0)
        self.assertEqual(state.iteration_bucket, 0)

        state.iteration = 1
        self.assertEqual(state.iteration_bucket, 1)

        state.iteration = 2
        self.assertEqual(state.iteration_bucket, 2)

        state.iteration = 3
        self.assertEqual(state.iteration_bucket, 3)

        # Should cap at 3
        state.iteration = 5
        self.assertEqual(state.iteration_bucket, 3)

        state.iteration = 100
        self.assertEqual(state.iteration_bucket, 3)

    def test_to_features(self):
        """Test converting state to feature dictionary."""
        state = State(has_plan=True, has_code=False, iteration=2)
        features = state.to_features()

        self.assertIsInstance(features, dict)
        self.assertTrue(features["has_plan"])
        self.assertFalse(features["has_code"])
        self.assertEqual(features["iteration_bucket"], 2)

    def test_to_key(self):
        """Test converting state to hashable tuple key."""
        state = State(has_plan=True, has_code=True, has_error=False, tests_pass=False, iteration=1)
        key = state.to_key()

        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 5)
        self.assertEqual(key, (1, 1, 0, 0, 1))

    def test_to_key_all_false(self):
        """Test to_key with all false values."""
        state = State.initial()
        key = state.to_key()
        self.assertEqual(key, (0, 0, 0, 0, 0))

    def test_to_key_all_true(self):
        """Test to_key with all true values."""
        state = State(has_plan=True, has_code=True, has_error=True, tests_pass=True, iteration=3)
        key = state.to_key()
        self.assertEqual(key, (1, 1, 1, 1, 3))

    def test_to_index(self):
        """Test converting state to integer index."""
        state = State.initial()
        self.assertEqual(state.to_index(), 0)

        state = State(has_plan=True)
        self.assertEqual(state.to_index(), 32)

        state = State(has_plan=True, has_code=True, has_error=True, tests_pass=True, iteration=3)
        self.assertEqual(state.to_index(), 63)

    def test_from_index(self):
        """Test creating state from index."""
        state = State.from_index(0)
        self.assertFalse(state.has_plan)
        self.assertFalse(state.has_code)
        self.assertEqual(state.iteration, 0)

        state = State.from_index(32)
        self.assertTrue(state.has_plan)
        self.assertFalse(state.has_code)

        state = State.from_index(63)
        self.assertTrue(state.has_plan)
        self.assertTrue(state.has_code)
        self.assertTrue(state.has_error)
        self.assertTrue(state.tests_pass)
        self.assertEqual(state.iteration, 3)

    def test_index_roundtrip(self):
        """Test that to_index and from_index are inverses."""
        for i in range(64):
            state = State.from_index(i)
            self.assertEqual(state.to_index(), i)

    def test_is_terminal_success(self):
        """Test is_terminal when tests pass."""
        state = State(tests_pass=True)
        self.assertTrue(state.is_terminal())

    def test_is_terminal_timeout(self):
        """Test is_terminal when iteration limit reached."""
        state = State(iteration=5)
        self.assertTrue(state.is_terminal())

        state = State(iteration=6)
        self.assertTrue(state.is_terminal())

    def test_is_terminal_not_terminal(self):
        """Test is_terminal for non-terminal states."""
        state = State(iteration=3)
        self.assertFalse(state.is_terminal())

        state = State(tests_pass=False, iteration=4)
        self.assertFalse(state.is_terminal())

    def test_copy(self):
        """Test copying a state."""
        original = State(has_plan=True, has_code=True, iteration=2)
        copied = original.copy()

        self.assertEqual(original.has_plan, copied.has_plan)
        self.assertEqual(original.has_code, copied.has_code)
        self.assertEqual(original.iteration, copied.iteration)

        # Modifying copy shouldn't affect original
        copied.has_plan = False
        self.assertTrue(original.has_plan)

    def test_repr(self):
        """Test string representation."""
        state = State.initial()
        repr_str = repr(state)
        self.assertIn("State", repr_str)
        self.assertIn("empty", repr_str)

        state = State(has_plan=True, has_code=True)
        repr_str = repr(state)
        self.assertIn("plan", repr_str)
        self.assertIn("code", repr_str)


class TestActions(unittest.TestCase):
    """Test action definitions."""

    def test_actions_defined(self):
        """Test that all actions are defined."""
        self.assertEqual(len(ACTIONS), 4)
        self.assertIn("planner", ACTIONS)
        self.assertIn("coder", ACTIONS)
        self.assertIn("tester", ACTIONS)
        self.assertIn("debugger", ACTIONS)

    def test_action_to_idx(self):
        """Test action to index mapping."""
        self.assertEqual(ACTION_TO_IDX["planner"], 0)
        self.assertEqual(ACTION_TO_IDX["coder"], 1)
        self.assertEqual(ACTION_TO_IDX["tester"], 2)
        self.assertEqual(ACTION_TO_IDX["debugger"], 3)

    def test_idx_to_action(self):
        """Test index to action mapping."""
        self.assertEqual(IDX_TO_ACTION[0], "planner")
        self.assertEqual(IDX_TO_ACTION[1], "coder")
        self.assertEqual(IDX_TO_ACTION[2], "tester")
        self.assertEqual(IDX_TO_ACTION[3], "debugger")

    def test_action_mappings_consistent(self):
        """Test that action mappings are consistent."""
        for action, idx in ACTION_TO_IDX.items():
            self.assertEqual(IDX_TO_ACTION[idx], action)


class TestGetValidActions(unittest.TestCase):
    """Test get_valid_actions function."""

    def test_initial_state_actions(self):
        """Test valid actions for initial state."""
        state = State.initial()
        valid = get_valid_actions(state)

        self.assertIn("planner", valid)
        self.assertIn("coder", valid)
        self.assertNotIn("tester", valid)  # No code yet
        self.assertNotIn("debugger", valid)  # No error yet

    def test_with_code_actions(self):
        """Test valid actions when code exists."""
        state = State(has_code=True)
        valid = get_valid_actions(state)

        self.assertIn("planner", valid)
        self.assertIn("coder", valid)
        self.assertIn("tester", valid)  # Can test with code
        self.assertNotIn("debugger", valid)  # No error yet

    def test_with_error_actions(self):
        """Test valid actions when error exists."""
        state = State(has_code=True, has_error=True)
        valid = get_valid_actions(state)

        self.assertIn("planner", valid)
        self.assertIn("coder", valid)
        self.assertIn("tester", valid)
        self.assertIn("debugger", valid)  # Can debug with error

    def test_no_debugger_without_code(self):
        """Test that debugger requires code."""
        state = State(has_error=True)  # Error but no code
        valid = get_valid_actions(state)

        self.assertNotIn("debugger", valid)


class TestRewardFunction(unittest.TestCase):
    """Test RewardFunction class."""

    def setUp(self):
        """Create reward function for each test."""
        self.rf = RewardFunction()

    def test_reward_constants_exist(self):
        """Test that all reward constants exist."""
        expected_keys = [
            "task_success", "task_timeout", "progress_plan",
            "progress_code", "tests_pass_partial", "error_fixed",
            "redundant_action", "invalid_action", "step_cost"
        ]
        for key in expected_keys:
            self.assertIn(key, REWARDS)

    def test_reward_signs(self):
        """Test that reward signs are correct."""
        self.assertGreater(REWARDS["task_success"], 0)
        self.assertLess(REWARDS["task_timeout"], 0)
        self.assertGreater(REWARDS["progress_plan"], 0)
        self.assertLess(REWARDS["redundant_action"], 0)
        self.assertLess(REWARDS["step_cost"], 0)

    def test_task_success_reward(self):
        """Test reward for task success."""
        state = State(has_plan=True, has_code=True)
        next_state = State(has_plan=True, has_code=True, tests_pass=True)

        reward = self.rf.calculate(state, "tester", next_state, done=True)

        # Should include success reward + step cost
        self.assertGreater(reward, 0)
        self.assertAlmostEqual(
            reward,
            REWARDS["task_success"] + REWARDS["step_cost"]
        )

    def test_task_timeout_reward(self):
        """Test reward for task timeout."""
        state = State(iteration=4)
        next_state = State(iteration=5)

        reward = self.rf.calculate(state, "coder", next_state, done=True)

        # Should include timeout penalty + step cost
        self.assertLess(reward, 0)
        self.assertAlmostEqual(
            reward,
            REWARDS["task_timeout"] + REWARDS["step_cost"]
        )

    def test_progress_plan_reward(self):
        """Test reward for making a plan."""
        state = State.initial()
        next_state = State(has_plan=True)

        reward = self.rf.calculate(state, "planner", next_state, done=False)

        expected = REWARDS["progress_plan"] + REWARDS["step_cost"]
        self.assertAlmostEqual(reward, expected)

    def test_progress_code_reward(self):
        """Test reward for writing code."""
        state = State(has_plan=True)
        next_state = State(has_plan=True, has_code=True)

        reward = self.rf.calculate(state, "coder", next_state, done=False)

        expected = REWARDS["progress_code"] + REWARDS["step_cost"]
        self.assertAlmostEqual(reward, expected)

    def test_redundant_plan_penalty(self):
        """Test penalty for redundant planning."""
        state = State(has_plan=True)
        next_state = State(has_plan=True)

        reward = self.rf.calculate(state, "planner", next_state, done=False)

        expected = REWARDS["redundant_action"] + REWARDS["step_cost"]
        self.assertAlmostEqual(reward, expected)

    def test_invalid_tester_penalty(self):
        """Test penalty for testing without code."""
        state = State.initial()
        next_state = State.initial()

        reward = self.rf.calculate(state, "tester", next_state, done=False)

        expected = REWARDS["invalid_action"] + REWARDS["step_cost"]
        self.assertAlmostEqual(reward, expected)

    def test_invalid_debugger_penalty(self):
        """Test penalty for debugging without error."""
        state = State(has_code=True, has_error=False)
        next_state = State(has_code=True, has_error=False)

        reward = self.rf.calculate(state, "debugger", next_state, done=False)

        expected = REWARDS["invalid_action"] + REWARDS["step_cost"]
        self.assertAlmostEqual(reward, expected)

    def test_error_fixed_reward(self):
        """Test reward for fixing an error."""
        state = State(has_code=True, has_error=True)
        next_state = State(has_code=True, has_error=False)

        reward = self.rf.calculate(state, "debugger", next_state, done=False)

        expected = REWARDS["error_fixed"] + REWARDS["step_cost"]
        self.assertAlmostEqual(reward, expected)

    def test_tests_pass_partial_reward(self):
        """Test reward for partial test success."""
        state = State(has_code=True)
        next_state = State(has_code=True, tests_pass=True)

        # Not done yet, but tests pass
        reward = self.rf.calculate(state, "tester", next_state, done=False)

        expected = REWARDS["tests_pass_partial"] + REWARDS["step_cost"]
        self.assertAlmostEqual(reward, expected)

    def test_custom_rewards(self):
        """Test using custom reward values."""
        custom_rewards = {
            "task_success": 100.0,
            "task_timeout": -50.0,
            "progress_plan": 1.0,
            "progress_code": 1.0,
            "tests_pass_partial": 5.0,
            "error_fixed": 2.0,
            "redundant_action": -1.0,
            "invalid_action": -1.0,
            "step_cost": -0.5,
        }
        rf = RewardFunction(rewards=custom_rewards)

        state = State.initial()
        next_state = State(has_plan=True)
        reward = rf.calculate(state, "planner", next_state, done=False)

        self.assertAlmostEqual(reward, 1.0 + (-0.5))  # progress + step cost

    def test_get_reward_description(self):
        """Test getting reward description."""
        description = self.rf.get_reward_description()

        self.assertIn("Reward Structure", description)
        self.assertIn("task_success", description)
        self.assertIn("step_cost", description)


# Try to import SimulatedEnv at module level
try:
    from environment.simulated_env import SimulatedEnv
    SIMULATED_ENV_AVAILABLE = True
except ImportError:
    SIMULATED_ENV_AVAILABLE = False
    SimulatedEnv = None


@unittest.skipUnless(SIMULATED_ENV_AVAILABLE, "SimulatedEnv not available")
class TestSimulatedEnv(unittest.TestCase):
    """Test SimulatedEnv class (if available)."""

    def setUp(self):
        """Create environment for each test."""
        self.env = SimulatedEnv()

    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        self.assertIsInstance(state, State)
        self.assertFalse(state.has_plan)
        self.assertFalse(state.has_code)
        self.assertEqual(state.iteration, 0)

    def test_step_returns_correct_types(self):
        """Test that step returns correct types."""
        self.env.reset()
        next_state, reward, done = self.env.step("planner")

        self.assertIsInstance(next_state, State)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_step_increments_iteration(self):
        """Test that step increments iteration."""
        state = self.env.reset()
        self.assertEqual(state.iteration, 0)

        next_state, _, _ = self.env.step("planner")
        self.assertEqual(next_state.iteration, 1)

    def test_get_valid_actions(self):
        """Test getting valid actions from environment."""
        self.env.reset()
        valid = self.env.get_valid_actions()

        self.assertIsInstance(valid, list)
        self.assertIn("planner", valid)
        self.assertIn("coder", valid)

    def test_episode_terminates(self):
        """Test that episodes eventually terminate."""
        self.env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            valid = self.env.get_valid_actions()
            action = valid[0]
            _, _, done = self.env.step(action)
            steps += 1

        self.assertTrue(done or steps < 100)


class TestEnvironmentIntegration(unittest.TestCase):
    """Integration tests for environment components."""

    def test_state_and_rewards_together(self):
        """Test state transitions with rewards."""
        rf = RewardFunction()

        # Simulate a successful episode
        s0 = State.initial()
        s1 = State(has_plan=True, iteration=1)
        s2 = State(has_plan=True, has_code=True, iteration=2)
        s3 = State(has_plan=True, has_code=True, tests_pass=True, iteration=3)

        r1 = rf.calculate(s0, "planner", s1, done=False)
        r2 = rf.calculate(s1, "coder", s2, done=False)
        r3 = rf.calculate(s2, "tester", s3, done=True)

        # Progress rewards should be positive (minus step cost)
        self.assertGreater(r1, REWARDS["step_cost"])
        self.assertGreater(r2, REWARDS["step_cost"])

        # Success reward should be large positive
        self.assertGreater(r3, 5.0)

    def test_valid_actions_progression(self):
        """Test valid actions change as state progresses."""
        # Initial state
        s = State.initial()
        valid = get_valid_actions(s)
        self.assertEqual(len(valid), 2)  # planner, coder

        # After getting code
        s.has_code = True
        valid = get_valid_actions(s)
        self.assertEqual(len(valid), 3)  # + tester

        # After getting error
        s.has_error = True
        valid = get_valid_actions(s)
        self.assertEqual(len(valid), 4)  # + debugger


if __name__ == "__main__":
    unittest.main()
