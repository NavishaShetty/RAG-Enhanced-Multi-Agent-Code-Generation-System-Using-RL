"""
Simulated Environment for Fast RL Training.

This environment simulates agent success/failure probabilities
to enable rapid training without calling the LLM API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from typing import Tuple, Optional
from dataclasses import dataclass

from environment.state import State, ACTIONS
from environment.rewards import RewardFunction


@dataclass
class SimulationConfig:
    """Configuration for simulation probabilities.

    These probabilities are tuned to better match real LLM behavior
    based on validation experiments.
    """
    planner_success: float = 0.95
    coder_success_with_plan: float = 0.85      # Tuned from 0.70 - LLM is quite good
    coder_success_without_plan: float = 0.60   # Tuned from 0.30 - LLM can code simple tasks
    tester_finds_error: float = 0.40  # Probability of finding bugs if there are any
    debugger_fixes_error: float = 0.70         # Tuned from 0.50 - debugger is effective
    max_iterations: int = 5


class SimulatedEnv:
    """
    Fast simulated environment for RL training.

    Simulates agent success/failure without LLM calls.
    Should run >1000 episodes per second.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulated environment.

        Args:
            config: Simulation configuration (probabilities)
        """
        self.config = config or SimulationConfig()
        self.reward_fn = RewardFunction()
        self.state: Optional[State] = None
        self._task_difficulty = 0.5  # Current task difficulty (0-1)

    def reset(self, task: str = "simulated_task") -> State:
        """
        Reset environment for new episode.

        Args:
            task: Task description (used to set difficulty)

        Returns:
            Initial state
        """
        self.state = State.initial()

        # Set task difficulty based on task description
        task_lower = task.lower()
        if "fibonacci" in task_lower or "binary search" in task_lower:
            self._task_difficulty = 0.8
        elif "palindrome" in task_lower or "factorial" in task_lower:
            self._task_difficulty = 0.6
        elif "reverse" in task_lower or "sum" in task_lower or "even" in task_lower:
            self._task_difficulty = 0.3
        else:
            self._task_difficulty = 0.5

        return self.state.copy()

    def step(self, action: str) -> Tuple[State, float, bool]:
        """
        Take an action in the environment.

        Args:
            action: One of ["planner", "coder", "tester", "debugger"]

        Returns:
            Tuple of (next_state, reward, done)
        """
        if self.state is None:
            raise ValueError("Call reset() before step()")

        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        old_state = self.state.copy()
        new_state = self._simulate_action(action)
        self.state = new_state

        # Check if done
        done = new_state.is_terminal() or new_state.iteration >= self.config.max_iterations

        # Calculate reward
        reward = self.reward_fn.calculate(old_state, action, new_state, done)

        return new_state.copy(), reward, done

    def _simulate_action(self, action: str) -> State:
        """
        Simulate the effect of an action.

        Args:
            action: Action to simulate

        Returns:
            New state after action
        """
        new_state = self.state.copy()
        new_state.iteration += 1

        if action == "planner":
            # Planning almost always succeeds
            if random.random() < self.config.planner_success:
                new_state.has_plan = True

        elif action == "coder":
            # Coding success depends on having a plan
            if new_state.has_plan:
                success_prob = self.config.coder_success_with_plan
            else:
                success_prob = self.config.coder_success_without_plan

            # Adjust for task difficulty
            success_prob *= (1 - 0.3 * self._task_difficulty)

            if random.random() < success_prob:
                new_state.has_code = True
                new_state.has_error = False
            else:
                new_state.has_code = True
                new_state.has_error = True  # Code has bugs

        elif action == "tester":
            if new_state.has_code:
                # Tester determines if code passes
                if new_state.has_error:
                    # Code has bugs - tester should find them
                    if random.random() < self.config.tester_finds_error:
                        new_state.tests_pass = False
                    else:
                        # Tester missed the bug (rare)
                        new_state.tests_pass = random.random() > 0.7
                else:
                    # Code is good - high chance of passing
                    new_state.tests_pass = random.random() > 0.1

        elif action == "debugger":
            if new_state.has_code and new_state.has_error:
                # Try to fix the error
                fix_prob = self.config.debugger_fixes_error * (1 - 0.2 * self._task_difficulty)
                if random.random() < fix_prob:
                    new_state.has_error = False
                    new_state.tests_pass = False  # Need to re-test

        return new_state

    def get_valid_actions(self) -> list:
        """
        Get valid actions for current state.

        Returns:
            List of valid action names
        """
        if self.state is None:
            return []

        valid = ["planner", "coder"]

        if self.state.has_code:
            valid.append("tester")

        if self.state.has_code and self.state.has_error:
            valid.append("debugger")

        return valid


def benchmark_speed():
    """Benchmark simulation speed."""
    import time

    env = SimulatedEnv()
    actions = ["planner", "coder", "tester", "debugger"]

    num_episodes = 10000
    start = time.time()

    for _ in range(num_episodes):
        env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            action = random.choice(valid)
            _, _, done = env.step(action)

    elapsed = time.time() - start
    eps_per_sec = num_episodes / elapsed

    print(f"Ran {num_episodes} episodes in {elapsed:.2f}s")
    print(f"Speed: {eps_per_sec:.0f} episodes/second")
    return eps_per_sec


if __name__ == "__main__":
    print("Testing SimulatedEnv...")

    env = SimulatedEnv()

    # Run a few episodes
    successes = 0
    total_episodes = 100

    for ep in range(total_episodes):
        state = env.reset("Write a function that reverses a string")
        done = False
        total_reward = 0
        steps = 0

        while not done:
            valid = env.get_valid_actions()
            # Use simple policy: plan -> code -> test -> debug loop
            if not state.has_plan:
                action = "planner"
            elif not state.has_code:
                action = "coder"
            elif state.has_error:
                action = "debugger"
            else:
                action = "tester"

            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        if state.tests_pass:
            successes += 1

        if ep < 3:
            print(f"Episode {ep+1}: {'SUCCESS' if state.tests_pass else 'FAIL'}, "
                  f"steps={steps}, reward={total_reward:.2f}")

    print(f"\nSuccess rate: {successes}/{total_episodes} ({100*successes/total_episodes:.0f}%)")

    print("\n" + "="*50)
    print("Benchmarking speed...")
    benchmark_speed()

    print("\nSimulated environment working!")
