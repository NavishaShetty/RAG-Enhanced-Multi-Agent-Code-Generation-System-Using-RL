"""
Thompson Sampling Agent Implementation.

Implements Thompson Sampling for exploration in the code generation task.
Uses Beta distributions to model uncertainty about action values.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from environment.state import State, ACTIONS


class ThompsonSamplingAgent:
    """
    Thompson Sampling agent for exploration.

    For each state-action pair, maintains a Beta distribution:
    theta(s,a) ~ Beta(alpha_sa, beta_sa)

    Action selection samples from these distributions and picks
    the action with the highest sampled value.

    This provides principled exploration based on uncertainty.
    """

    def __init__(
        self,
        actions: List[str] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        """
        Initialize Thompson Sampling agent.

        Args:
            actions: List of available actions
            prior_alpha: Initial alpha for Beta prior (optimism)
            prior_beta: Initial beta for Beta prior (pessimism)
        """
        self.actions = actions or ACTIONS
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Beta distribution parameters: maps (state_key, action) -> (alpha, beta)
        # Alpha tracks "successes", beta tracks "failures"
        self.alpha: Dict[Tuple, Dict[str, float]] = defaultdict(
            lambda: {a: self.prior_alpha for a in self.actions}
        )
        self.beta: Dict[Tuple, Dict[str, float]] = defaultdict(
            lambda: {a: self.prior_beta for a in self.actions}
        )

        # Statistics
        self.total_updates = 0

    def sample(self, state: State) -> Dict[str, float]:
        """
        Sample from Beta distributions for all actions.

        Args:
            state: Current state

        Returns:
            Dictionary mapping action -> sampled value
        """
        state_key = state.to_key()
        samples = {}
        for action in self.actions:
            alpha = self.alpha[state_key][action]
            beta = self.beta[state_key][action]
            samples[action] = np.random.beta(alpha, beta)
        return samples

    def get_mean(self, state: State) -> Dict[str, float]:
        """
        Get mean of Beta distributions (expected success rate).

        Args:
            state: Current state

        Returns:
            Dictionary mapping action -> mean value
        """
        state_key = state.to_key()
        means = {}
        for action in self.actions:
            alpha = self.alpha[state_key][action]
            beta = self.beta[state_key][action]
            means[action] = alpha / (alpha + beta)
        return means

    def get_uncertainty(self, state: State, action: str) -> float:
        """
        Get uncertainty (variance) for a state-action pair.

        Args:
            state: State
            action: Action

        Returns:
            Variance of Beta distribution
        """
        state_key = state.to_key()
        alpha = self.alpha[state_key][action]
        beta = self.beta[state_key][action]
        # Variance of Beta distribution
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return var

    def choose_action(
        self,
        state: State,
        valid_actions: Optional[List[str]] = None
    ) -> str:
        """
        Choose action using Thompson Sampling.

        Samples from Beta distributions and picks action with highest sample.

        Args:
            state: Current state
            valid_actions: List of valid actions (all if None)

        Returns:
            Chosen action
        """
        if valid_actions is None:
            valid_actions = self.actions

        samples = self.sample(state)

        # Filter to valid actions
        valid_samples = {a: samples[a] for a in valid_actions}

        # Return action with highest sample
        return max(valid_samples.keys(), key=lambda a: valid_samples[a])

    def update(self, state: State, action: str, reward: float):
        """
        Update Beta distribution parameters based on reward.

        Positive rewards increase alpha (successes).
        Negative/zero rewards increase beta (failures).

        Args:
            state: State
            action: Action taken
            reward: Reward received
        """
        state_key = state.to_key()

        # Scale reward to [0, 1] range for Beta distribution
        # Assuming rewards typically in [-5, 10] range
        scaled = (reward + 5) / 15  # Maps [-5, 10] to [0, 1]
        scaled = max(0, min(1, scaled))  # Clip to [0, 1]

        # Update: success increases alpha, failure increases beta
        if scaled > 0.5:
            self.alpha[state_key][action] += scaled
        else:
            self.beta[state_key][action] += (1 - scaled)

        self.total_updates += 1

    def get_exploration_bonus(self, state: State, action: str) -> float:
        """
        Get exploration bonus based on uncertainty.

        Higher uncertainty = more exploration value.

        Args:
            state: State
            action: Action

        Returns:
            Exploration bonus value
        """
        uncertainty = self.get_uncertainty(state, action)
        # Scale uncertainty to reasonable bonus
        return np.sqrt(uncertainty) * 2

    def save(self, filepath: str):
        """
        Save parameters to JSON file.

        Args:
            filepath: Path to save file
        """
        data = {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "total_updates": self.total_updates,
            "alpha": {str(k): v for k, v in self.alpha.items()},
            "beta": {str(k): v for k, v in self.beta.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load parameters from JSON file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.prior_alpha = data.get("prior_alpha", self.prior_alpha)
        self.prior_beta = data.get("prior_beta", self.prior_beta)
        self.total_updates = data.get("total_updates", 0)

        self.alpha = defaultdict(lambda: {a: self.prior_alpha for a in self.actions})
        self.beta = defaultdict(lambda: {a: self.prior_beta for a in self.actions})

        for key_str, values in data["alpha"].items():
            self.alpha[eval(key_str)] = values
        for key_str, values in data["beta"].items():
            self.beta[eval(key_str)] = values


if __name__ == "__main__":
    print("Testing ThompsonSamplingAgent...")

    # Create agent
    agent = ThompsonSamplingAgent()

    # Test sampling
    s = State.initial()
    print(f"Initial samples for {s}:")
    for _ in range(3):
        samples = agent.sample(s)
        print(f"  {samples}")

    print(f"\nMeans: {agent.get_mean(s)}")
    print(f"Uncertainty (planner): {agent.get_uncertainty(s, 'planner'):.4f}")

    # Test action selection
    print("\nAction selections (should be diverse due to sampling):")
    action_counts = defaultdict(int)
    for _ in range(100):
        action = agent.choose_action(s, ["planner", "coder"])
        action_counts[action] += 1
    print(f"  {dict(action_counts)}")

    # Test updates
    print("\nTesting updates...")
    agent.update(s, "planner", 5.0)  # Good reward
    agent.update(s, "planner", 3.0)
    agent.update(s, "coder", -2.0)  # Bad reward
    agent.update(s, "coder", -1.0)

    print(f"After updates, means: {agent.get_mean(s)}")
    print(f"Planner uncertainty: {agent.get_uncertainty(s, 'planner'):.4f}")
    print(f"Coder uncertainty: {agent.get_uncertainty(s, 'coder'):.4f}")

    # Train on simulated environment
    print("\nTraining on simulated environment...")
    from environment.simulated_env import SimulatedEnv

    env = SimulatedEnv()
    successes = 0

    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward)
            state = next_state

        if state.tests_pass:
            successes += 1

    print(f"Success rate: {successes}/1000 ({100*successes/1000:.0f}%)")
    print(f"Total updates: {agent.total_updates}")

    print("\nThompson Sampling agent working!")
