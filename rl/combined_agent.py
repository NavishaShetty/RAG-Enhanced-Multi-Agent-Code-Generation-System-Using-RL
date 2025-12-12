"""
Combined RL Agent - Q-Learning + Thompson Sampling.

This agent combines Q-Learning for value estimation with
Thompson Sampling for principled exploration.
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
from rl.q_learning import QLearningAgent
from rl.thompson_sampling import ThompsonSamplingAgent


class CombinedAgent:
    """
    Combined Q-Learning + Thompson Sampling agent.

    Uses Q-values for exploitation and Thompson Sampling for exploration.
    The key idea: sample from uncertainty around Q-values.

    sampled_q = Q(s,a) + exploration_bonus * (beta_sample - 0.5)

    This provides:
    - Q-learning's ability to learn from temporal differences
    - Thompson Sampling's principled uncertainty-based exploration
    """

    def __init__(
        self,
        actions: List[str] = None,
        alpha: float = 0.1,
        gamma: float = 0.95,
        exploration_scale: float = 2.0
    ):
        """
        Initialize combined agent.

        Args:
            actions: List of available actions
            alpha: Q-learning rate
            gamma: Discount factor
            exploration_scale: Scale factor for exploration bonus
        """
        self.actions = actions or ACTIONS
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_scale = exploration_scale

        # Q-learning component for value estimation
        self.q_agent = QLearningAgent(
            actions=self.actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=0.0  # We use Thompson Sampling instead
        )

        # Thompson Sampling component for exploration
        self.ts_agent = ThompsonSamplingAgent(
            actions=self.actions,
            prior_alpha=1.0,
            prior_beta=1.0
        )

        # Statistics
        self.total_updates = 0

    def get_q_values(self, state: State) -> Dict[str, float]:
        """Get Q-values for all actions in a state."""
        return self.q_agent.get_q_values(state)

    def get_sampled_values(self, state: State) -> Dict[str, float]:
        """
        Get sampled action values combining Q-values and exploration bonus.

        sampled_q = Q(s,a) + scale * (ts_sample - 0.5)

        Args:
            state: Current state

        Returns:
            Dictionary mapping action -> sampled value
        """
        q_values = self.q_agent.get_q_values(state)
        ts_samples = self.ts_agent.sample(state)

        sampled = {}
        for action in self.actions:
            # Combine Q-value with exploration bonus
            exploration_bonus = self.ts_agent.get_exploration_bonus(state, action)
            sampled[action] = (
                q_values[action] +
                self.exploration_scale * exploration_bonus * (ts_samples[action] - 0.5)
            )
        return sampled

    def choose_action(
        self,
        state: State,
        valid_actions: Optional[List[str]] = None,
        explore: bool = True
    ) -> str:
        """
        Choose action using combined Q-values and Thompson Sampling.

        Args:
            state: Current state
            valid_actions: List of valid actions (all if None)
            explore: Whether to use exploration (if False, pure Q greedy)

        Returns:
            Chosen action
        """
        if valid_actions is None:
            valid_actions = self.actions

        if explore:
            # Use sampled values (Q + exploration bonus)
            values = self.get_sampled_values(state)
        else:
            # Pure exploitation - use Q-values
            values = self.q_agent.get_q_values(state)

        # Filter to valid actions
        valid_values = {a: values[a] for a in valid_actions}

        # Return action with highest value (random tie-breaking)
        max_val = max(valid_values.values())
        best_actions = [a for a, v in valid_values.items() if abs(v - max_val) < 1e-6]

        return random.choice(best_actions)

    def update(
        self,
        state: State,
        action: str,
        reward: float,
        next_state: State,
        done: bool
    ):
        """
        Update both Q-learning and Thompson Sampling components.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode is done
        """
        # Update Q-learning component
        self.q_agent.update(state, action, reward, next_state, done)

        # Update Thompson Sampling component
        self.ts_agent.update(state, action, reward)

        self.total_updates += 1

    def get_policy(self) -> Dict[Tuple, str]:
        """
        Get the learned policy (greedy w.r.t. Q-values).

        Returns:
            Dictionary mapping state_key -> best_action
        """
        return self.q_agent.get_policy()

    def get_uncertainty(self, state: State) -> Dict[str, float]:
        """
        Get uncertainty for all actions in a state.

        Args:
            state: State

        Returns:
            Dictionary mapping action -> uncertainty
        """
        return {
            action: self.ts_agent.get_uncertainty(state, action)
            for action in self.actions
        }

    def save(self, filepath: str):
        """
        Save agent parameters to JSON file.

        Args:
            filepath: Path to save file
        """
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "exploration_scale": self.exploration_scale,
            "total_updates": self.total_updates,
            "q_table": {
                str(k): v for k, v in self.q_agent.q_table.items()
            },
            "ts_alpha": {
                str(k): v for k, v in self.ts_agent.alpha.items()
            },
            "ts_beta": {
                str(k): v for k, v in self.ts_agent.beta.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load agent parameters from JSON file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.exploration_scale = data.get("exploration_scale", self.exploration_scale)
        self.total_updates = data.get("total_updates", 0)

        # Load Q-table
        self.q_agent.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})
        for key_str, values in data.get("q_table", {}).items():
            self.q_agent.q_table[eval(key_str)] = values

        # Load Thompson Sampling parameters
        self.ts_agent.alpha = defaultdict(
            lambda: {a: self.ts_agent.prior_alpha for a in self.actions}
        )
        self.ts_agent.beta = defaultdict(
            lambda: {a: self.ts_agent.prior_beta for a in self.actions}
        )
        for key_str, values in data.get("ts_alpha", {}).items():
            self.ts_agent.alpha[eval(key_str)] = values
        for key_str, values in data.get("ts_beta", {}).items():
            self.ts_agent.beta[eval(key_str)] = values


if __name__ == "__main__":
    print("Testing CombinedAgent...")

    # Create agent
    agent = CombinedAgent(alpha=0.1, gamma=0.95)

    # Test basic functionality
    s = State.initial()
    print(f"Initial Q-values: {agent.get_q_values(s)}")
    print(f"Initial sampled values: {agent.get_sampled_values(s)}")
    print(f"Uncertainty: {agent.get_uncertainty(s)}")

    # Test action selection diversity
    print("\nAction selection (should have exploration):")
    action_counts = defaultdict(int)
    for _ in range(100):
        action = agent.choose_action(s, ["planner", "coder"])
        action_counts[action] += 1
    print(f"  {dict(action_counts)}")

    # Train on simulated environment
    print("\nTraining on simulated environment...")
    from environment.simulated_env import SimulatedEnv

    env = SimulatedEnv()
    successes = 0
    total_reward = 0

    for episode in range(2000):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid, explore=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

        if state.tests_pass:
            successes += 1
        total_reward += ep_reward

    print(f"Success rate: {successes}/2000 ({100*successes/2000:.0f}%)")
    print(f"Average reward: {total_reward/2000:.2f}")
    print(f"Total updates: {agent.total_updates}")

    # Compare explore vs exploit
    print("\nCompare explore vs exploit on initial state:")
    explore_counts = defaultdict(int)
    exploit_counts = defaultdict(int)
    for _ in range(100):
        explore_counts[agent.choose_action(s, explore=True)] += 1
        exploit_counts[agent.choose_action(s, explore=False)] += 1
    print(f"  Explore: {dict(explore_counts)}")
    print(f"  Exploit: {dict(exploit_counts)}")

    # Show learned policy
    print("\nLearned policy (sample states):")
    for i in [0, 32, 48, 56]:
        s = State.from_index(i)
        q_vals = agent.get_q_values(s)
        best = max(q_vals.keys(), key=lambda a: q_vals[a])
        print(f"  {s}: {best} (Q={q_vals[best]:.2f})")

    print("\nCombined agent working!")
