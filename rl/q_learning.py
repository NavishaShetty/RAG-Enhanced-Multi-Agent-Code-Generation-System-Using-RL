"""
Q-Learning Agent Implementation.

Implements tabular Q-Learning for the code generation orchestration task.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from environment.state import State, ACTIONS


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Q-Learning update formula:
    Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

    Where:
    - alpha = learning rate
    - gamma = discount factor
    - r = immediate reward
    - max_a' Q(s',a') = maximum Q-value in next state
    """

    def __init__(
        self,
        actions: List[str] = None,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1
    ):
        """
        Initialize Q-Learning agent.

        Args:
            actions: List of available actions
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon: Exploration rate for epsilon-greedy (0-1)
        """
        self.actions = actions or ACTIONS
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: maps (state_key, action) -> Q-value
        self.q_table: Dict[Tuple, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )

        # Training statistics
        self.total_updates = 0

    def get_q_values(self, state: State) -> Dict[str, float]:
        """
        Get Q-values for all actions in a state.

        Args:
            state: Current state

        Returns:
            Dictionary mapping action -> Q-value
        """
        state_key = state.to_key()
        return dict(self.q_table[state_key])

    def get_q_value(self, state: State, action: str) -> float:
        """
        Get Q-value for a specific state-action pair.

        Args:
            state: State
            action: Action

        Returns:
            Q-value
        """
        return self.q_table[state.to_key()][action]

    def choose_action(
        self,
        state: State,
        valid_actions: Optional[List[str]] = None,
        explore: bool = True
    ) -> str:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state
            valid_actions: List of valid actions (all if None)
            explore: Whether to use epsilon-greedy exploration

        Returns:
            Chosen action
        """
        if valid_actions is None:
            valid_actions = self.actions

        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Greedy action selection
        q_values = self.get_q_values(state)

        # Filter to valid actions
        valid_q = {a: q_values[a] for a in valid_actions}

        # Get action with max Q-value (random tie-breaking)
        max_q = max(valid_q.values())
        best_actions = [a for a, q in valid_q.items() if q == max_q]

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
        Update Q-value using Q-learning formula.

        Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode is done
        """
        state_key = state.to_key()
        current_q = self.q_table[state_key][action]

        if done:
            # Terminal state - no future rewards
            target = reward
        else:
            # Bootstrap from next state
            next_q_values = self.get_q_values(next_state)
            max_next_q = max(next_q_values.values())
            target = reward + self.gamma * max_next_q

        # Q-learning update
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state_key][action] = new_q

        self.total_updates += 1

    def get_policy(self) -> Dict[Tuple, str]:
        """
        Get the learned policy (best action for each state).

        Returns:
            Dictionary mapping state_key -> best_action
        """
        policy = {}
        for state_key, q_values in self.q_table.items():
            best_action = max(q_values.keys(), key=lambda a: q_values[a])
            policy[state_key] = best_action
        return policy

    def get_state_value(self, state: State) -> float:
        """
        Get the value of a state (max Q-value).

        Args:
            state: State

        Returns:
            State value V(s) = max_a Q(s,a)
        """
        q_values = self.get_q_values(state)
        return max(q_values.values())

    def save(self, filepath: str):
        """
        Save Q-table to JSON file.

        Args:
            filepath: Path to save file
        """
        # Convert tuple keys to strings for JSON
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "q_table": {
                str(k): v for k, v in self.q_table.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load Q-table from JSON file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.total_updates = data.get("total_updates", 0)

        # Convert string keys back to tuples
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})
        for key_str, values in data["q_table"].items():
            key = eval(key_str)  # Convert string back to tuple
            self.q_table[key] = values

    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """
        Decay exploration rate.

        Args:
            decay_rate: Multiplicative decay factor
            min_epsilon: Minimum epsilon value
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


if __name__ == "__main__":
    print("Testing QLearningAgent...")

    # Create agent
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1)

    # Test basic functionality
    s1 = State.initial()
    print(f"Initial Q-values for {s1}: {agent.get_q_values(s1)}")

    # Simulate some updates
    action = agent.choose_action(s1)
    print(f"Chosen action: {action}")

    s2 = State(has_plan=True)
    agent.update(s1, "planner", 0.2, s2, done=False)
    print(f"After update: {agent.get_q_values(s1)}")

    # Test many updates
    print("\nSimulating training...")
    from environment.simulated_env import SimulatedEnv

    env = SimulatedEnv()

    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid, explore=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        if episode % 200 == 0:
            agent.decay_epsilon()

    print(f"Total updates: {agent.total_updates}")
    print(f"Final epsilon: {agent.epsilon:.4f}")

    # Print learned policy for a few states
    print("\nLearned policy (sample states):")
    for i in [0, 32, 48, 56]:
        s = State.from_index(i)
        q_vals = agent.get_q_values(s)
        best = max(q_vals.keys(), key=lambda a: q_vals[a])
        print(f"  {s}: {best} (Q={q_vals[best]:.2f})")

    print("\nQ-Learning agent working!")
