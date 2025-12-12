"""
State Representation for the RL Environment.

The state captures the current status of the code generation pipeline.
Designed for tabular RL (small discrete state space).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class State:
    """
    State representation for the RL agent.

    State features (keep simple for tabular RL):
    - has_plan: bool (2 values)
    - has_code: bool (2 values)
    - has_error: bool (2 values)
    - tests_pass: bool (2 values)
    - iteration_bucket: int (0, 1, 2, 3+ = 4 values)

    Total state space: 2 * 2 * 2 * 2 * 4 = 64 states
    """
    has_plan: bool = False
    has_code: bool = False
    has_error: bool = False
    tests_pass: bool = False
    iteration: int = 0

    @property
    def iteration_bucket(self) -> int:
        """
        Bucket the iteration count for tabular RL.

        Returns:
            0, 1, 2, or 3 (for 3+)
        """
        return min(self.iteration, 3)

    def to_features(self) -> dict:
        """
        Convert state to feature dictionary.

        Returns:
            Dictionary of state features
        """
        return {
            "has_plan": self.has_plan,
            "has_code": self.has_code,
            "has_error": self.has_error,
            "tests_pass": self.tests_pass,
            "iteration_bucket": self.iteration_bucket
        }

    def to_key(self) -> Tuple:
        """
        Convert state to hashable tuple key for Q-table lookup.

        Returns:
            Tuple key representing this state
        """
        return (
            int(self.has_plan),
            int(self.has_code),
            int(self.has_error),
            int(self.tests_pass),
            self.iteration_bucket
        )

    def to_index(self) -> int:
        """
        Convert state to a single integer index.

        Returns:
            Integer index (0-63)
        """
        # Binary encoding with iteration bucket
        idx = (
            int(self.has_plan) * 32 +
            int(self.has_code) * 16 +
            int(self.has_error) * 8 +
            int(self.tests_pass) * 4 +
            self.iteration_bucket
        )
        return idx

    @classmethod
    def from_index(cls, idx: int) -> 'State':
        """
        Create state from integer index.

        Args:
            idx: Integer index (0-63)

        Returns:
            State object
        """
        return cls(
            has_plan=bool(idx // 32 % 2),
            has_code=bool(idx // 16 % 2),
            has_error=bool(idx // 8 % 2),
            tests_pass=bool(idx // 4 % 2),
            iteration=idx % 4
        )

    @classmethod
    def initial(cls) -> 'State':
        """
        Create initial state for a new episode.

        Returns:
            Initial state with all flags False
        """
        return cls()

    def is_terminal(self) -> bool:
        """
        Check if this is a terminal state.

        Returns:
            True if tests pass (success) or iteration >= 5 (timeout)
        """
        return self.tests_pass or self.iteration >= 5

    def copy(self) -> 'State':
        """Create a copy of this state."""
        return State(
            has_plan=self.has_plan,
            has_code=self.has_code,
            has_error=self.has_error,
            tests_pass=self.tests_pass,
            iteration=self.iteration
        )

    def __repr__(self):
        features = []
        if self.has_plan:
            features.append("plan")
        if self.has_code:
            features.append("code")
        if self.has_error:
            features.append("error")
        if self.tests_pass:
            features.append("pass")
        return f"State({'+'.join(features) or 'empty'}, iter={self.iteration})"


# Action definitions
ACTIONS = ["planner", "coder", "tester", "debugger"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


def get_valid_actions(state: State) -> list:
    """
    Get valid actions for a given state.

    Some actions don't make sense in certain states:
    - Can't test without code
    - Can't debug without error

    Args:
        state: Current state

    Returns:
        List of valid action names
    """
    valid = ["planner", "coder"]  # Always available

    if state.has_code:
        valid.append("tester")

    if state.has_code and state.has_error:
        valid.append("debugger")

    return valid


if __name__ == "__main__":
    # Test state representation
    print("Testing State representation...")

    # Initial state
    s = State.initial()
    print(f"Initial: {s}")
    print(f"  Key: {s.to_key()}")
    print(f"  Index: {s.to_index()}")
    print(f"  Features: {s.to_features()}")

    # After planning
    s.has_plan = True
    print(f"\nAfter planning: {s}")
    print(f"  Key: {s.to_key()}")

    # After coding
    s.has_code = True
    print(f"\nAfter coding: {s}")
    print(f"  Key: {s.to_key()}")

    # After error
    s.has_error = True
    s.iteration = 1
    print(f"\nAfter error: {s}")
    print(f"  Key: {s.to_key()}")
    print(f"  Valid actions: {get_valid_actions(s)}")

    # Test from_index roundtrip
    print("\n\nTesting index roundtrip...")
    for i in range(64):
        s = State.from_index(i)
        assert s.to_index() == i, f"Failed for index {i}"
    print("All 64 states roundtrip correctly")

    print("\nState representation working!")
