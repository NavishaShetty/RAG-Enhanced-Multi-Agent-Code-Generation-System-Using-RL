"""
Reward Function for the RL Environment.

Defines rewards for different state transitions and outcomes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.state import State, ACTIONS


# Reward constants
# Tuned based on validation experiments to better match real LLM behavior
REWARDS = {
    "task_success": 10.0,       # Tests pass - big reward
    "task_timeout": -5.0,       # Max iterations exceeded
    "progress_plan": 0.2,       # Made a plan
    "progress_code": 0.3,       # Wrote code
    "tests_pass_partial": 1.0,  # Some tests pass
    "error_fixed": 0.5,         # Debugger fixed an error
    "redundant_action": -0.2,   # Tuned from -0.5 - less harsh for redundant actions
    "invalid_action": -0.3,     # Called tester without code
    "step_cost": -0.1,          # Small cost per step (encourages efficiency)
}


class RewardFunction:
    """
    Calculates rewards based on state transitions.

    The reward function encourages:
    1. Task completion (big positive reward)
    2. Progress towards completion (small positive)
    3. Efficiency (small negative step cost)
    4. Avoiding redundant or invalid actions
    """

    def __init__(self, rewards: dict = None):
        """
        Initialize reward function.

        Args:
            rewards: Optional custom reward values
        """
        self.rewards = rewards or REWARDS.copy()

    def calculate(
        self,
        state: State,
        action: str,
        next_state: State,
        done: bool
    ) -> float:
        """
        Calculate reward for a state transition.

        Args:
            state: Current state before action
            action: Action taken
            next_state: Resulting state
            done: Whether episode is done

        Returns:
            Reward value
        """
        reward = self.rewards["step_cost"]  # Base step cost

        # Terminal rewards
        if done:
            if next_state.tests_pass:
                reward += self.rewards["task_success"]
            else:
                reward += self.rewards["task_timeout"]
            return reward

        # Progress rewards
        if action == "planner":
            if state.has_plan:
                # Redundant - already have a plan
                reward += self.rewards["redundant_action"]
            else:
                reward += self.rewards["progress_plan"]

        elif action == "coder":
            if not state.has_plan and not state.has_code:
                # Coding without plan is less effective but allowed
                pass
            if not state.has_code and next_state.has_code:
                reward += self.rewards["progress_code"]

        elif action == "tester":
            if not state.has_code:
                # Invalid - can't test without code
                reward += self.rewards["invalid_action"]
            elif next_state.tests_pass:
                # Not quite done yet but tests passing
                reward += self.rewards["tests_pass_partial"]

        elif action == "debugger":
            if not state.has_error:
                # Invalid - no error to debug
                reward += self.rewards["invalid_action"]
            elif state.has_error and not next_state.has_error:
                # Successfully fixed error
                reward += self.rewards["error_fixed"]

        return reward

    def get_reward_description(self) -> str:
        """
        Get human-readable description of reward structure.

        Returns:
            Formatted reward description
        """
        lines = ["Reward Structure:"]
        for name, value in self.rewards.items():
            sign = "+" if value > 0 else ""
            lines.append(f"  {name}: {sign}{value}")
        return "\n".join(lines)


if __name__ == "__main__":
    rf = RewardFunction()
    print(rf.get_reward_description())

    print("\n" + "="*50)
    print("Testing reward calculations:")

    # Test task success
    s1 = State(has_plan=True, has_code=True, has_error=False, iteration=2)
    s2 = State(has_plan=True, has_code=True, has_error=False, tests_pass=True, iteration=2)
    r = rf.calculate(s1, "tester", s2, done=True)
    print(f"\nTask success: {r} (expected ~{REWARDS['task_success'] + REWARDS['step_cost']})")

    # Test progress plan
    s1 = State.initial()
    s2 = State(has_plan=True)
    r = rf.calculate(s1, "planner", s2, done=False)
    print(f"Progress (plan): {r} (expected ~{REWARDS['progress_plan'] + REWARDS['step_cost']})")

    # Test redundant action
    s1 = State(has_plan=True)
    s2 = State(has_plan=True)
    r = rf.calculate(s1, "planner", s2, done=False)
    print(f"Redundant planner: {r} (expected ~{REWARDS['redundant_action'] + REWARDS['step_cost']})")

    # Test invalid action
    s1 = State.initial()
    s2 = State.initial()
    r = rf.calculate(s1, "tester", s2, done=False)
    print(f"Invalid tester: {r} (expected ~{REWARDS['invalid_action'] + REWARDS['step_cost']})")

    print("\nReward function working!")
