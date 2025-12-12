"""
Real Environment Wrapper for LLM-based Code Generation.

Connects the RL agent to real LLM agents for validation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, Optional

from agents.planner_agent import PlannerAgent
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.debugger_agent import DebuggerAgent
from communication.blackboard import Blackboard, MessageType
from tools.test_runner import TestRunner
from environment.state import State, ACTIONS
from environment.rewards import RewardFunction


class CodingEnv:
    """
    Real environment that uses LLM agents.

    Same interface as SimulatedEnv but calls actual LLM APIs.
    Used for validation and real-world testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_iterations: int = 5
    ):
        """
        Initialize the real environment.

        Args:
            api_key: OpenRouter API key
            max_iterations: Maximum steps per episode
        """
        # Create LLM agents
        self.planner = PlannerAgent(api_key=api_key)
        self.coder = CoderAgent(api_key=api_key)
        self.tester_agent = TesterAgent(api_key=api_key)
        self.debugger = DebuggerAgent(api_key=api_key)

        # Tools
        self.test_runner = TestRunner()
        self.reward_fn = RewardFunction()

        # Environment state
        self.max_iterations = max_iterations
        self.blackboard: Optional[Blackboard] = None
        self.state: Optional[State] = None
        self.task: Optional[str] = None
        self.current_code: Optional[str] = None

    def reset(self, task: str) -> State:
        """
        Reset environment for a new task.

        Args:
            task: The coding task description

        Returns:
            Initial state
        """
        self.task = task
        self.blackboard = Blackboard()
        self.blackboard.set_task(task)
        self.state = State.initial()
        self.current_code = None

        return self.state.copy()

    def step(self, action: str) -> Tuple[State, float, bool]:
        """
        Take an action by invoking the corresponding LLM agent.

        Args:
            action: One of ["planner", "coder", "tester", "debugger"]

        Returns:
            Tuple of (next_state, reward, done)
        """
        if self.state is None or self.blackboard is None:
            raise ValueError("Call reset() before step()")

        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        old_state = self.state.copy()

        # Execute action
        if action == "planner":
            self._do_planner()
        elif action == "coder":
            self._do_coder()
        elif action == "tester":
            self._do_tester()
        elif action == "debugger":
            self._do_debugger()

        # Update iteration
        self.state.iteration += 1

        # Check if done
        done = self.state.is_terminal() or self.state.iteration >= self.max_iterations

        # Calculate reward
        reward = self.reward_fn.calculate(old_state, action, self.state, done)

        return self.state.copy(), reward, done

    def _do_planner(self):
        """Execute planner agent."""
        try:
            plan = self.planner.generate_plan(self.task, self.blackboard)
            if plan and len(plan) > 10:
                self.state.has_plan = True
        except Exception as e:
            print(f"Planner error: {e}")

    def _do_coder(self):
        """Execute coder agent."""
        try:
            code = self.coder.generate_code(self.task, self.blackboard)
            if code and len(code) > 10:
                self.current_code = code
                self.state.has_code = True
                # Assume new code might have errors until tested
                self.state.has_error = False
                self.state.tests_pass = False
        except Exception as e:
            print(f"Coder error: {e}")

    def _do_tester(self):
        """Execute tester agent."""
        if not self.state.has_code or not self.current_code:
            return

        try:
            # Run actual tests
            test_result = self.test_runner.run_tests(self.current_code, self.task)

            # Get LLM analysis
            analysis = self.tester_agent.analyze_code(
                self.current_code, self.blackboard, self.task
            )

            # Update state based on results
            llm_passed = self.tester_agent.check_passed(analysis)

            if test_result.all_passed and llm_passed:
                self.state.tests_pass = True
                self.state.has_error = False
            else:
                self.state.tests_pass = False
                self.state.has_error = True

                # Post error info to blackboard
                if test_result.error_messages:
                    error_msg = "\n".join(test_result.error_messages[:3])
                    self.blackboard.post(
                        sender="tester",
                        content=error_msg,
                        message_type=MessageType.ERROR
                    )

        except Exception as e:
            print(f"Tester error: {e}")
            self.state.has_error = True

    def _do_debugger(self):
        """Execute debugger agent."""
        if not self.state.has_code or not self.state.has_error:
            return

        try:
            # Get error info from blackboard
            error_msg = self.blackboard.get_latest_by_type(MessageType.ERROR)
            feedback_msg = self.blackboard.get_latest_by_type(MessageType.FEEDBACK)

            error_info = ""
            if error_msg:
                error_info += error_msg.content
            if feedback_msg:
                error_info += "\n" + feedback_msg.content

            if not error_info:
                error_info = "Fix the bugs in the code"

            # Fix code
            fixed_code = self.debugger.fix_code(
                self.current_code, error_info, self.blackboard
            )

            if fixed_code and len(fixed_code) > 10:
                self.current_code = fixed_code
                self.state.has_error = False  # Assume fixed until tested
                self.state.tests_pass = False  # Need to re-test

        except Exception as e:
            print(f"Debugger error: {e}")

    def get_valid_actions(self) -> list:
        """Get valid actions for current state."""
        if self.state is None:
            return []

        valid = ["planner", "coder"]

        if self.state.has_code:
            valid.append("tester")

        if self.state.has_code and self.state.has_error:
            valid.append("debugger")

        return valid

    def get_current_code(self) -> Optional[str]:
        """Get the current code."""
        return self.current_code


if __name__ == "__main__":
    print("Testing CodingEnv (requires API key)...")

    try:
        env = CodingEnv(max_iterations=3)

        task = "Write a function that checks if a number is even"
        print(f"\nTask: {task}")

        state = env.reset(task)
        print(f"Initial state: {state}")

        # Manual policy: plan -> code -> test
        actions = ["planner", "coder", "tester"]
        for action in actions:
            if action in env.get_valid_actions():
                print(f"\nAction: {action}")
                state, reward, done = env.step(action)
                print(f"State: {state}")
                print(f"Reward: {reward:.2f}")
                print(f"Done: {done}")

                if done:
                    break

        if env.current_code:
            print(f"\nFinal code:\n{env.current_code}")

        print("\nReal environment working!")

    except Exception as e:
        print(f"Error (API key may not be set): {e}")
