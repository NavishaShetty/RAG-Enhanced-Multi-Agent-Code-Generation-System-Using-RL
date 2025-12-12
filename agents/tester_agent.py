"""
Tester Agent - Analyzes code and identifies potential issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from agents.base_agent import BaseAgent
from communication.blackboard import Blackboard, MessageType


class TesterAgent(BaseAgent):
    """
    Tester agent that analyzes code for potential issues.

    The Tester examines code from the blackboard and provides feedback
    about bugs, edge cases, and improvements needed.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(name="tester", api_key=api_key, model=model)

    def _get_system_prompt(self) -> str:
        return """You are a testing and code review agent. Analyze Python code for:

            1. Logical errors and bugs
            2. Missing edge case handling
            3. Potential runtime errors
            4. Incorrect implementation of requirements

            Provide specific, actionable feedback. If the code looks correct, say "PASS" and briefly explain why.
            If there are issues, list them clearly with "FAIL:" prefix."""

    def analyze_code(self, code: str, blackboard: Blackboard, task: Optional[str] = None) -> str:
        """
        Analyze code for issues and provide feedback.

        Args:
            code: The code to analyze
            blackboard: Blackboard to post feedback to
            task: Optional task description for context

        Returns:
            Analysis feedback
        """
        task_desc = task or blackboard.get_task() or "Unknown task"

        prompt = f"""Task: {task_desc}

        Code to analyze:
        ```python
        {code}
        ```

        Analyze this code:
        1. Does it correctly implement the task?
        2. Are there any bugs or logical errors?
        3. Are edge cases handled properly?
        4. Will it raise any runtime errors?

        If all good: Start with "PASS:" and brief explanation.
        If issues found: Start with "FAIL:" and list specific issues."""

        feedback = self.call_llm(prompt)

        # Post to blackboard
        blackboard.post(
            sender=self.name,
            content=feedback,
            message_type=MessageType.FEEDBACK
        )

        return feedback

    def check_passed(self, feedback: str) -> bool:
        """
        Check if the analysis indicates the code passed.

        Args:
            feedback: The analysis feedback

        Returns:
            True if code passed analysis, False otherwise
        """
        feedback_lower = feedback.lower()
        return feedback_lower.startswith("pass") or "pass:" in feedback_lower[:50]


if __name__ == "__main__":

    bb = Blackboard()
    task = "Write a function that finds the maximum in a list"
    bb.set_task(task)

    code = """def find_max(lst):
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val"""

    print("Testing TesterAgent...")
    try:
        agent = TesterAgent()
        feedback = agent.analyze_code(code, bb, task)
        print(f"Feedback:\n{feedback}")
        print(f"\nPassed: {agent.check_passed(feedback)}")
        print("\nTester agent working!")
    except Exception as e:
        print(f"Error (API key may not be set): {e}")
