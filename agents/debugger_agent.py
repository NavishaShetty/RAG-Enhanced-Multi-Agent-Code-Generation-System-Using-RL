"""
Debugger Agent - Fixes errors in code based on feedback.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from agents.base_agent import BaseAgent
from communication.blackboard import Blackboard, MessageType


class DebuggerAgent(BaseAgent):
    """
    Debugger agent that fixes code errors based on feedback.

    The Debugger reads error messages and feedback from the blackboard
    and produces corrected code.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(name="debugger", api_key=api_key, model=model)

    def _get_system_prompt(self) -> str:
        return """You are a debugging agent. Your job is to fix code errors.

            Rules:
            1. Read the error message or feedback carefully
            2. Identify the root cause of the issue
            3. Fix ONLY the issues mentioned - don't over-engineer
            4. Output ONLY the corrected Python code in a ```python block
            5. Do NOT include explanations outside the code block

            Example output format:
            ```python
            def fixed_function(params):
                \"\"\"Docstring.\"\"\"
                # corrected implementation
                return result
            ```"""

    def fix_code(self, code: str, error: str, blackboard: Blackboard) -> str:
        """
        Fix code based on error message or feedback.

        Args:
            code: The original code with issues
            error: Error message or feedback describing the issues
            blackboard: Blackboard to post fixed code to

        Returns:
            The corrected code
        """
        task = blackboard.get_task() or "Fix the code"

        prompt = f"""Original task: {task}

        Code with issues:
        ```python
        {code}
        ```

        Issues to fix:
        {error}

        Fix these issues and output ONLY the corrected code in a ```python block."""

        response = self.call_llm(prompt)
        fixed_code = self.extract_code(response)

        # Post to blackboard
        blackboard.post(
            sender=self.name,
            content=fixed_code,
            message_type=MessageType.CODE,
            metadata={"is_fix": True}
        )

        # Also post debug info
        blackboard.post(
            sender=self.name,
            content=f"Fixed issues: {error[:200]}...",
            message_type=MessageType.DEBUG
        )

        return fixed_code


if __name__ == "__main__":
    from communication.blackboard import Blackboard

    bb = Blackboard()
    task = "Write a function that finds the maximum in a list"
    bb.set_task(task)

    code = """def find_max(lst):
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val"""

    error = "FAIL: The function will raise IndexError if the list is empty. Need to handle empty list edge case."

    print("Testing DebuggerAgent...")
    try:
        agent = DebuggerAgent()
        fixed = agent.fix_code(code, error, bb)
        print(f"Fixed code:\n{fixed}")
        print("\nDebugger agent working!")
    except Exception as e:
        print(f"Error (API key may not be set): {e}")
