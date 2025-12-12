"""
Planner Agent - Breaks down coding tasks into clear steps.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from agents.base_agent import BaseAgent
from communication.blackboard import Blackboard, MessageType


class PlannerAgent(BaseAgent):
    """
    Planner agent that breaks down coding tasks into actionable steps.

    The Planner is typically the first agent invoked in the pipeline.
    It analyzes the task and creates a structured plan for the Coder.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(name="planner", api_key=api_key, model=model)

    def _get_system_prompt(self) -> str:
        return """You are a planning agent for code generation tasks. Your job is to:

            1. Analyze the coding task carefully
            2. Break it down into clear, actionable steps
            3. Identify edge cases that should be handled
            4. Suggest the function signature and parameters

            Output a structured plan with numbered steps. Be concise but thorough.
            Focus on the logic and approach, not the actual code."""

    def generate_plan(self, task: str, blackboard: Blackboard) -> str:
        """
        Generate a plan for the given coding task.

        Args:
            task: The coding task description
            blackboard: Blackboard for posting the plan

        Returns:
            The generated plan
        """
        prompt = f"""Task: {task}

        Create a detailed plan to implement this. Include:
        1. Function signature (name, parameters, return type)
        2. Step-by-step logic
        3. Edge cases to handle
        4. Any helper functions needed"""

        plan = self.call_llm(prompt)

        # Post to blackboard
        blackboard.post(
            sender=self.name,
            content=plan,
            message_type=MessageType.PLAN
        )

        return plan


if __name__ == "__main__":
    # Quick test
    task = "Write a function that finds the maximum element in a list"
    bb = Blackboard()
    bb.set_task(task)

    print("Testing PlannerAgent...")
    try:
        agent = PlannerAgent()
        plan = agent.generate_plan(task, bb)
        print(f"Generated plan:\n{plan}")
        print("\nPlanner agent working!")
    except Exception as e:
        print(f"Error (API key may not be set): {e}")
