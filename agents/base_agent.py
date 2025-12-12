"""
Base Agent Class for LLM-powered agents.

All specialized agents (Planner, Coder, Tester, Debugger) inherit from this base class.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
from typing import Optional
from utils.api import OpenRouterClient
from communication.blackboard import Blackboard


class BaseAgent(ABC):
    """
    Abstract base class for all LLM-powered agents.

    Provides common functionality for making LLM calls and interacting
    with the blackboard communication system.
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique name for this agent (e.g., "planner", "coder")
            api_key: Optional OpenRouter API key (uses env var if not provided)
            model: Optional model override
        """
        self.name = name
        self._client = OpenRouterClient(api_key=api_key, model=model)

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent type.

        Returns:
            System prompt string defining the agent's role
        """
        pass

    def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Make a call to the LLM.

        Args:
            prompt: User prompt to send
            system_prompt: Override default system prompt
            temperature: Override default temperature

        Returns:
            LLM response text
        """
        sys_prompt = system_prompt if system_prompt else self._get_system_prompt()
        try:
            response = self._client.call(
                prompt=prompt,
                system_prompt=sys_prompt,
                temperature=temperature
            )
            return response
        except Exception as e:
            return f"Error calling LLM: {e}"

    def extract_code(self, response: str) -> str:
        """
        Extract Python code from an LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Extracted code (or original response if no code block found)
        """
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return response.strip()

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


if __name__ == "__main__":
    # Test basic functionality (this will fail since BaseAgent is abstract)
    print("BaseAgent is an abstract class - cannot be instantiated directly")
    print("Base agent module loaded successfully!")
