"""
Coder Agent - Writes clean Python code based on plans.

Enhanced with RAG (Retrieval-Augmented Generation) for better code quality
by retrieving relevant context from the knowledge base.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, TYPE_CHECKING
from agents.base_agent import BaseAgent
from communication.blackboard import Blackboard, MessageType

if TYPE_CHECKING:
    from rag.retriever import Retriever

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """
    Coder agent that writes Python code based on plans and task descriptions.

    The Coder reads the plan from the blackboard (if available) and generates
    clean, working Python code. Optionally uses RAG to retrieve relevant
    context from the knowledge base for better code quality.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        retriever: Optional["Retriever"] = None,
        use_rag: bool = True
    ):
        """
        Initialize the Coder Agent.

        Args:
            api_key: OpenRouter API key
            model: Model to use for code generation
            retriever: RAG retriever instance for context augmentation
            use_rag: Whether to use RAG context (default: True)
        """
        super().__init__(name="coder", api_key=api_key, model=model)
        self.retriever = retriever
        self.use_rag = use_rag and retriever is not None
        self._last_rag_context = ""  # Store for debugging/display

    def _get_system_prompt(self) -> str:
        return """You are a coding agent. Output ONLY Python code in ```python blocks. No explanations."""

    def _get_rag_context(self, task: str) -> str:
        """
        Retrieve relevant context from knowledge base using RAG.

        Args:
            task: The task description to search for

        Returns:
            Formatted context string or empty string if RAG is disabled/unavailable
        """
        if not self.use_rag or self.retriever is None:
            return ""

        try:
            context = self.retriever.format_context(task, n_results=3)
            if context:
                logger.info(f"RAG retrieved context ({len(context)} chars) for task")
                self._last_rag_context = context
            return context
        except Exception as e:
            logger.warning(f"RAG retrieval failed, continuing without context: {e}")
            return ""

    def get_last_rag_context(self) -> str:
        """Get the RAG context used in the last code generation (for debugging/display)."""
        return self._last_rag_context

    def generate_code(self, task: str, blackboard: Blackboard) -> str:
        """
        Generate code for the given task, using plan from blackboard if available.

        Uses RAG to retrieve relevant context from the knowledge base when available.

        Args:
            task: The coding task description
            blackboard: Blackboard to read plan from and post code to

        Returns:
            The generated code
        """
        # Reset last RAG context
        self._last_rag_context = ""

        # Get RAG context if available
        rag_context = self._get_rag_context(task)

        # Check if there's a plan available
        plan_msg = blackboard.get_latest_by_type(MessageType.PLAN)

        # Build the prompt with all available context
        prompt_parts = [f"Task: {task}"]

        # Add RAG context if available
        if rag_context:
            prompt_parts.append("")
            prompt_parts.append(rag_context)

        # Add plan if available
        if plan_msg:
            plan_content = plan_msg.content[:800] if len(plan_msg.content) > 800 else plan_msg.content
            prompt_parts.append("")
            prompt_parts.append("Plan:")
            prompt_parts.append(plan_content)

        prompt_parts.append("")
        prompt_parts.append("Write Python code. Output ONLY code in ```python block.")

        prompt = "\n".join(prompt_parts)

        response = self.call_llm(prompt)
        code = self.extract_code(response)

        # Post to blackboard
        blackboard.post(
            sender=self.name,
            content=code,
            message_type=MessageType.CODE
        )

        return code


if __name__ == "__main__":
    bb = Blackboard()
    task = "Write a function that reverses a string"
    bb.set_task(task)

    print("Testing CoderAgent...")

    # Test without RAG first
    print("\n--- Testing without RAG ---")
    try:
        agent = CoderAgent(use_rag=False)
        code = agent.generate_code(task, bb)
        print(f"Generated code:\n{code}")
        print("\nCoder agent (no RAG) working!")
    except Exception as e:
        print(f"Error (API key may not be set): {e}")

    # Test with RAG if available
    print("\n--- Testing with RAG ---")
    try:
        from rag.retriever import Retriever
        retriever = Retriever()
        retriever.initialize()

        agent_with_rag = CoderAgent(retriever=retriever, use_rag=True)
        bb2 = Blackboard()
        bb2.set_task(task)
        code = agent_with_rag.generate_code(task, bb2)
        print(f"Generated code:\n{code}")

        rag_context = agent_with_rag.get_last_rag_context()
        if rag_context:
            print(f"\nRAG context used:\n{rag_context[:500]}...")
        print("\nCoder agent (with RAG) working!")
    except ImportError:
        print("RAG module not available, skipping RAG test")
    except Exception as e:
        print(f"RAG test error: {e}")
