"""
Blackboard Communication System for Multi-Agent Coordination.

The Blackboard pattern provides a shared memory space where agents can post
and retrieve messages. This enables loose coupling between agents while
maintaining structured communication.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class MessageType(Enum):
    """Types of messages that can be posted to the blackboard."""
    TASK = "task"                   # Initial coding task
    PLAN = "plan"                   # Plan from Planner agent
    CODE = "code"                   # Code from Coder agent
    TEST_RESULT = "test_result"     # Results from test execution
    ERROR = "error"                 # Error messages
    FEEDBACK = "feedback"           # Feedback/analysis from Tester agent
    DEBUG = "debug"                 # Debug info from Debugger agent


@dataclass
class Message:
    """A message posted to the blackboard."""
    sender: str                 # Agent that sent the message (e.g., "planner", "coder")
    receiver: Optional[str]     # Target agent (None = broadcast to all)
    content: str                # The actual message content
    message_type: MessageType   # Type of message
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)  # Additional data

    def __repr__(self):
        return (f"Message(sender='{self.sender}', type={self.message_type.value}, "
                f"receiver={self.receiver}, len={len(self.content)})")


class Blackboard:
    """
    Shared memory space for agent communication.

    Provides methods to post messages and retrieve messages based on
    various filters (sender, receiver, type).
    """

    def __init__(self):
        """Initialize an empty blackboard."""
        self._messages: List[Message] = []
        self._current_task: Optional[str] = None

    def post(
        self,
        sender: str,
        content: str,
        message_type: MessageType,
        receiver: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Message:
        """
        Post a message to the blackboard.

        Args:
            sender: Name of the sending agent
            content: Message content
            message_type: Type of message
            receiver: Target agent (None for broadcast)
            metadata: Optional additional data

        Returns:
            The posted Message object
        """
        message = Message(
            sender=sender,
            receiver=receiver,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self._messages.append(message)
        return message

    def get_messages_for(
        self,
        receiver: str,
        message_type: Optional[MessageType] = None,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages for a specific receiver.

        Args:
            receiver: Name of the receiving agent
            message_type: Optional filter by message type
            limit: Maximum number of messages to return (newest first)

        Returns:
            List of matching messages
        """
        messages = [
            m for m in self._messages
            if m.receiver is None or m.receiver == receiver
        ]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_latest_by_type(self, message_type: MessageType) -> Optional[Message]:
        """
        Get the most recent message of a specific type.

        Args:
            message_type: Type of message to find

        Returns:
            Most recent message of that type, or None if not found
        """
        for message in reversed(self._messages):
            if message.message_type == message_type:
                return message
        return None

    def get_all_by_type(self, message_type: MessageType) -> List[Message]:
        """
        Get all messages of a specific type.

        Args:
            message_type: Type of message to filter by

        Returns:
            List of matching messages in chronological order
        """
        return [m for m in self._messages if m.message_type == message_type]

    def get_latest_from(self, sender: str) -> Optional[Message]:
        """
        Get the most recent message from a specific sender.

        Args:
            sender: Name of the sending agent

        Returns:
            Most recent message from that sender, or None
        """
        for message in reversed(self._messages):
            if message.sender == sender:
                return message
        return None

    def set_task(self, task: str) -> Message:
        """
        Set the current coding task (convenience method).

        Args:
            task: The coding task description

        Returns:
            The posted task message
        """
        self._current_task = task
        return self.post(
            sender="user",
            content=task,
            message_type=MessageType.TASK
        )

    def get_task(self) -> Optional[str]:
        """Get the current coding task."""
        return self._current_task

    def clear(self):
        """Clear all messages from the blackboard."""
        self._messages = []
        self._current_task = None

    def get_history(self) -> List[Message]:
        """Get all messages in chronological order."""
        return list(self._messages)

    def get_state_summary(self) -> dict:
        """
        Get a summary of the blackboard state.

        Returns:
            Dictionary with current state information
        """
        latest_plan = self.get_latest_by_type(MessageType.PLAN)
        latest_code = self.get_latest_by_type(MessageType.CODE)
        latest_error = self.get_latest_by_type(MessageType.ERROR)
        latest_feedback = self.get_latest_by_type(MessageType.FEEDBACK)

        return {
            "has_task": self._current_task is not None,
            "has_plan": latest_plan is not None,
            "has_code": latest_code is not None,
            "has_error": latest_error is not None,
            "has_feedback": latest_feedback is not None,
            "total_messages": len(self._messages),
            "latest_code": latest_code.content if latest_code else None,
            "latest_error": latest_error.content if latest_error else None
        }

    def __len__(self):
        return len(self._messages)

    def __repr__(self):
        return f"Blackboard(messages={len(self._messages)}, task={self._current_task is not None})"


if __name__ == "__main__":
    # Quick test
    bb = Blackboard()

    # Post a task
    bb.set_task("Write a function that adds two numbers")
    print(f"Task set: {bb.get_task()}")

    # Post a plan
    bb.post(
        sender="planner",
        content="1. Define function\n2. Add parameters\n3. Return sum",
        message_type=MessageType.PLAN
    )

    # Post code
    bb.post(
        sender="coder",
        content="def add(a, b):\n    return a + b",
        message_type=MessageType.CODE
    )

    # Get messages
    print(f"\nLatest plan: {bb.get_latest_by_type(MessageType.PLAN)}")
    print(f"Latest code: {bb.get_latest_by_type(MessageType.CODE)}")

    # Get state summary
    print(f"\nState: {bb.get_state_summary()}")
    print("Blackboard communication system working!")
