"""
Unit tests for the Blackboard communication system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta
from communication.blackboard import Blackboard, Message, MessageType


class TestMessageType(unittest.TestCase):
    """Test MessageType enum."""

    def test_message_types_exist(self):
        """Test that all expected message types are defined."""
        expected_types = ["TASK", "PLAN", "CODE", "TEST_RESULT", "ERROR", "FEEDBACK", "DEBUG"]
        for msg_type in expected_types:
            self.assertTrue(hasattr(MessageType, msg_type))

    def test_message_type_values(self):
        """Test message type values are correct strings."""
        self.assertEqual(MessageType.TASK.value, "task")
        self.assertEqual(MessageType.PLAN.value, "plan")
        self.assertEqual(MessageType.CODE.value, "code")
        self.assertEqual(MessageType.ERROR.value, "error")


class TestMessage(unittest.TestCase):
    """Test Message dataclass."""

    def test_message_creation(self):
        """Test creating a message with required fields."""
        msg = Message(
            sender="planner",
            receiver="coder",
            content="Test content",
            message_type=MessageType.PLAN
        )
        self.assertEqual(msg.sender, "planner")
        self.assertEqual(msg.receiver, "coder")
        self.assertEqual(msg.content, "Test content")
        self.assertEqual(msg.message_type, MessageType.PLAN)

    def test_message_default_timestamp(self):
        """Test that messages get a timestamp by default."""
        before = datetime.now()
        msg = Message(
            sender="test",
            receiver=None,
            content="test",
            message_type=MessageType.TASK
        )
        after = datetime.now()
        self.assertIsInstance(msg.timestamp, datetime)
        self.assertTrue(before <= msg.timestamp <= after)

    def test_message_default_metadata(self):
        """Test that messages get empty metadata by default."""
        msg = Message(
            sender="test",
            receiver=None,
            content="test",
            message_type=MessageType.TASK
        )
        self.assertEqual(msg.metadata, {})

    def test_message_with_metadata(self):
        """Test creating a message with custom metadata."""
        metadata = {"key": "value", "count": 42}
        msg = Message(
            sender="test",
            receiver=None,
            content="test",
            message_type=MessageType.CODE,
            metadata=metadata
        )
        self.assertEqual(msg.metadata, metadata)

    def test_message_repr(self):
        """Test message string representation."""
        msg = Message(
            sender="coder",
            receiver="tester",
            content="def foo(): pass",
            message_type=MessageType.CODE
        )
        repr_str = repr(msg)
        self.assertIn("coder", repr_str)
        self.assertIn("code", repr_str)


class TestBlackboard(unittest.TestCase):
    """Test Blackboard class."""

    def setUp(self):
        """Create a fresh blackboard for each test."""
        self.bb = Blackboard()

    def test_initial_state(self):
        """Test blackboard starts empty."""
        self.assertEqual(len(self.bb), 0)
        self.assertIsNone(self.bb.get_task())

    def test_post_message(self):
        """Test posting a message."""
        msg = self.bb.post(
            sender="planner",
            content="Step 1: Define function",
            message_type=MessageType.PLAN
        )
        self.assertEqual(len(self.bb), 1)
        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.sender, "planner")

    def test_post_multiple_messages(self):
        """Test posting multiple messages."""
        self.bb.post("planner", "plan content", MessageType.PLAN)
        self.bb.post("coder", "code content", MessageType.CODE)
        self.bb.post("tester", "test results", MessageType.TEST_RESULT)
        self.assertEqual(len(self.bb), 3)

    def test_set_and_get_task(self):
        """Test setting and getting a task."""
        task = "Write a function to add two numbers"
        self.bb.set_task(task)
        self.assertEqual(self.bb.get_task(), task)
        self.assertEqual(len(self.bb), 1)

    def test_get_messages_for_broadcast(self):
        """Test getting broadcast messages (receiver=None)."""
        self.bb.post("planner", "plan", MessageType.PLAN, receiver=None)
        self.bb.post("coder", "code", MessageType.CODE, receiver=None)

        messages = self.bb.get_messages_for("anyone")
        self.assertEqual(len(messages), 2)

    def test_get_messages_for_specific_receiver(self):
        """Test getting messages for a specific receiver."""
        self.bb.post("planner", "plan", MessageType.PLAN, receiver="coder")
        self.bb.post("coder", "code", MessageType.CODE, receiver="tester")
        self.bb.post("broadcast", "info", MessageType.FEEDBACK, receiver=None)

        coder_messages = self.bb.get_messages_for("coder")
        self.assertEqual(len(coder_messages), 2)  # direct + broadcast

        tester_messages = self.bb.get_messages_for("tester")
        self.assertEqual(len(tester_messages), 2)  # direct + broadcast

    def test_get_messages_with_type_filter(self):
        """Test filtering messages by type."""
        self.bb.post("planner", "plan", MessageType.PLAN)
        self.bb.post("coder", "code", MessageType.CODE)
        self.bb.post("coder", "more code", MessageType.CODE)

        code_messages = self.bb.get_messages_for("anyone", message_type=MessageType.CODE)
        self.assertEqual(len(code_messages), 2)

        plan_messages = self.bb.get_messages_for("anyone", message_type=MessageType.PLAN)
        self.assertEqual(len(plan_messages), 1)

    def test_get_messages_with_limit(self):
        """Test limiting number of returned messages."""
        for i in range(5):
            self.bb.post("sender", f"message {i}", MessageType.FEEDBACK)

        messages = self.bb.get_messages_for("anyone", limit=2)
        self.assertEqual(len(messages), 2)
        # Should return the last 2 messages
        self.assertIn("message 3", messages[0].content)
        self.assertIn("message 4", messages[1].content)

    def test_get_latest_by_type(self):
        """Test getting the latest message of a type."""
        self.bb.post("coder", "code v1", MessageType.CODE)
        self.bb.post("coder", "code v2", MessageType.CODE)
        self.bb.post("coder", "code v3", MessageType.CODE)

        latest = self.bb.get_latest_by_type(MessageType.CODE)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.content, "code v3")

    def test_get_latest_by_type_not_found(self):
        """Test getting latest when type doesn't exist."""
        self.bb.post("planner", "plan", MessageType.PLAN)
        latest = self.bb.get_latest_by_type(MessageType.ERROR)
        self.assertIsNone(latest)

    def test_get_all_by_type(self):
        """Test getting all messages of a type."""
        self.bb.post("coder", "code 1", MessageType.CODE)
        self.bb.post("planner", "plan", MessageType.PLAN)
        self.bb.post("coder", "code 2", MessageType.CODE)

        all_code = self.bb.get_all_by_type(MessageType.CODE)
        self.assertEqual(len(all_code), 2)
        self.assertEqual(all_code[0].content, "code 1")
        self.assertEqual(all_code[1].content, "code 2")

    def test_get_latest_from(self):
        """Test getting latest message from a sender."""
        self.bb.post("coder", "code 1", MessageType.CODE)
        self.bb.post("planner", "plan", MessageType.PLAN)
        self.bb.post("coder", "code 2", MessageType.CODE)

        latest_from_coder = self.bb.get_latest_from("coder")
        self.assertIsNotNone(latest_from_coder)
        self.assertEqual(latest_from_coder.content, "code 2")

    def test_get_latest_from_not_found(self):
        """Test getting latest from non-existent sender."""
        self.bb.post("coder", "code", MessageType.CODE)
        latest = self.bb.get_latest_from("debugger")
        self.assertIsNone(latest)

    def test_clear(self):
        """Test clearing the blackboard."""
        self.bb.set_task("test task")
        self.bb.post("coder", "code", MessageType.CODE)
        self.bb.post("planner", "plan", MessageType.PLAN)

        self.assertEqual(len(self.bb), 3)
        self.assertIsNotNone(self.bb.get_task())

        self.bb.clear()

        self.assertEqual(len(self.bb), 0)
        self.assertIsNone(self.bb.get_task())

    def test_get_history(self):
        """Test getting message history."""
        self.bb.post("a", "msg 1", MessageType.TASK)
        self.bb.post("b", "msg 2", MessageType.PLAN)
        self.bb.post("c", "msg 3", MessageType.CODE)

        history = self.bb.get_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].sender, "a")
        self.assertEqual(history[2].sender, "c")

    def test_get_state_summary(self):
        """Test getting state summary."""
        summary = self.bb.get_state_summary()

        self.assertFalse(summary["has_task"])
        self.assertFalse(summary["has_plan"])
        self.assertFalse(summary["has_code"])
        self.assertFalse(summary["has_error"])
        self.assertEqual(summary["total_messages"], 0)

        # Add some messages
        self.bb.set_task("test task")
        self.bb.post("planner", "plan", MessageType.PLAN)
        self.bb.post("coder", "def foo(): pass", MessageType.CODE)

        summary = self.bb.get_state_summary()
        self.assertTrue(summary["has_task"])
        self.assertTrue(summary["has_plan"])
        self.assertTrue(summary["has_code"])
        self.assertFalse(summary["has_error"])
        self.assertEqual(summary["total_messages"], 3)
        self.assertEqual(summary["latest_code"], "def foo(): pass")

    def test_repr(self):
        """Test blackboard string representation."""
        repr_str = repr(self.bb)
        self.assertIn("Blackboard", repr_str)
        self.assertIn("messages=0", repr_str)


class TestBlackboardIntegration(unittest.TestCase):
    """Integration tests for Blackboard communication patterns."""

    def test_typical_workflow(self):
        """Test a typical agent workflow pattern."""
        bb = Blackboard()

        # Step 1: User sets task
        bb.set_task("Write a function to reverse a string")

        # Step 2: Planner creates plan
        bb.post(
            sender="planner",
            content="1. Define function\n2. Use slicing\n3. Return result",
            message_type=MessageType.PLAN
        )

        # Step 3: Coder reads plan and creates code
        plan = bb.get_latest_by_type(MessageType.PLAN)
        self.assertIsNotNone(plan)

        bb.post(
            sender="coder",
            content="def reverse(s):\n    return s[::-1]",
            message_type=MessageType.CODE
        )

        # Step 4: Tester finds no errors
        bb.post(
            sender="tester",
            content="All tests passed",
            message_type=MessageType.TEST_RESULT
        )

        # Verify final state
        summary = bb.get_state_summary()
        self.assertTrue(summary["has_plan"])
        self.assertTrue(summary["has_code"])
        self.assertFalse(summary["has_error"])
        self.assertEqual(summary["total_messages"], 4)

    def test_debug_cycle(self):
        """Test a debugging cycle with error and fix."""
        bb = Blackboard()

        # Initial code with error
        bb.post("coder", "def add(a, b): a + b", MessageType.CODE)

        # Tester finds error
        bb.post("tester", "Missing return statement", MessageType.ERROR)

        # Verify error state
        summary = bb.get_state_summary()
        self.assertTrue(summary["has_error"])

        # Debugger fixes code
        bb.post("debugger", "def add(a, b): return a + b", MessageType.CODE)

        # Get latest code (should be the fixed version)
        latest_code = bb.get_latest_by_type(MessageType.CODE)
        self.assertIn("return", latest_code.content)


if __name__ == "__main__":
    unittest.main()
