"""
UI Pipeline - connects Streamlit UI to the multi-agent system.

This module provides a unified interface for the UI to interact with
the multi-agent code generation system, including RAG retrieval and
RL orchestration.
"""

import sys
import os
import time
import logging
from typing import Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.blackboard import Blackboard, MessageType
from agents.planner_agent import PlannerAgent
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.debugger_agent import DebuggerAgent
from tools.test_runner import TestRunner
from tools.complexity_analyzer import ComplexityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool
    code: str
    steps: int
    time_seconds: float
    agent_logs: list = field(default_factory=list)
    rag_context: str = ""
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    test_results: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""


class UIPipeline:
    """Pipeline connecting UI to multi-agent system."""

    def __init__(
        self,
        use_rag: bool = True,
        use_rl: bool = True,
        max_iterations: int = 5,
    ):
        """
        Initialize the UI pipeline.

        Args:
            use_rag: Whether to use RAG for context retrieval
            use_rl: Whether to use RL policy for orchestration
            max_iterations: Maximum number of agent iterations
        """
        self.use_rag = use_rag
        self.use_rl = use_rl
        self.max_iterations = max_iterations

        # Initialize components (lazy loading)
        self._blackboard: Optional[Blackboard] = None
        self._retriever = None
        self._agents: Dict[str, Any] = {}
        self._rl_agent = None
        self._test_runner = TestRunner()
        self._complexity_analyzer = ComplexityAnalyzer()

        # State
        self._initialized = False
        self._agent_logs = []
        self._current_agent = None

    def _initialize_rag(self):
        """Initialize RAG retriever if enabled."""
        if not self.use_rag:
            return None

        try:
            from rag.retriever import Retriever
            retriever = Retriever()
            retriever.initialize()
            if retriever.is_initialized():
                logger.info("RAG retriever initialized")
                return retriever
            else:
                logger.warning("RAG retriever failed to initialize")
                return None
        except ImportError:
            logger.warning("RAG module not available")
            return None
        except Exception as e:
            logger.error(f"Error initializing RAG: {e}")
            return None

    def _initialize_agents(self):
        """Initialize all agents."""
        self._blackboard = Blackboard()

        # Initialize RAG first
        if self.use_rag and self._retriever is None:
            self._retriever = self._initialize_rag()

        # Initialize agents
        self._agents = {
            'planner': PlannerAgent(),
            'coder': CoderAgent(
                retriever=self._retriever,
                use_rag=self.use_rag and self._retriever is not None
            ),
            'tester': TesterAgent(),
            'debugger': DebuggerAgent(),
        }

        logger.info("Agents initialized")

    def _initialize_rl(self):
        """Initialize RL agent if enabled."""
        if not self.use_rl:
            return None

        try:
            from rl.combined_agent import CombinedAgent
            agent = CombinedAgent()

            # Try to load trained Q-table
            q_table_path = "experiments/results/q_table.json"
            if os.path.exists(q_table_path):
                agent.load(q_table_path)
                logger.info(f"Loaded RL policy from {q_table_path}")
            else:
                logger.warning("No trained RL policy found, using default")

            return agent
        except ImportError:
            logger.warning("RL module not available")
            return None
        except Exception as e:
            logger.error(f"Error initializing RL: {e}")
            return None

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if initialization successful
        """
        try:
            self._initialize_agents()

            if self.use_rl:
                self._rl_agent = self._initialize_rl()

            self._initialized = True
            logger.info("Pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    def _log(self, agent: str, message: str, details: str = ""):
        """Add entry to agent log."""
        log_entry = {
            "agent": agent,
            "message": message,
            "details": details,
            "timestamp": time.time()
        }
        self._agent_logs.append(log_entry)

    def _get_state(self) -> Dict[str, Any]:
        """Get current state from blackboard."""
        if self._blackboard is None:
            return {}

        summary = self._blackboard.get_state_summary()
        return {
            "has_plan": summary.get("has_plan", False),
            "has_code": summary.get("has_code", False),
            "has_error": summary.get("has_error", False),
            "tests_pass": False,  # Will be set by test runner
        }

    def _select_action(self, state: Dict[str, Any], iteration: int) -> str:
        """
        Select next agent action.

        Uses RL policy if available, otherwise uses fixed pipeline.
        """
        if self.use_rl and self._rl_agent is not None:
            try:
                from environment.state import State

                # Convert dict to State object
                rl_state = State(
                    has_plan=state.get("has_plan", False),
                    has_code=state.get("has_code", False),
                    has_error=state.get("has_error", False),
                    tests_pass=state.get("tests_pass", False),
                    iteration=iteration
                )

                # Get valid actions
                valid_actions = ['planner', 'coder']
                if state.get("has_code"):
                    valid_actions.append('tester')
                    if state.get("has_error"):
                        valid_actions.append('debugger')

                # Use RL agent to select action (no exploration in inference)
                action = self._rl_agent.choose_action(
                    rl_state,
                    valid_actions,
                    explore=False
                )
                return action
            except Exception as e:
                logger.warning(f"RL action selection failed: {e}, using fixed policy")

        # Fixed pipeline fallback
        if not state.get("has_plan"):
            return "planner"
        elif not state.get("has_code"):
            return "coder"
        elif state.get("has_error"):
            return "debugger"
        else:
            return "tester"

    def run(
        self,
        task: str,
        on_agent_start: Optional[Callable[[str], None]] = None,
        on_agent_complete: Optional[Callable[[str, str], None]] = None,
        on_log: Optional[Callable[[Dict], None]] = None,
    ) -> PipelineResult:
        """
        Run the generation pipeline for a given task.

        Args:
            task: The code generation task description
            on_agent_start: Callback when agent starts (agent_name)
            on_agent_complete: Callback when agent completes (agent_name, result)
            on_log: Callback for log entries

        Returns:
            PipelineResult with generated code and metrics
        """
        start_time = time.time()
        self._agent_logs = []

        # Initialize if needed
        if not self._initialized:
            self.initialize()

        # Reset blackboard
        self._blackboard = Blackboard()
        self._blackboard.set_task(task)

        # Re-initialize coder with current retriever
        self._agents['coder'] = CoderAgent(
            retriever=self._retriever,
            use_rag=self.use_rag and self._retriever is not None
        )

        self._log("System", f"Received task: {task[:100]}...")
        if on_log:
            on_log(self._agent_logs[-1])

        # Get RAG context if enabled
        rag_context = ""
        if self.use_rag and self._retriever is not None:
            try:
                rag_context = self._retriever.format_context(task)
                if rag_context:
                    self._log("RAG", "Retrieved relevant context", rag_context[:500])
                    if on_log:
                        on_log(self._agent_logs[-1])
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Run pipeline
        state = self._get_state()
        generated_code = ""
        test_results = {}
        iteration = 0
        success = False

        while iteration < self.max_iterations:
            iteration += 1

            # Select action
            action = self._select_action(state, iteration)
            self._current_agent = action.capitalize()

            self._log(self._current_agent, f"Starting ({iteration}/{self.max_iterations})")
            if on_agent_start:
                on_agent_start(self._current_agent)
            if on_log:
                on_log(self._agent_logs[-1])

            try:
                # Execute action
                if action == "planner":
                    result = self._agents['planner'].generate_plan(task, self._blackboard)
                    self._log("Planner", "Generated plan", result[:300] if result else "")

                elif action == "coder":
                    result = self._agents['coder'].generate_code(task, self._blackboard)
                    generated_code = result
                    self._log("Coder", f"Generated code ({len(result)} chars)", result[:500] if result else "")

                elif action == "tester":
                    # Run actual tests
                    code_msg = self._blackboard.get_latest_by_type(MessageType.CODE)
                    if code_msg:
                        test_result = self._test_runner.run_tests(code_msg.content, task)
                        test_results = {
                            "passed": test_result.tests_passed,
                            "failed": test_result.tests_failed,
                            "total": test_result.total_tests,
                            "success_rate": test_result.success_rate,
                            "all_passed": test_result.all_passed,
                            "errors": test_result.error_messages
                        }

                        if test_result.all_passed:
                            success = True
                            self._log("Tester", f"All tests passed! ({test_result.tests_passed}/{test_result.total_tests})")
                            if on_agent_complete:
                                on_agent_complete("Tester", "PASS")
                            if on_log:
                                on_log(self._agent_logs[-1])
                            break
                        else:
                            state["has_error"] = True
                            error_msg = "; ".join(test_result.error_messages[:3])
                            self._log("Tester", f"Tests failed ({test_result.tests_passed}/{test_result.total_tests})", error_msg)
                            # Post error to blackboard for debugger
                            self._blackboard.post(
                                sender="tester",
                                content=error_msg,
                                message_type=MessageType.ERROR
                            )

                elif action == "debugger":
                    code_msg = self._blackboard.get_latest_by_type(MessageType.CODE)
                    error_msg = self._blackboard.get_latest_by_type(MessageType.ERROR)
                    if code_msg and error_msg:
                        result = self._agents['debugger'].fix_code(
                            code_msg.content,
                            error_msg.content,
                            self._blackboard
                        )
                        generated_code = result
                        state["has_error"] = False
                        self._log("Debugger", "Fixed code", result[:300] if result else "")

                # Update state
                state = self._get_state()
                state["tests_pass"] = success

                if on_agent_complete:
                    on_agent_complete(self._current_agent, "Complete")
                if on_log:
                    on_log(self._agent_logs[-1])

            except Exception as e:
                error_msg = f"Error in {action}: {str(e)}"
                logger.error(error_msg)
                self._log(self._current_agent, f"Error: {str(e)}")
                if on_log:
                    on_log(self._agent_logs[-1])

        # Calculate complexity metrics
        complexity_metrics = {}
        if generated_code:
            try:
                analysis = self._complexity_analyzer.analyze(generated_code)
                complexity_metrics = {
                    "cyclomatic": analysis.get("cyclomatic_complexity", 0),
                    "loc": analysis.get("lines_of_code", 0),
                    "cognitive": analysis.get("cognitive_complexity", 0),
                    "rating": analysis.get("complexity_rating", "Unknown"),
                }
            except Exception as e:
                logger.warning(f"Complexity analysis failed: {e}")

        elapsed_time = time.time() - start_time

        return PipelineResult(
            success=success,
            code=generated_code,
            steps=iteration,
            time_seconds=elapsed_time,
            agent_logs=self._agent_logs,
            rag_context=rag_context,
            complexity_metrics=complexity_metrics,
            test_results=test_results,
            error_message="" if success else "Max iterations reached without passing tests"
        )

    def run_simple(self, task: str) -> PipelineResult:
        """
        Run a simple fixed pipeline without RL.

        This is useful for demo purposes or when RL policy is not available.
        """
        # Temporarily disable RL
        original_use_rl = self.use_rl
        self.use_rl = False

        try:
            return self.run(task)
        finally:
            self.use_rl = original_use_rl

    def get_agent_status(self) -> Dict[str, str]:
        """Get current status of all agents."""
        statuses = {}
        for name in ['Planner', 'Coder', 'Tester', 'Debugger']:
            if self._current_agent == name:
                statuses[name] = "active"
            else:
                statuses[name] = "idle"
        return statuses

    def is_rag_available(self) -> bool:
        """Check if RAG is available and initialized."""
        return self._retriever is not None and self._retriever.is_initialized()

    def is_rl_available(self) -> bool:
        """Check if RL policy is available."""
        return self._rl_agent is not None


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)

    print("Testing UI Pipeline...")

    pipeline = UIPipeline(use_rag=True, use_rl=False)

    def on_log(entry):
        print(f"  [{entry['agent']}] {entry['message']}")

    result = pipeline.run(
        "Write a function that reverses a string",
        on_log=on_log
    )

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Steps: {result.steps}")
    print(f"  Time: {result.time_seconds:.2f}s")
    print(f"  Code:\n{result.code[:500] if result.code else 'None'}")
