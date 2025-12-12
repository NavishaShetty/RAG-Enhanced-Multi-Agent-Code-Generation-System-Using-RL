"""
Fixed Orchestration Pipeline - Hardcoded agent sequence.

This implements the baseline fixed-sequence pipeline:
Planner -> Coder -> Tester -> (Debugger -> Tester)*

The RL agent will later learn to improve upon this fixed strategy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

from agents.planner_agent import PlannerAgent
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.debugger_agent import DebuggerAgent
from communication.blackboard import Blackboard, MessageType
from tools.test_runner import TestRunner
from tools.complexity_analyzer import ComplexityAnalyzer


@dataclass
class PipelineResult:
    """Result of running the fixed pipeline."""
    success: bool
    final_code: Optional[str]
    iterations: int
    agent_calls: Dict[str, int]
    time_elapsed: float
    test_results: Optional[dict] = None
    complexity_metrics: Optional[dict] = None
    error_message: Optional[str] = None
    history: List[dict] = field(default_factory=list)


class FixedPipeline:
    """
    Fixed orchestration pipeline with hardcoded sequence.

    Sequence: Planner -> Coder -> Tester -> (Debugger -> Tester)*
    Max iterations: configurable (default 5)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_iterations: int = 5
    ):
        """
        Initialize the fixed pipeline.

        Args:
            api_key: OpenRouter API key
            max_iterations: Maximum debug-test iterations
        """
        self.planner = PlannerAgent(api_key=api_key)
        self.coder = CoderAgent(api_key=api_key)
        self.tester = TesterAgent(api_key=api_key)
        self.debugger = DebuggerAgent(api_key=api_key)

        self.test_runner = TestRunner()
        self.complexity_analyzer = ComplexityAnalyzer()

        self.max_iterations = max_iterations

    def run(self, task: str, verbose: bool = True) -> PipelineResult:
        """
        Run the fixed pipeline on a coding task.

        Args:
            task: The coding task description
            verbose: Whether to print progress

        Returns:
            PipelineResult with all metrics
        """
        start_time = time.time()
        blackboard = Blackboard()
        blackboard.set_task(task)

        agent_calls = {"planner": 0, "coder": 0, "tester": 0, "debugger": 0}
        history = []
        iterations = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}\n")

        try:
            # Step 1: Plan
            if verbose:
                print("1. Planning...")
            plan = self.planner.generate_plan(task, blackboard)
            agent_calls["planner"] += 1
            history.append({"step": "plan", "content": plan[:200]})
            if verbose:
                print(f"   Plan generated ({len(plan)} chars)")

            # Step 2: Code
            if verbose:
                print("2. Coding...")
            code = self.coder.generate_code(task, blackboard)
            agent_calls["coder"] += 1
            history.append({"step": "code", "content": code[:200]})
            if verbose:
                print(f"   Code generated ({len(code)} chars)")

            # Step 3-N: Test and Debug loop
            current_code = code
            for iteration in range(self.max_iterations):
                iterations = iteration + 1

                # Test the code
                if verbose:
                    print(f"3.{iteration + 1}. Testing (iteration {iterations})...")

                # Run actual tests
                test_result = self.test_runner.run_tests(current_code, task)

                # Also get LLM analysis
                analysis = self.tester.analyze_code(current_code, blackboard, task)
                agent_calls["tester"] += 1

                passed = test_result.all_passed and self.tester.check_passed(analysis)

                history.append({
                    "step": f"test_{iterations}",
                    "passed": passed,
                    "test_passed": test_result.tests_passed,
                    "test_total": test_result.total_tests
                })

                if verbose:
                    print(f"   Tests: {test_result.tests_passed}/{test_result.total_tests} passed")
                    print(f"   Analysis: {'PASS' if self.tester.check_passed(analysis) else 'Issues found'}")

                if passed:
                    # Success!
                    elapsed = time.time() - start_time
                    metrics = self.complexity_analyzer.analyze(current_code)

                    if verbose:
                        print(f"\n SUCCESS after {iterations} iteration(s)!")
                        print(f"   Time: {elapsed:.2f}s")
                        print(f"   Complexity score: {metrics.overall_score()}")

                    return PipelineResult(
                        success=True,
                        final_code=current_code,
                        iterations=iterations,
                        agent_calls=agent_calls,
                        time_elapsed=elapsed,
                        test_results={
                            "passed": test_result.tests_passed,
                            "total": test_result.total_tests
                        },
                        complexity_metrics=metrics.to_dict(),
                        history=history
                    )

                # Debug
                if verbose:
                    print(f"4.{iteration + 1}. Debugging...")

                error_info = analysis
                if test_result.error_messages:
                    error_info += "\nTest failures:\n" + "\n".join(test_result.error_messages[:3])

                current_code = self.debugger.fix_code(current_code, error_info, blackboard)
                agent_calls["debugger"] += 1
                history.append({"step": f"debug_{iterations}", "content": current_code[:200]})

                if verbose:
                    print(f"   Fixed code generated")

            # Max iterations reached
            elapsed = time.time() - start_time
            metrics = self.complexity_analyzer.analyze(current_code)

            if verbose:
                print(f"\n✗ TIMEOUT after {iterations} iterations")

            return PipelineResult(
                success=False,
                final_code=current_code,
                iterations=iterations,
                agent_calls=agent_calls,
                time_elapsed=elapsed,
                test_results={
                    "passed": test_result.tests_passed,
                    "total": test_result.total_tests
                },
                complexity_metrics=metrics.to_dict(),
                error_message="Max iterations reached",
                history=history
            )

        except Exception as e:
            elapsed = time.time() - start_time
            if verbose:
                print(f"\n✗ ERROR: {e}")

            return PipelineResult(
                success=False,
                final_code=None,
                iterations=iterations,
                agent_calls=agent_calls,
                time_elapsed=elapsed,
                error_message=str(e),
                history=history
            )


if __name__ == "__main__":
    # Quick test
    pipeline = FixedPipeline(max_iterations=3)

    task = "Write a function that checks if a number is even"
    result = pipeline.run(task, verbose=True)

    print(f"\n{'='*60}")
    print("Pipeline Result Summary:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Agent calls: {result.agent_calls}")
    print(f"  Time: {result.time_elapsed:.2f}s")
    if result.complexity_metrics:
        print(f"  Complexity: {result.complexity_metrics['overall_score']}")
    if result.final_code:
        print(f"\nFinal code:\n{result.final_code}")
