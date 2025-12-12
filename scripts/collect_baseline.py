"""
V1.8: Baseline Metrics Collection

Runs the fixed pipeline on coding tasks and saves baseline metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from pathlib import Path

from orchestrator.fixed_pipeline import FixedPipeline


# Coding tasks from CLAUDE.md
CODING_TASKS = [
    # Simple (1-5)
    "Write a function that returns the sum of two numbers",
    "Write a function that reverses a string",
    "Write a function that checks if a number is even",
    "Write a function that finds the maximum in a list",
    "Write a function that counts vowels in a string",
    # Medium (6-10)
    "Write a function that checks if a string is a palindrome",
    "Write a function that computes factorial recursively",
    "Write a function that returns Fibonacci sequence up to n",
    "Write a function that merges two sorted lists",
    "Write a function that removes duplicates from a list",
]


def collect_baseline(num_tasks: int = 10, max_iterations: int = 5, verbose: bool = True):
    """
    Collect baseline metrics from the fixed pipeline.

    Args:
        num_tasks: Number of tasks to run (max 10)
        max_iterations: Max debug iterations per task
        verbose: Print progress

    Returns:
        Dictionary of collected metrics
    """
    pipeline = FixedPipeline(max_iterations=max_iterations)
    tasks = CODING_TASKS[:num_tasks]

    results = []
    total_success = 0
    total_iterations = 0
    total_time = 0
    total_agent_calls = {"planner": 0, "coder": 0, "tester": 0, "debugger": 0}

    print(f"\n{'='*60}")
    print("BASELINE METRICS COLLECTION")
    print(f"Tasks: {len(tasks)}")
    print(f"Max iterations per task: {max_iterations}")
    print(f"{'='*60}\n")

    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task[:50]}...")

        result = pipeline.run(task, verbose=verbose)

        task_result = {
            "task_id": i,
            "task": task,
            "success": result.success,
            "iterations": result.iterations,
            "time": round(result.time_elapsed, 2),
            "agent_calls": result.agent_calls,
            "test_results": result.test_results,
            "complexity": result.complexity_metrics,
            "error": result.error_message
        }
        results.append(task_result)

        # Aggregate stats
        if result.success:
            total_success += 1
        total_iterations += result.iterations
        total_time += result.time_elapsed
        for agent, count in result.agent_calls.items():
            total_agent_calls[agent] += count

        # Print task summary
        status = "✓" if result.success else "✗"
        print(f"   {status} {'Success' if result.success else 'Failed'} in {result.iterations} iterations ({result.time_elapsed:.1f}s)")

    # Calculate summary statistics
    summary = {
        "total_tasks": len(tasks),
        "successful_tasks": total_success,
        "success_rate": round(total_success / len(tasks), 3),
        "avg_iterations": round(total_iterations / len(tasks), 2),
        "avg_time_seconds": round(total_time / len(tasks), 2),
        "total_time_seconds": round(total_time, 2),
        "total_agent_calls": total_agent_calls,
        "avg_agent_calls_per_task": {
            k: round(v / len(tasks), 2) for k, v in total_agent_calls.items()
        }
    }

    baseline_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pipeline_type": "fixed",
            "max_iterations": max_iterations
        },
        "summary": summary,
        "results": results
    }

    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Success rate: {summary['success_rate']*100:.0f}% ({summary['successful_tasks']}/{summary['total_tasks']})")
    print(f"Avg iterations: {summary['avg_iterations']}")
    print(f"Avg time per task: {summary['avg_time_seconds']}s")
    print(f"Total time: {summary['total_time_seconds']}s")
    print(f"Agent calls (avg): {summary['avg_agent_calls_per_task']}")

    return baseline_data


def save_baseline(data: dict, output_path: str = "experiments/results/baseline.json"):
    """Save baseline data to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nBaseline saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect baseline metrics")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks to run (1-10)")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations per task")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    baseline_data = collect_baseline(
        num_tasks=min(args.tasks, 10),
        max_iterations=args.max_iter,
        verbose=not args.quiet
    )

    save_baseline(baseline_data)
