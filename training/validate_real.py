"""
Validation Script - Test learned policy on real LLM environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from environment.coding_env import CodingEnv
from rl.combined_agent import CombinedAgent


# Tasks for validation (subset of training tasks)
VALIDATION_TASKS = [
    "Write a function that returns the sum of two numbers",
    "Write a function that reverses a string",
    "Write a function that checks if a number is even",
    "Write a function that finds the maximum in a list",
    "Write a function that counts vowels in a string",
]


def validate_learned_policy(
    q_table_path: str = "experiments/results/q_table.json",
    num_tasks: int = 5,
    max_iterations: int = 5,
    verbose: bool = True
) -> dict:
    """
    Validate learned policy on real LLM environment.

    Args:
        q_table_path: Path to saved Q-table
        num_tasks: Number of tasks to run
        max_iterations: Max iterations per task
        verbose: Print progress

    Returns:
        Validation results dictionary
    """
    # Load trained agent
    agent = CombinedAgent()
    agent.load(q_table_path)
    print(f"Loaded Q-table from: {q_table_path}")

    # Create real environment
    env = CodingEnv(max_iterations=max_iterations)

    tasks = VALIDATION_TASKS[:num_tasks]
    results = []
    total_success = 0
    total_reward = 0

    print(f"\n{'='*60}")
    print("VALIDATION WITH REAL LLM")
    print(f"Tasks: {len(tasks)}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*60}\n")

    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"[{i}/{len(tasks)}] {task}")

        try:
            state = env.reset(task)
            done = False
            total_task_reward = 0
            actions_taken = []
            steps = 0

            while not done and steps < max_iterations * 2:
                valid = env.get_valid_actions()
                # Use learned policy (no exploration)
                action = agent.choose_action(state, valid, explore=False)
                actions_taken.append(action)

                state, reward, done = env.step(action)
                total_task_reward += reward
                steps += 1

                if verbose:
                    print(f"   Step {steps}: {action} -> {state} (r={reward:.2f})")

            success = state.tests_pass
            if success:
                total_success += 1

            results.append({
                "task_id": i,
                "task": task,
                "success": success,
                "steps": steps,
                "reward": total_task_reward,
                "actions": actions_taken,
                "final_code": env.get_current_code()
            })

            total_reward += total_task_reward

            status = "Pass" if success else "Fail"
            if verbose:
                print(f"   {status} {'Success' if success else 'Failed'}\n")

        except Exception as e:
            print(f"   Error: {e}\n")
            results.append({
                "task_id": i,
                "task": task,
                "success": False,
                "error": str(e)
            })

    # Summary
    summary = {
        "total_tasks": len(tasks),
        "successful": total_success,
        "success_rate": total_success / len(tasks) if tasks else 0,
        "avg_reward": total_reward / len(tasks) if tasks else 0
    }

    print(f"{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Success rate: {summary['success_rate']*100:.0f}% ({summary['successful']}/{summary['total_tasks']})")
    print(f"Average reward: {summary['avg_reward']:.2f}")

    validation_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "q_table_path": q_table_path,
            "max_iterations": max_iterations
        },
        "summary": summary,
        "results": results
    }

    return validation_data


def save_validation(data: dict, output_path: str = "experiments/results/validation.json"):
    """Save validation results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nValidation saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate learned policy on real LLM")
    parser.add_argument("--tasks", type=int, default=3, help="Number of tasks")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations")
    parser.add_argument("--quiet", action="store_true", help="Less verbose")
    args = parser.parse_args()

    validation_data = validate_learned_policy(
        num_tasks=args.tasks,
        max_iterations=args.max_iter,
        verbose=not args.quiet
    )

    save_validation(validation_data)
