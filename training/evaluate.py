"""
Evaluation Script - Compare fixed vs learned policy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from datetime import datetime


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compare_policies(
    baseline_path: str = "experiments/results/baseline.json",
    validation_path: str = "experiments/results/validation.json",
    output_path: str = "experiments/results/comparison.json"
) -> dict:
    """
    Compare fixed pipeline vs learned policy.

    Args:
        baseline_path: Path to baseline (fixed policy) results
        validation_path: Path to validation (learned policy) results
        output_path: Path to save comparison

    Returns:
        Comparison data dictionary
    """
    print("Loading results...")

    baseline = load_json(baseline_path)
    validation = load_json(validation_path)

    baseline_summary = baseline["summary"]
    validation_summary = validation["summary"]

    print(f"\n{'='*60}")
    print("POLICY COMPARISON")
    print(f"{'='*60}")

    # Create comparison table
    comparison = {
        "fixed_policy": {
            "success_rate": baseline_summary["success_rate"],
            "avg_iterations": baseline_summary.get("avg_iterations", "N/A"),
            "avg_time_seconds": baseline_summary.get("avg_time_seconds", "N/A"),
            "total_agent_calls": baseline_summary.get("total_agent_calls", {})
        },
        "learned_policy": {
            "success_rate": validation_summary["success_rate"],
            "avg_reward": validation_summary.get("avg_reward", "N/A")
        },
        "improvement": {
            "success_rate_diff": validation_summary["success_rate"] - baseline_summary["success_rate"]
        }
    }

    # Print comparison
    print("\n{:<30} {:<20} {:<20}".format("Metric", "Fixed Policy", "Learned Policy"))
    print("-" * 70)

    print("{:<30} {:<20.1%} {:<20.1%}".format(
        "Success Rate",
        baseline_summary["success_rate"],
        validation_summary["success_rate"]
    ))

    if "avg_iterations" in baseline_summary:
        print("{:<30} {:<20.2f} {:<20}".format(
            "Avg Iterations",
            baseline_summary["avg_iterations"],
            "N/A (varies)"
        ))

    if "avg_time_seconds" in baseline_summary:
        print("{:<30} {:<20.2f}s {:<20}".format(
            "Avg Time/Task",
            baseline_summary["avg_time_seconds"],
            "N/A"
        ))

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    if comparison["improvement"]["success_rate_diff"] > 0:
        print(f"✓ Learned policy improved success rate by {comparison['improvement']['success_rate_diff']*100:.1f}%")
    elif comparison["improvement"]["success_rate_diff"] < 0:
        print(f"✗ Learned policy decreased success rate by {abs(comparison['improvement']['success_rate_diff'])*100:.1f}%")
    else:
        print("= Success rates are equal")

    # Save comparison
    comparison_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "baseline_path": baseline_path,
            "validation_path": validation_path
        },
        "comparison": comparison
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nComparison saved to: {output_path}")

    return comparison_data


if __name__ == "__main__":
    try:
        compare_policies()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to run these first:")
        print("  1. scripts/collect_baseline.py (for baseline.json)")
        print("  2. training/validate_real.py (for validation.json)")
