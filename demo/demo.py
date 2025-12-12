"""
Interactive Demo Script.

Shows the RL Code Generation system in action.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

from environment.coding_env import CodingEnv
from environment.simulated_env import SimulatedEnv
from orchestrator.fixed_pipeline import FixedPipeline
from rl.combined_agent import CombinedAgent


def demo_simulated():
    """Demo the RL agent on simulated environment."""
    print("\n" + "="*60)
    print("DEMO 1: RL Agent on Simulated Environment")
    print("="*60)

    # Load trained agent
    agent = CombinedAgent()
    q_table_path = "experiments/results/q_table.json"

    if Path(q_table_path).exists():
        agent.load(q_table_path)
        print(f" Loaded trained Q-table")
    else:
        print("! Q-table not found, using untrained agent")

    env = SimulatedEnv()

    tasks = [
        "Write a function that returns the sum of two numbers",
        "Write a function that reverses a string",
        "Write a function that checks if a number is even",
    ]

    print("\nRunning 3 episodes:")
    for i, task in enumerate(tasks, 1):
        state = env.reset(task)
        done = False
        steps = []

        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid, explore=False)
            steps.append(action)
            state, reward, done = env.step(action)

        result = "SUCCESS" if state.tests_pass else "FAIL"
        print(f"\n  Task {i}: {task[:40]}...")
        print(f"  Actions: {' -> '.join(steps)}")
        print(f"  Result: {result}")


def demo_fixed_vs_learned(task: str = None):
    """Compare fixed pipeline vs learned policy on same task."""
    print("\n" + "="*60)
    print("DEMO 2: Fixed Pipeline vs Learned Policy")
    print("="*60)

    if task is None:
        task = "Write a function that checks if a number is even"

    print(f"\nTask: {task}\n")

    # Fixed Pipeline
    print("-" * 40)
    print("FIXED PIPELINE:")
    print("-" * 40)

    try:
        pipeline = FixedPipeline(max_iterations=3)
        result = pipeline.run(task, verbose=False)

        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Agent calls: {result.agent_calls}")
        if result.final_code:
            print(f"  Code preview: {result.final_code[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")

    # Learned Policy
    print("\n" + "-" * 40)
    print("LEARNED POLICY:")
    print("-" * 40)

    try:
        agent = CombinedAgent()
        q_table_path = "experiments/results/q_table.json"
        if Path(q_table_path).exists():
            agent.load(q_table_path)

        env = CodingEnv(max_iterations=5)
        state = env.reset(task)
        done = False
        steps = []
        agent_calls = {"planner": 0, "coder": 0, "tester": 0, "debugger": 0}

        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid, explore=False)
            steps.append(action)
            agent_calls[action] += 1
            state, reward, done = env.step(action)

        print(f"  Success: {state.tests_pass}")
        print(f"  Steps: {len(steps)}")
        print(f"  Actions: {' -> '.join(steps)}")
        print(f"  Agent calls: {agent_calls}")
        if env.get_current_code():
            print(f"  Code preview: {env.get_current_code()[:100]}...")

    except Exception as e:
        print(f"  Error: {e}")


def demo_policy_interpretation():
    """Show what the RL agent learned."""
    print("\n" + "="*60)
    print("DEMO 3: What the RL Agent Learned")
    print("="*60)

    from environment.state import State

    agent = CombinedAgent()
    q_table_path = "experiments/results/q_table.json"

    if not Path(q_table_path).exists():
        print("Q-table not found. Run training first.")
        return

    agent.load(q_table_path)

    print("\nLearned strategy for key states:")

    states = [
        ("Starting fresh (no plan, no code)", State.initial()),
        ("Has plan, no code", State(has_plan=True)),
        ("Has code, no errors", State(has_plan=True, has_code=True)),
        ("Has code with errors", State(has_plan=True, has_code=True, has_error=True)),
    ]

    for desc, state in states:
        q_vals = agent.get_q_values(state)
        best = max(q_vals.keys(), key=lambda a: q_vals[a])
        print(f"\n  {desc}:")
        print(f"    -> Best action: {best} (Q={q_vals[best]:.2f})")

        # Show all Q-values
        sorted_q = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)
        others = ", ".join(f"{a}:{v:.1f}" for a, v in sorted_q[1:] if v != 0)
        if others:
            print(f"    -> Others: {others}")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("#  RL CODE GENERATION WORKFLOW AGENT - DEMO")
    print("#"*60)

    demo_simulated()
    demo_policy_interpretation()

    # Only run real LLM demo if API key is available
    if os.getenv("OPENROUTER_API_KEY"):
        print("\n" + "-"*60)
        response = input("Run comparison with real LLM? (y/n): ")
        if response.lower() == 'y':
            demo_fixed_vs_learned()
    else:
        print("\n[Skipping real LLM demo - OPENROUTER_API_KEY not set]")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
