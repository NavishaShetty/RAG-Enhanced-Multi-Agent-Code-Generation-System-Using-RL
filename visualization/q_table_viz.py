"""
Q-Table Visualization.

Visualize the learned Q-values as heatmaps and policy diagrams.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from environment.state import State, ACTIONS


def load_q_table(path: str) -> dict:
    """Load Q-table from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    q_table = {}
    for key_str, values in data.get("q_table", {}).items():
        key = eval(key_str)
        q_table[key] = values

    return q_table


def q_table_to_matrix(q_table: dict) -> tuple:
    """
    Convert Q-table to matrix format for heatmap.

    Returns:
        Tuple of (q_matrix, state_labels, action_labels)
    """
    # Create matrix: rows = states, cols = actions
    num_states = 64  # 2^4 * 4 = 64 states
    num_actions = len(ACTIONS)

    q_matrix = np.zeros((num_states, num_actions))

    for i in range(num_states):
        state = State.from_index(i)
        state_key = state.to_key()

        if state_key in q_table:
            for j, action in enumerate(ACTIONS):
                q_matrix[i, j] = q_table[state_key].get(action, 0)

    return q_matrix


def plot_q_table_heatmap(
    q_table_path: str = "experiments/results/q_table.json",
    output_path: str = "experiments/results/q_table_heatmap.png"
):
    """
    Plot Q-table as heatmap.

    Args:
        q_table_path: Path to Q-table JSON
        output_path: Path to save heatmap
    """
    q_table = load_q_table(q_table_path)
    q_matrix = q_table_to_matrix(q_table)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 16))

    # Only show states with non-zero Q-values
    non_zero_rows = np.any(q_matrix != 0, axis=1)
    active_indices = np.where(non_zero_rows)[0]

    if len(active_indices) == 0:
        print("Warning: Q-table appears empty")
        active_indices = np.arange(min(20, len(q_matrix)))

    q_active = q_matrix[active_indices]

    # Create state labels
    state_labels = []
    for i in active_indices:
        s = State.from_index(i)
        label = f"{i}: "
        parts = []
        if s.has_plan:
            parts.append("P")
        if s.has_code:
            parts.append("C")
        if s.has_error:
            parts.append("E")
        if s.tests_pass:
            parts.append("pass")
        label += "+".join(parts) if parts else "empty"
        label += f" (i={s.iteration})"
        state_labels.append(label)

    # Plot heatmap
    im = ax.imshow(q_active, cmap='RdYlGn', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Q-Value', rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(len(ACTIONS)))
    ax.set_yticks(np.arange(len(state_labels)))
    ax.set_xticklabels(ACTIONS)
    ax.set_yticklabels(state_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add value annotations
    for i in range(len(state_labels)):
        for j in range(len(ACTIONS)):
            value = q_active[i, j]
            text_color = 'white' if abs(value) > 3 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   color=text_color, fontsize=8)

    ax.set_title('Q-Table Heatmap\n(P=Plan, C=Code, E=Error, pass=Pass, i=iteration)')
    ax.set_xlabel('Action')
    ax.set_ylabel('State')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Q-table heatmap saved to: {output_path}")


def plot_policy_diagram(
    q_table_path: str = "experiments/results/q_table.json",
    output_path: str = "experiments/results/policy_diagram.png"
):
    """
    Plot the learned policy as a state diagram.

    Args:
        q_table_path: Path to Q-table JSON
        output_path: Path to save diagram
    """
    q_table = load_q_table(q_table_path)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Key states and their positions
    states = [
        (State.initial(), (0.1, 0.5), "Start\n(empty)"),
        (State(has_plan=True), (0.3, 0.7), "Has Plan"),
        (State(has_plan=True, has_code=True), (0.5, 0.5), "Has Code"),
        (State(has_plan=True, has_code=True, has_error=True), (0.5, 0.2), "Has Error"),
        (State(has_plan=True, has_code=True, tests_pass=True), (0.8, 0.5), "SUCCESS\n(Tests Pass)"),
    ]

    # Draw states
    for state, (x, y), label in states:
        color = 'lightgreen' if state.tests_pass else 'lightblue'
        circle = plt.Circle((x, y), 0.08, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

        # Get best action for this state
        state_key = state.to_key()
        if state_key in q_table:
            q_vals = q_table[state_key]
            best_action = max(q_vals.keys(), key=lambda a: q_vals[a])
            best_q = q_vals[best_action]

            # Show best action
            ax.text(x, y - 0.12, f"→ {best_action}\n(Q={best_q:.1f})",
                   ha='center', va='top', fontsize=8, color='darkblue')

    # Draw arrows showing transitions
    arrows = [
        ((0.18, 0.5), (0.22, 0.67), "plan"),
        ((0.38, 0.67), (0.42, 0.53), "code"),
        ((0.58, 0.5), (0.72, 0.5), "test\n(pass)"),
        ((0.55, 0.45), (0.55, 0.28), "test\n(fail)"),
        ((0.5, 0.12), (0.48, 0.42), "debug"),
    ]

    for start, end, label in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(mid[0], mid[1], label, ha='center', va='center',
               fontsize=8, color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Learned Policy Overview\n(showing best action per state)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Policy diagram saved to: {output_path}")


def print_policy_summary(q_table_path: str = "experiments/results/q_table.json"):
    """Print human-readable policy summary."""
    q_table = load_q_table(q_table_path)

    print("\n" + "="*60)
    print("LEARNED POLICY INTERPRETATION")
    print("="*60)

    # Key insights
    key_states = [
        ("Initial (no plan, no code)", State.initial()),
        ("Has plan only", State(has_plan=True)),
        ("Has plan + code", State(has_plan=True, has_code=True)),
        ("Has plan + code + error", State(has_plan=True, has_code=True, has_error=True)),
    ]

    for desc, state in key_states:
        state_key = state.to_key()
        if state_key in q_table:
            q_vals = q_table[state_key]
            best = max(q_vals.keys(), key=lambda a: q_vals[a])
            print(f"\n{desc}:")
            print(f"  Best action: {best} (Q={q_vals[best]:.2f})")
            print(f"  All Q-values: {', '.join(f'{a}:{v:.2f}' for a,v in q_vals.items())}")
        else:
            print(f"\n{desc}: (no data)")


if __name__ == "__main__":
    q_table_path = "experiments/results/q_table.json"

    if Path(q_table_path).exists():
        plot_q_table_heatmap(q_table_path)
        plot_policy_diagram(q_table_path)
        print_policy_summary(q_table_path)
        print("\nQ-table visualization complete!")
    else:
        print(f"Q-table not found at {q_table_path}")
        print("Run training/train_simulated.py first.")
