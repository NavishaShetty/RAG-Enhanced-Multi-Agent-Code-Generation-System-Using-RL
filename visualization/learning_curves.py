"""
Learning Curves Visualization.

Plots training progress: rewards, success rates, episode lengths.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def smooth(data: list, window: int = 50) -> np.ndarray:
    """
    Smooth data using moving average.

    Args:
        data: Raw data
        window: Smoothing window size

    Returns:
        Smoothed data
    """
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curves(
    history_path: str = "experiments/results/training_history.json",
    output_path: str = "experiments/results/learning_curves.png"
):
    """
    Plot learning curves from training history.

    Args:
        history_path: Path to training history JSON
        output_path: Path to save the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        data = json.load(f)

    history = data["history"]
    metadata = data["metadata"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Learning Curves - {metadata["agent_type"]} Agent\n'
                 f'(α={metadata["alpha"]}, γ={metadata["gamma"]}, '
                 f'{metadata["num_episodes"]} episodes)', fontsize=14)

    # 1. Episode Reward
    ax1 = axes[0, 0]
    rewards = history["episode_rewards"]
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) > 50:
        smoothed = smooth(rewards, 50)
        ax1.plot(range(49, 49 + len(smoothed)), smoothed, color='blue',
                 linewidth=2, label='Smoothed (50)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Reward over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Success Rate
    ax2 = axes[0, 1]
    successes = history["successes"]
    # Calculate running success rate
    running_rate = [sum(successes[:i+1])/(i+1) for i in range(len(successes))]
    ax2.plot(running_rate, alpha=0.3, color='green', label='Cumulative')
    if len(successes) > 50:
        smoothed = smooth(successes, 50)
        ax2.plot(range(49, 49 + len(smoothed)), smoothed, color='green',
                 linewidth=2, label='Window (50)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate over Training')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Episode Length
    ax3 = axes[1, 0]
    lengths = history["episode_lengths"]
    ax3.plot(lengths, alpha=0.3, color='orange', label='Raw')
    if len(lengths) > 50:
        smoothed = smooth(lengths, 50)
        ax3.plot(range(49, 49 + len(smoothed)), smoothed, color='orange',
                 linewidth=2, label='Smoothed (50)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Length over Training')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Evaluation Metrics
    ax4 = axes[1, 1]
    if history["eval_episodes"]:
        eval_eps = history["eval_episodes"]
        eval_success = history["eval_success_rates"]
        eval_reward = history["eval_avg_rewards"]

        ax4.plot(eval_eps, eval_success, 'go-', label='Eval Success Rate', linewidth=2)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(eval_eps, eval_reward, 'b^-', label='Eval Avg Reward', linewidth=2)

        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Success Rate', color='green')
        ax4_twin.set_ylabel('Average Reward', color='blue')
        ax4.set_title('Evaluation Metrics')
        ax4.set_ylim(0, 1.05)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Learning curves saved to: {output_path}")


def plot_comparison(
    histories: list,
    labels: list,
    output_path: str = "experiments/results/comparison.png"
):
    """
    Plot comparison of multiple training runs.

    Args:
        histories: List of history dictionaries
        labels: Labels for each run
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    # Success rate comparison
    ax1 = axes[0]
    for hist, label, color in zip(histories, labels, colors):
        successes = hist["history"]["successes"]
        if len(successes) > 50:
            smoothed = smooth(successes, 50)
            ax1.plot(range(49, 49 + len(smoothed)), smoothed,
                     color=color, linewidth=2, label=label)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate Comparison')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reward comparison
    ax2 = axes[1]
    for hist, label, color in zip(histories, labels, colors):
        rewards = hist["history"]["episode_rewards"]
        if len(rewards) > 50:
            smoothed = smooth(rewards, 50)
            ax2.plot(range(49, 49 + len(smoothed)), smoothed,
                     color=color, linewidth=2, label=label)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Reward Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    # Check if training history exists
    history_path = "experiments/results/training_history.json"

    if Path(history_path).exists():
        plot_learning_curves(history_path)
        print("Learning curves generated!")
    else:
        print(f"Training history not found at {history_path}")
        print("Run training/train_simulated.py first.")
