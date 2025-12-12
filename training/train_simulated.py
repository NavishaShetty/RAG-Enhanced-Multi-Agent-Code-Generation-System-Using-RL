"""
Training Script for Simulated Environment.

Trains the RL agent on the fast simulated environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import List, Dict

from environment.simulated_env import SimulatedEnv, SimulationConfig
from rl.combined_agent import CombinedAgent
from rl.q_learning import QLearningAgent


# Coding tasks for training variation
TRAINING_TASKS = [
    "Write a function that returns the sum of two numbers",
    "Write a function that reverses a string",
    "Write a function that checks if a number is even",
    "Write a function that finds the maximum in a list",
    "Write a function that counts vowels in a string",
    "Write a function that checks if a string is a palindrome",
    "Write a function that computes factorial recursively",
    "Write a function that returns Fibonacci sequence up to n",
    "Write a function that merges two sorted lists",
    "Write a function that removes duplicates from a list",
]


def train(
    num_episodes: int = 5000,
    eval_every: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.95,
    use_combined: bool = True,
    save_dir: str = "experiments/results"
) -> Dict:
    """
    Train the RL agent on simulated environment.

    Args:
        num_episodes: Number of training episodes
        eval_every: Evaluate every N episodes
        alpha: Learning rate
        gamma: Discount factor
        use_combined: Use CombinedAgent (True) or QLearningAgent (False)
        save_dir: Directory to save results

    Returns:
        Training history dictionary
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create environment and agent
    env = SimulatedEnv()

    if use_combined:
        agent = CombinedAgent(alpha=alpha, gamma=gamma)
        agent_type = "combined"
    else:
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=0.1)
        agent_type = "q_learning"

    # Training metrics
    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "successes": [],
        "eval_success_rates": [],
        "eval_avg_rewards": [],
        "eval_episodes": []
    }

    # Rolling windows for smoothing
    reward_window = deque(maxlen=100)
    success_window = deque(maxlen=100)

    print(f"Training {agent_type} agent for {num_episodes} episodes...")
    print(f"Hyperparameters: alpha={alpha}, gamma={gamma}")
    print("=" * 60)

    start_time = time.time()

    for episode in range(num_episodes):
        # Select random task for variety
        task = TRAINING_TASKS[episode % len(TRAINING_TASKS)]

        state = env.reset(task)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid, explore=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        # Record metrics
        history["episode_rewards"].append(total_reward)
        history["episode_lengths"].append(steps)
        history["successes"].append(1 if state.tests_pass else 0)

        reward_window.append(total_reward)
        success_window.append(1 if state.tests_pass else 0)

        # Periodic evaluation
        if (episode + 1) % eval_every == 0:
            eval_success, eval_reward = evaluate(agent, env, num_eval=100)
            history["eval_success_rates"].append(eval_success)
            history["eval_avg_rewards"].append(eval_reward)
            history["eval_episodes"].append(episode + 1)

            # Print progress
            elapsed = time.time() - start_time
            avg_reward = sum(reward_window) / len(reward_window)
            avg_success = sum(success_window) / len(success_window)

            print(f"Episode {episode + 1:5d} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Success: {avg_success:.0%} | "
                  f"Eval Success: {eval_success:.0%} | "
                  f"Time: {elapsed:.1f}s")

            # Decay exploration for Q-learning
            if not use_combined and hasattr(agent, 'decay_epsilon'):
                agent.decay_epsilon(0.99, 0.01)

    # Final evaluation
    final_success, final_reward = evaluate(agent, env, num_eval=500)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final eval success rate: {final_success:.0%}")
    print(f"Final eval avg reward: {final_reward:.2f}")

    # Save results
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "num_episodes": num_episodes,
            "alpha": alpha,
            "gamma": gamma,
            "training_time_seconds": elapsed
        },
        "final_metrics": {
            "success_rate": final_success,
            "avg_reward": final_reward
        },
        "history": history
    }

    # Save training history
    history_path = save_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Save Q-table
    q_table_path = save_path / "q_table.json"
    agent.save(str(q_table_path))
    print(f"Q-table saved to: {q_table_path}")

    return results


def evaluate(agent, env, num_eval: int = 100) -> tuple:
    """
    Evaluate agent without exploration.

    Args:
        agent: RL agent
        env: Environment
        num_eval: Number of evaluation episodes

    Returns:
        Tuple of (success_rate, avg_reward)
    """
    successes = 0
    total_reward = 0

    for i in range(num_eval):
        task = TRAINING_TASKS[i % len(TRAINING_TASKS)]
        state = env.reset(task)
        done = False
        ep_reward = 0

        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid, explore=False)
            state, reward, done = env.step(action)
            ep_reward += reward

        if state.tests_pass:
            successes += 1
        total_reward += ep_reward

    return successes / num_eval, total_reward / num_eval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent on simulation")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--eval-every", type=int, default=100, help="Eval frequency")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--agent", choices=["combined", "q_learning"], default="combined",
                        help="Agent type")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_every=args.eval_every,
        alpha=args.alpha,
        gamma=args.gamma,
        use_combined=(args.agent == "combined")
    )
