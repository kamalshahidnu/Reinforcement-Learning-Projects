"""
EECE 5614 - Reinforcement Learning and Decision Making Under Uncertainty
Project 1: 2-Armed Bandit Problem



Environment:
- Lever 1 (a¹): Gaussian(μ=8, σ²=20) → Q*(a¹) = 8
- Lever 2 (a²): Mixture of Gaussians → Q*(a²) = 11
"""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)


# CONFIGURATION


N_ACTIONS = 2
N_STEPS = 1000
N_RUNS = 100
OUTPUT_DIR = 'plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ENVIRONMENT


class TwoArmedBandit:
    """2-Armed Bandit with Gaussian and Mixture-of-Gaussians rewards."""
    
    def __init__(self):
        self.q_star = np.array([8.0, 11.0])
    
    def pull(self, action: int) -> float:
        if action == 0:
            return np.random.normal(8, np.sqrt(20))
        else:
            if np.random.random() < 0.5:
                return np.random.normal(8, np.sqrt(15))
            return np.random.normal(14, np.sqrt(10))



# AGENTS


class BaseAgent(ABC):
    """Abstract base class for bandit agents."""
    
    def __init__(self, n_actions: int = N_ACTIONS):
        self.n_actions = n_actions
        self.step_count = 0
    
    @abstractmethod
    def select_action(self) -> int:
        pass
    
    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        pass
    
    def _argmax_random_tie_break(self, values: np.ndarray) -> int:
        max_val = np.max(values)
        return np.random.choice(np.where(values == max_val)[0])


class EpsilonGreedyAgent(BaseAgent):
    """ε-greedy agent with configurable learning rate."""
    
    def __init__(
        self,
        epsilon: float = 0.1,
        learning_rate_fn: Callable[[int], float] = None,
        initial_q: np.ndarray = None
    ):
        super().__init__()
        self.epsilon = epsilon
        self.learning_rate_fn = learning_rate_fn or (lambda k: 0.1)
        self._initial_q = initial_q if initial_q is not None else np.zeros(N_ACTIONS)
        self.q_values = self._initial_q.copy()
    
    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self._argmax_random_tie_break(self.q_values)
    
    def update(self, action: int, reward: float) -> None:
        self.step_count += 1
        alpha = self.learning_rate_fn(self.step_count)
        self.q_values[action] += alpha * (reward - self.q_values[action])


class GradientBanditAgent(BaseAgent):
    """Gradient bandit with softmax action selection."""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.preferences = np.zeros(N_ACTIONS)
        self.total_reward = 0.0
    
    def _softmax(self) -> np.ndarray:
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))
        return exp_prefs / np.sum(exp_prefs)
    
    def select_action(self) -> int:
        return np.random.choice(self.n_actions, p=self._softmax())
    
    def update(self, action: int, reward: float) -> None:
        self.step_count += 1
        self.total_reward += reward
        baseline = self.total_reward / self.step_count
        probs = self._softmax()
        advantage = reward - baseline
        self.preferences -= self.alpha * advantage * probs
        self.preferences[action] += self.alpha * advantage


class UCBAgent(BaseAgent):
    """Upper Confidence Bound agent."""
    
    def __init__(self, c: float = 2.0):
        super().__init__()
        self.c = c
        self.q_values = np.zeros(N_ACTIONS)
        self.action_counts = np.zeros(N_ACTIONS)
    
    def select_action(self) -> int:
        self.step_count += 1
        untried = np.where(self.action_counts == 0)[0]
        if len(untried) > 0:
            return untried[0]
        exploration = self.c * np.sqrt(np.log(self.step_count) / self.action_counts)
        return self._argmax_random_tie_break(self.q_values + exploration)
    
    def update(self, action: int, reward: float) -> None:
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])


# EXPERIMENT RUNNER


def run_experiment(agent_factory: Callable, track_q: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run experiment and return average accumulated rewards."""
    rewards = np.zeros((N_RUNS, N_STEPS))
    final_q = np.zeros((N_RUNS, N_ACTIONS)) if track_q else None
    bandit = TwoArmedBandit()
    
    for run in range(N_RUNS):
        agent = agent_factory()
        total = 0.0
        for step in range(N_STEPS):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
            total += reward
            rewards[run, step] = total / (step + 1)
        if track_q and hasattr(agent, 'q_values'):
            final_q[run] = agent.q_values.copy()
    
    return np.mean(rewards, axis=0), final_q



# VISUALIZATION


def plot_curves(data: Dict[str, np.ndarray], title: str, filename: str, labels: Dict = None):
    """Plot multiple curves on a single figure."""
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, (name, rewards) in enumerate(data.items()):
        label = labels.get(name, name) if labels else name
        ax.plot(range(1, N_STEPS + 1), rewards, color=colors[idx % 4], label=label, linewidth=2)
    
    ax.set_xlabel('Step (k)', fontsize=12)
    ax.set_ylabel(r'$\overline{AccR}_k$', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, N_STEPS])
    plt.tight_layout()
    
    filepath = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to '{filepath}'")


def plot_grid(data: Dict[str, Dict[str, np.ndarray]], title: str, filename: str):
    """Plot 2x2 grid of subplots."""
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (subplot_title, curves) in enumerate(data.items()):
        ax = axes.flatten()[idx]
        for cidx, (name, rewards) in enumerate(curves.items()):
            ax.plot(range(1, N_STEPS + 1), rewards, color=colors[cidx], label=name, linewidth=1.5)
        ax.set_xlabel('Step (k)', fontsize=11)
        ax.set_ylabel(r'$\overline{AccR}_k$', fontsize=11)
        ax.set_title(f'Learning Rate: {subplot_title}', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, N_STEPS])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to '{filepath}'")


def print_table(data: List[Tuple[str, float]], headers: Tuple[str, str], filename: str = None):
    """Print and save formatted table."""
    lines = []
    sep = "─" * 65
    lines.append(f"\n{sep}")
    lines.append(f"{headers[0]:<45} | {headers[1]:>15}")
    lines.append(sep)
    for label, value in data:
        lines.append(f"{label:<45} | {value:>15.4f}")
    lines.append(sep)
    
    output = '\n'.join(lines)
    print(output)
    
    if filename:
        filepath = f"{OUTPUT_DIR}/{filename}"
        with open(filepath, 'w') as f:
            f.write(output)
        print(f"Table saved to '{filepath}'")


def print_q_table(results: Dict, title: str, filename: str = None):
    """Print Q-value comparison table."""
    lines = [f"\n{'='*75}", title, "True values: Q*(a¹) = 8, Q*(a²) = 11", '='*75]
    sep = "─" * 75
    lines.append(f"\n{sep}")
    lines.append(f"{'Config':<25} | {'Q(a¹)':>12} | {'Q*(a¹)':>8} | {'Q(a²)':>12} | {'Q*(a²)':>8}")
    lines.append(sep)
    
    for label, data in results.items():
        q1, q2 = data['mean_q_a1'], data['mean_q_a2']
        label_clean = label.replace('$', '').replace('\\epsilon', 'ε')
        lines.append(f"{label_clean:<25} | {q1:>12.4f} | {8:>8} | {q2:>12.4f} | {11:>8}")
    lines.append(sep)
    
    output = '\n'.join(lines)
    print(output)
    
    if filename:
        filepath = f"{OUTPUT_DIR}/{filename}"
        with open(filepath, 'w') as f:
            f.write(output)
        print(f"Table saved to '{filepath}'")



# LEARNING RATE FUNCTIONS


LEARNING_RATES = {
    r'$\alpha = 1$': lambda k: 1.0,
    r'$\alpha = 0.9^k$': lambda k: 0.9 ** k,
    r'$\alpha = \frac{1}{1 + \ln(1+k)}$': lambda k: 1.0 / (1 + np.log(1 + k)),
    r'$\alpha = \frac{1}{k}$': lambda k: 1.0 / k
}

EPSILONS = [0.0, 0.1, 0.2, 0.5]
EPSILON_LABELS = [r'$\epsilon = 0$ (greedy)', r'$\epsilon = 0.1$', 
                  r'$\epsilon = 0.2$', r'$\epsilon = 0.5$ (random)']


# PART 1: ε-greedy with Different Learning Rates


def run_part1():
    print("\n" + "=" * 70)
    print("PART 1: ε-greedy with Different Learning Rates")
    print("=" * 70)
    
    results = {}
    total = len(LEARNING_RATES) * len(EPSILONS)
    current = 0
    
    for lr_name, lr_fn in LEARNING_RATES.items():
        results[lr_name] = {}
        for eps, eps_label in zip(EPSILONS, EPSILON_LABELS):
            current += 1
            print(f"Running {current}/{total}: {lr_name}, ε={eps}")
            
            avg_rewards, final_q = run_experiment(
                lambda lr=lr_fn, e=eps: EpsilonGreedyAgent(epsilon=e, learning_rate_fn=lr),
                track_q=True
            )
            results[lr_name][eps_label] = {
                'rewards': avg_rewards,
                'mean_q_a1': np.mean(final_q[:, 0]),
                'mean_q_a2': np.mean(final_q[:, 1])
            }
    
    # Tables
    for idx, (lr_name, eps_results) in enumerate(results.items()):
        print_q_table(eps_results, f"Part 1 - {lr_name}", f"part1_table_{idx+1}.txt")
    
    # Plot
    plot_data = {lr: {el: d['rewards'] for el, d in er.items()} for lr, er in results.items()}
    plot_grid(plot_data, 'Part 1: Average Accumulated Reward\n(100 runs)', 'part1_plot.png')
    
    return results


# PART 2: Optimistic Initial Values


def run_part2():
    print("\n" + "=" * 70)
    print("PART 2: Optimistic Initial Values")
    print("=" * 70)
    
    settings = {
        'Q = [0, 0]': np.array([0.0, 0.0]),
        'Q = [8, 11]': np.array([8.0, 11.0]),
        'Q = [20, 20]': np.array([20.0, 20.0])
    }
    
    results = {}
    for idx, (label, init_q) in enumerate(settings.items()):
        print(f"Running {idx+1}/{len(settings)}: {label}")
        avg_rewards, final_q = run_experiment(
            lambda q=init_q: EpsilonGreedyAgent(epsilon=0.1, learning_rate_fn=lambda k: 0.1, initial_q=q.copy()),
            track_q=True
        )
        results[label] = {
            'rewards': avg_rewards,
            'mean_q_a1': np.mean(final_q[:, 0]),
            'mean_q_a2': np.mean(final_q[:, 1])
        }
    
    print_q_table(results, "Part 2 - Optimistic Initialization (α=0.1, ε=0.1)", "part2_table.txt")
    plot_curves({k: v['rewards'] for k, v in results.items()}, 
                'Part 2: Optimistic Initial Values\n(100 runs)', 'part2_plot.png')
    
    return results



# PART 3: Gradient-Bandit


def run_part3():
    print("\n" + "=" * 70)
    print("PART 3: Gradient-Bandit Policy")
    print("=" * 70)
    
    print("Running Gradient-Bandit...")
    gb_rewards, _ = run_experiment(lambda: GradientBanditAgent(alpha=0.1))
    
    print("Running ε-greedy baseline...")
    eg_rewards, _ = run_experiment(
        lambda: EpsilonGreedyAgent(epsilon=0.1, learning_rate_fn=lambda k: 0.1)
    )
    
    results = {'Gradient-Bandit': gb_rewards, 'ε-greedy': eg_rewards}
    
    print_table([
        ('Gradient-Bandit (α=0.1)', gb_rewards[-1]),
        ('ε-greedy (α=0.1, ε=0.1)', eg_rewards[-1])
    ], ('Method', 'Final AccR̄'), 'part3_table.txt')
    
    plot_curves(results, 'Part 3: Gradient-Bandit vs ε-greedy\n(100 runs)', 'part3_plot.png',
                {'Gradient-Bandit': r'Gradient-Bandit ($\alpha=0.1$)',
                 'ε-greedy': r'$\epsilon$-greedy ($\alpha=0.1$, $\epsilon=0.1$)'})
    
    return results



# PART 4: UCB


def run_part4():
    print("\n" + "=" * 70)
    print("PART 4: Upper Confidence Bound (UCB)")
    print("=" * 70)
    
    # UCB experiments
    ucb_results = {}
    for c in [2, 5, 100]:
        print(f"Running UCB (c={c})...")
        rewards, _ = run_experiment(lambda c=c: UCBAgent(c=c))
        ucb_results[f'UCB (c={c})'] = rewards
    
    # Plot 1: UCB comparison
    plot_curves(ucb_results, 'Part 4: UCB Performance\n(100 runs)', 'part4_ucb.png')
    
    # Find best UCB
    best_c = max(ucb_results.keys(), key=lambda k: ucb_results[k][-1])
    
    # Comparison
    print("Running best ε-greedy...")
    eg_rewards, _ = run_experiment(
        lambda: EpsilonGreedyAgent(epsilon=0.1, learning_rate_fn=lambda k: 1/(1+np.log(1+k)))
    )
    
    print("Running Gradient-Bandit...")
    gb_rewards, _ = run_experiment(lambda: GradientBanditAgent(alpha=0.1))
    
    comparison = {best_c: ucb_results[best_c], 'Best ε-greedy': eg_rewards, 'Gradient-Bandit': gb_rewards}
    
    # Plot 2: Method comparison
    plot_curves(comparison, 'Part 4: Best Methods Comparison\n(100 runs)', 'part4_comparison.png',
                {best_c: best_c, 'Best ε-greedy': r'$\epsilon$-greedy ($\alpha=\frac{1}{1+\ln(1+k)}$)',
                 'Gradient-Bandit': r'Gradient-Bandit ($\alpha=0.1$)'})
    
    # Tables
    print_table([(k, v[-1]) for k, v in ucb_results.items()], ('UCB Config', 'Final AccR̄'), 'part4_ucb_table.txt')
    print_table([(k, v[-1]) for k, v in comparison.items()], ('Method', 'Final AccR̄'), 'part4_comparison_table.txt')
    
    return ucb_results, comparison



# MAIN


def main():
    print("=" * 70)
    print("EECE 5614 - Project 1: 2-Armed Bandit Problem")
    print("=" * 70)
    print(f"\nSettings: {N_STEPS} steps, {N_RUNS} runs")
    print(f"Output: {OUTPUT_DIR}/")
    
    run_part1()
    run_part2()
    run_part3()
    run_part4()
    
    print("\n" + "=" * 70)
    print(f"Done! All results saved to '{OUTPUT_DIR}/'")
    print("=" * 70)


if __name__ == "__main__":
    main()
