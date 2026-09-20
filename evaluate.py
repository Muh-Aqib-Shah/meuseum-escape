"""
Full evaluation harness for MansionHeistEnv PI vs VI.

Produces four plots:
  1. pi_vs_vi_comparison.png   - reward per iteration + total running time
  2. convergence_delta.png     - value-function delta per iteration (log scale)
  3. outcome_breakdown.png     - success / caught / timeout rate per iteration
  4. coverage_vs_episodes.png  - state-action coverage vs. estimation episode count
"""

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from mansion_heist_env import MansionHeistEnv
from estimate_dynamics import estimate_dynamics


def run_policy_with_outcomes(env, policy, n_test_episodes=30, seed=1000):
    rng = np.random.default_rng(seed)
    rewards = []
    outcomes = {"success": 0, "caught": 0, "timeout": 0}

    for ep in range(n_test_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        ep_reward = 0.0
        terminated = truncated = False
        last_reward = 0.0
        while not (terminated or truncated):
            a = policy[obs]
            obs, r, terminated, truncated, _ = env.step(a)
            ep_reward += r
            last_reward = r
        rewards.append(ep_reward)

        if truncated:
            outcomes["timeout"] += 1
        elif last_reward > 0:      # only a successful exit gives a positive final reward
            outcomes["success"] += 1
        else:                      # terminated with a negative final reward = caught
            outcomes["caught"] += 1

    avg_reward = float(np.mean(rewards))
    fractions = {k: v / n_test_episodes for k, v in outcomes.items()}
    return avg_reward, fractions


def evaluate_snapshots(env, snapshots, n_test_episodes=30):
    """snapshots: list of (policy, elapsed_time, delta)."""
    avg_rewards, outcome_fracs = [], []
    for policy, elapsed, delta in snapshots:
        avg_r, fracs = run_policy_with_outcomes(env, policy, n_test_episodes=n_test_episodes)
        avg_rewards.append(avg_r)
        outcome_fracs.append(fracs)
    return avg_rewards, outcome_fracs


def plot_reward_and_time(pi_snapshots, vi_snapshots, pi_rewards, vi_rewards):
    pi_times = [t for _, t, _ in pi_snapshots]
    vi_times = [t for _, t, _ in vi_snapshots]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, len(pi_rewards) + 1), pi_rewards, marker="o", label="Policy Iteration")
    axes[0].plot(range(1, len(vi_rewards) + 1), vi_rewards, marker="s", label="Value Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Average test reward (30 episodes)")
    axes[0].set_title("Reward per Iteration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(["Policy Iteration", "Value Iteration"], [pi_times[-1], vi_times[-1]],
                color=["tab:blue", "tab:orange"])
    axes[1].set_ylabel("Total running time (s)")
    axes[1].set_title("Running Time to Convergence")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/pi_vs_vi_comparison.png", dpi=130)
    plt.close(fig)
    print("Saved pi_vs_vi_comparison.png")


def plot_convergence_delta(pi_snapshots, vi_snapshots):
    pi_deltas = [d for _, _, d in pi_snapshots]
    vi_deltas = [d for _, _, d in vi_snapshots]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(pi_deltas) + 1), pi_deltas, marker="o", label="Policy Iteration")
    ax.plot(range(1, len(vi_deltas) + 1), vi_deltas, marker="s", label="Value Iteration")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value-function delta (log scale)")
    ax.set_title("Convergence: Value-Function Delta per Iteration")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/convergence_delta.png", dpi=130)
    plt.close(fig)
    print("Saved convergence_delta.png")


def plot_outcome_breakdown(pi_fracs, vi_fracs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, fracs, title in [(axes[0], pi_fracs, "Policy Iteration"),
                              (axes[1], vi_fracs, "Value Iteration")]:
        iters = range(1, len(fracs) + 1)
        success = [f["success"] for f in fracs]
        caught = [f["caught"] for f in fracs]
        timeout = [f["timeout"] for f in fracs]

        ax.stackplot(iters, success, caught, timeout,
                     labels=["Success", "Caught", "Timeout"],
                     colors=["tab:green", "tab:red", "tab:gray"], alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Fraction of 30 test episodes")
    axes[1].legend(loc="upper right")

    plt.suptitle("Outcome Breakdown per Iteration")
    plt.tight_layout()
    plt.savefig("outputs/outcome_breakdown.png", dpi=130)
    plt.close(fig)
    print("Saved outcome_breakdown.png")


def plot_coverage_vs_episodes(model, episode_counts=(100, 500, 1_000, 2_000, 5_000, 10_000, 20_000)):
    coverages = []
    for n_ep in episode_counts:
        env = MansionHeistEnv(size=model["size"], slip_prob=0.2,
                               loot_pos=model["loot_pos"], exit_pos=model["exit_pos"], seed=0)
        t0 = time.perf_counter()
        _, _, visited = estimate_dynamics(env, n_episodes=n_ep, seed=1, exploring_starts=True,
                                           max_episode_len=20)
        coverage = visited.sum() / visited.size
        coverages.append(coverage)
        print(f"  {n_ep:>7} episodes -> {coverage:.1%} coverage "
              f"({time.perf_counter() - t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(episode_counts, coverages, marker="o", color="tab:purple")
    ax.set_xscale("log")
    ax.set_xlabel("Number of estimation episodes (log scale)")
    ax.set_ylabel("State-action coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Coverage vs. Estimation Episode Count")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/coverage_vs_episodes.png", dpi=130)
    plt.close(fig)
    print("Saved coverage_vs_episodes.png")


if __name__ == "__main__":
    with open("models/pi_vi_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open("models/dynamics_model.pkl", "rb") as f:
        model = pickle.load(f)

    env = MansionHeistEnv(size=model["size"], slip_prob=0.2,
                           loot_pos=model["loot_pos"], exit_pos=model["exit_pos"], seed=123)

    print("Evaluating PI snapshots on the real environment...")
    pi_rewards, pi_fracs = evaluate_snapshots(env, results["pi_snapshots"], n_test_episodes=30)
    print("Evaluating VI snapshots on the real environment...")
    vi_rewards, vi_fracs = evaluate_snapshots(env, results["vi_snapshots"], n_test_episodes=30)

    print(f"\nPI: {len(pi_rewards)} iterations, final avg test reward {pi_rewards[-1]:.2f}")
    print(f"VI: {len(vi_rewards)} iterations, final avg test reward {vi_rewards[-1]:.2f}\n")

    plot_reward_and_time(results["pi_snapshots"], results["vi_snapshots"], pi_rewards, vi_rewards)
    plot_convergence_delta(results["pi_snapshots"], results["vi_snapshots"])
    plot_outcome_breakdown(pi_fracs, vi_fracs)

    print("\nRunning coverage-vs-episodes sweep (this re-runs estimation several times)...")
    plot_coverage_vs_episodes(model)

    print("\nAll plots saved to outputs/")