"""
Policy Iteration (PI) and Value Iteration (VI) for MansionHeistEnv, operating
on the transition/reward tables estimated in estimate_dynamics.py.

Both return, per outer iteration:
    - the value function / policy at that point
    - wall-clock time spent (excluding any test-episode evaluation)
so that evaluate.py can plot reward-per-iteration and compare running time,
as required by the assignment.
"""

import numpy as np
import pickle
import time


def load_model(path="models/dynamics_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def policy_iteration(P, R, gamma=0.95, theta=1e-6, max_eval_iters=1000):
    n_states, n_actions, _ = P.shape
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)

    snapshots = []  # (policy_copy, elapsed_time_so_far, final_eval_delta)
    t0 = time.perf_counter()

    stable = False
    iteration = 0
    while not stable:
        iteration += 1
        # --- Policy evaluation ---
        delta = 0.0
        for _ in range(max_eval_iters):
            delta = 0.0
            for s in range(n_states):
                a = policy[s]
                v_new = R[s, a] + gamma * np.dot(P[s, a], V)
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break

        # --- Policy improvement ---
        stable = True
        for s in range(n_states):
            old_a = policy[s]
            q_sa = R[s] + gamma * P[s] @ V           # shape (n_actions,)
            best_a = np.argmax(q_sa)
            policy[s] = best_a
            if best_a != old_a:
                stable = False

        elapsed = time.perf_counter() - t0
        snapshots.append((policy.copy(), elapsed, delta))

    return policy, V, snapshots


def value_iteration(P, R, gamma=0.95, theta=1e-6, max_iters=10_000):
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)

    snapshots = []  # (policy_copy, elapsed_time_so_far, delta)
    t0 = time.perf_counter()

    for iteration in range(1, max_iters + 1):
        delta = 0.0
        for s in range(n_states):
            q_sa = R[s] + gamma * P[s] @ V
            v_new = np.max(q_sa)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        # Derive greedy policy at this iteration for evaluation/plotting.
        policy = np.array([np.argmax(R[s] + gamma * P[s] @ V) for s in range(n_states)])
        elapsed = time.perf_counter() - t0
        snapshots.append((policy.copy(), elapsed, delta))

        if delta < theta:
            break

    return policy, V, snapshots


if __name__ == "__main__":
    model = load_model()
    P, R = model["P"], model["R"]

    print("Running Policy Iteration...")
    pi_policy, pi_V, pi_snapshots = policy_iteration(P, R)
    print(f"  PI converged in {len(pi_snapshots)} outer iterations, "
          f"total time {pi_snapshots[-1][1]:.3f}s")

    print("Running Value Iteration...")
    vi_policy, vi_V, vi_snapshots = value_iteration(P, R)
    print(f"  VI converged in {len(vi_snapshots)} iterations, "
          f"total time {vi_snapshots[-1][1]:.3f}s")

    agree = np.mean(pi_policy == vi_policy)
    print(f"Policies agree on {agree:.1%} of states")

    with open("models/pi_vi_results.pkl", "wb") as f:
        pickle.dump({
            "pi_policy": pi_policy, "pi_V": pi_V, "pi_snapshots": pi_snapshots,
            "vi_policy": vi_policy, "vi_V": vi_V, "vi_snapshots": vi_snapshots,
        }, f)
    print("Saved results to pi_vi_results.pkl")