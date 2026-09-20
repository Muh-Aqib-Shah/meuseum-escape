"""
Estimate the transition dynamics p(s', r | s, a) for MansionHeistEnv by
running a random policy for many episodes and averaging observed outcomes,
as required by the assignment.

For each (state, action) pair we build:
    - a distribution over resulting next-states: P[s][a][s'] = count / total
    - the average reward observed:               R[s][a] = mean(reward)

These are saved to disk so Policy Iteration and Value Iteration can be run
purely from the estimated model (no direct calls to env.step() needed
during planning).
"""

import numpy as np
import pickle
from collections import defaultdict
from mansion_heist_env import MansionHeistEnv


def estimate_dynamics(env, n_episodes=200_000, seed=0, exploring_starts=True,
                       max_episode_len=20):
    n_states = env.n_states
    n_actions = env.action_space.n

    # counts[s][a][s'] -> number of times this transition was observed
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # reward_sum[s][a] -> running sum of rewards observed for (s, a)
    reward_sum = defaultdict(lambda: defaultdict(float))
    # sa_count[s][a] -> number of times (s, a) was tried at all
    sa_count = defaultdict(lambda: defaultdict(int))

    rng = np.random.default_rng(seed)
    ep = 0
    while ep < n_episodes:
        if exploring_starts:
            # Start from a uniformly random NON-terminal state so every
            # (s, a) pair gets a real chance of being sampled, not just
            # states reachable from the entrance under a fully random
            # policy. Terminal states (caught / already-succeeded) are
            # never live states a real trajectory would resume from, so
            # they're excluded as starting points.
            while True:
                row = int(rng.integers(env.size))
                col = int(rng.integers(env.size))
                phase = int(rng.integers(env.cycle_length))
                carrying = int(rng.integers(2))
                s0 = env.encode(row, col, phase, carrying)
                if not env.is_terminal(s0):
                    break
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)),
                                options={"state": (row, col, phase, carrying)})
        else:
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)))

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < max_episode_len:
            a = env.action_space.sample()
            next_obs, r, terminated, truncated, _ = env.step(a)

            counts[obs][a][next_obs] += 1
            reward_sum[obs][a] += r
            sa_count[obs][a] += 1

            obs = next_obs
            steps += 1
        ep += 1
        if ep % 50_000 == 0:
            print(f"  {ep}/{n_episodes} episodes")

    P = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R = np.zeros((n_states, n_actions), dtype=np.float64)
    visited = np.zeros((n_states, n_actions), dtype=bool)

    for s in counts:
        for a in counts[s]:
            total = sa_count[s][a]
            if total == 0:
                continue
            visited[s, a] = True
            R[s, a] = reward_sum[s][a] / total
            for s_next, c in counts[s][a].items():
                P[s, a, s_next] = c / total

    # Fallback for (s, a) pairs never observed under the random policy:
    # assume a self-loop with the ordinary step penalty. This keeps PI/VI
    # well-defined everywhere without pretending we have real data for
    # states a random walk rarely reaches (e.g. far corners reached only
    # under specific phases before getting caught).
    unvisited = ~visited
    for s in range(n_states):
        for a in range(n_actions):
            if unvisited[s, a]:
                P[s, a, s] = 1.0
                R[s, a] = -0.1

    # SAFEGUARD: force every terminal state (caught, or successful exit) to
    # be a zero-reward absorbing state, regardless of what data was
    # collected for it. This is what makes PI/VI stop "bootstrapping" future
    # reward past a real episode boundary -- the -20 / +20 reward has
    # already been credited on the TRANSITION INTO this state; nothing
    # further should accrue from being "in" it.
    for s in range(n_states):
        if env.is_terminal(s):
            P[s, :, :] = 0.0
            P[s, :, s] = 1.0
            R[s, :] = 0.0

    return P, R, visited


if __name__ == "__main__":
    env = MansionHeistEnv(size=6, slip_prob=0.2, seed=42)
    print("loot:", env.loot_pos, "exit:", env.exit_pos, "n_states:", env.n_states)

    P, R, visited = estimate_dynamics(env, n_episodes=400_000, seed=0)

    coverage = visited.sum() / visited.size
    print(f"State-action coverage: {coverage:.1%} "
          f"({visited.sum()} / {visited.size} pairs visited at least once)")

    with open("models/dynamics_model.pkl", "wb") as f:
        pickle.dump({
            "P": P, "R": R, "visited": visited,
            "n_states": env.n_states, "n_actions": env.action_space.n,
            "loot_pos": env.loot_pos, "exit_pos": env.exit_pos,
            "size": env.size, "cycle_length": env.cycle_length,
        }, f)
    print("Saved model to dynamics_model.pkl")