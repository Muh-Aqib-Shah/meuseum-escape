"""
Mansion Heist — a custom Gymnasium environment.

A robber starts at a fixed entrance on an N x N grid. One guard patrols each
column, moving straight up and down between row 0 and row N-1 and reversing
direction at the ends. Adjacent columns are phase-offset by half a cycle, so
neighboring guards always move in opposite directions.

The robber must reach a (fixed, randomly-placed-at-build-time) loot cell,
then reach a (fixed, randomly-placed-at-build-time) exit cell while carrying
the loot — all while avoiding the guards.

State (fully observable, Markov):
    (robber_row, robber_col, phase, carrying_loot)
    - robber_row, robber_col : 0..N-1
    - phase                  : 0..(2*(N-1)-1), a single clock value that
                                determines every guard's row and direction
    - carrying_loot          : 0 or 1

Actions: 0=up, 1=down, 2=left, 3=right
    - Moving off the grid is invalid: the robber simply stays in place
      (still incurs the step penalty).
    - With probability `slip_prob` (default 0.2 — "bad footing / dim
      lighting" in the mansion), the action is replaced by a uniformly
      random OTHER action before being applied. This models an unreliable
      environment so p(s', r | s, a) is nontrivial to estimate, and gives
      PI/VI a genuine risk-vs-reward tradeoff to solve near guards.

Reward:
    - -0.1 per step (efficiency pressure)
    - +10 for reaching the loot cell for the first time (sets carrying=1)
    - +20 for reaching the exit cell while carrying the loot (terminal, success)
    - -20 for being caught by a guard (terminal, failure)

Episode ends on: success, caught, or `max_steps` reached (truncation).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MansionHeistEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=6, slip_prob=0.2, max_steps=100,
                 entrance=(0, 0), loot_pos=None, exit_pos=None,
                 seed=None, render_mode=None):
        super().__init__()
        self.size = size
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.entrance = entrance
        self.render_mode = render_mode

        self.cycle_length = 2 * (size - 1)  # e.g. 10 for size=6

        # Fix loot/exit once per environment INSTANCE (not per episode).
        rng = np.random.default_rng(seed)
        all_cells = [(r, c) for r in range(size) for c in range(size)
                     if (r, c) != entrance]

        if loot_pos is None:
            loot_pos = tuple(all_cells[rng.integers(len(all_cells))])
        if exit_pos is None:
            
            remaining = [c for c in all_cells
                         if c != loot_pos and c[1] != loot_pos[1]]
            exit_pos = tuple(remaining[rng.integers(len(remaining))])
        self.loot_pos = loot_pos
        self.exit_pos = exit_pos

        # 4 discrete moves
        self.action_space = spaces.Discrete(4)
        self._deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up,down,left,right

        # Flattened discrete observation for tabular PI/VI.
        self.n_states = size * size * self.cycle_length * 2
        self.observation_space = spaces.Discrete(self.n_states)

        self._np_random = np.random.default_rng(seed)

        # Pick a starting phase under which the entrance itself is safe
        # (not coinciding with that column's guard catch-zone). Phase 0
        # isn't guaranteed safe for an arbitrary entrance, and a state that
        # is simultaneously "just reset" and "already caught" breaks the
        # Markov property -- the same raw state can't mean two things.
        self.start_phase = 0
        for p in range(self.cycle_length):
            if not self._is_caught(entrance[0], entrance[1], p):
                self.start_phase = p
                break

        self.robber_pos = None
        self.phase = self.start_phase
        self.carrying = 0
        self.t = 0

    def encode(self, row, col, phase, carrying):
        return ((row * self.size + col) * self.cycle_length + phase) * 2 + carrying

    def decode(self, s):
        carrying = s % 2
        s //= 2
        phase = s % self.cycle_length
        s //= self.cycle_length
        col = s % self.size
        row = s // self.size
        return row, col, phase, carrying

    def _obs(self):
        r, c = self.robber_pos
        return self.encode(r, c, self.phase, self.carrying)

    def guard_row_and_dir(self, phase, col):
        """Row and direction (+1=down, -1=up) of the guard in `col` at `phase`."""
        # Odd columns are offset by half a cycle so neighbors move oppositely.
        offset = (self.cycle_length // 2) if (col % 2 == 1) else 0
        p = (phase + offset) % self.cycle_length
        n = self.size - 1
        if p <= n:
            row, direction = p, 1          # heading down (0 -> n)
        else:
            row, direction = self.cycle_length - p, -1  # heading up (n -> 0)
        return row, direction

    def _is_caught(self, row, col, phase):
        g_row, g_dir = self.guard_row_and_dir(phase, col)
        # caught if on the guard's cell, or exactly 1 cell ahead of its travel direction
        return row == g_row or row == g_row + g_dir

    def is_terminal(self, state_index):
        """True if this state index is an absorbing terminal state: the
        robber has been caught there, or has reached the exit with the loot.
        Used so PI/VI don't bootstrap future value past a real episode end."""
        row, col, phase, carrying = self.decode(state_index)
        caught = self._is_caught(row, col, phase)
        success = (carrying == 1 and (row, col) == self.exit_pos)
        return caught or success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # "Exploring starts": if options={"state": (row, col, phase, carrying)}
        # is given, start from that state instead of the fixed entrance. Used
        # during dynamics estimation to guarantee coverage of the full
        # state-action space (a random policy started only at the entrance
        # rarely reaches the "carrying_loot=1" half of the state space).
        if options is not None and "state" in options:
            row, col, phase, carrying = options["state"]
            self.robber_pos = (row, col)
            self.phase = phase
            self.carrying = carrying
        else:
            self.robber_pos = self.entrance
            self.phase = self.start_phase
            self.carrying = 0
        self.t = 0
        info = {"loot_pos": self.loot_pos, "exit_pos": self.exit_pos}
        return self._obs(), info

    def step(self, action):
        assert self.action_space.contains(action)

        # Apply slip: with slip_prob, swap in a random OTHER action.
        if self._np_random.random() < self.slip_prob:
            other_actions = [a for a in range(4) if a != action]
            action = other_actions[self._np_random.integers(len(other_actions))]

        dr, dc = self._deltas[action]
        r, c = self.robber_pos
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            nr, nc = r, c  # invalid move: stay in place

        # Advance the world clock; guards move every step regardless of the robber's action.
        self.phase = (self.phase + 1) % self.cycle_length
        self.t += 1

        self.robber_pos = (nr, nc)

        reward = -0.1
        terminated = False
        truncated = False

        if self._is_caught(nr, nc, self.phase):
            reward = -20.0
            terminated = True
        else:
            if not self.carrying and self.robber_pos == self.loot_pos:
                self.carrying = 1
                reward += 10.0
            elif self.carrying and self.robber_pos == self.exit_pos:
                reward += 20.0
                terminated = True

        if not terminated and self.t >= self.max_steps:
            truncated = True

        info = {"loot_pos": self.loot_pos, "exit_pos": self.exit_pos}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(5, 5))
        n = self.size
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.grid(True)
        ax.set_aspect("equal")

        for col in range(n):
            g_row, g_dir = self.guard_row_and_dir(self.phase, col)
            y = n - 1 - g_row
            ax.add_patch(patches.Rectangle((col, y), 1, 1, color="red", alpha=0.6))
            ax.annotate("^" if g_dir == -1 else "v", (col + 0.5, y + 0.5),
                        ha="center", va="center", color="white", fontsize=12)

        if not self.carrying:
            lr, lc = self.loot_pos
            ax.add_patch(patches.Circle((lc + 0.5, n - 1 - lr + 0.5), 0.25, color="gold"))

        er, ec = self.exit_pos
        ax.add_patch(patches.Rectangle((ec, n - 1 - er), 1, 1, color="green", alpha=0.4))

        rr, rc = self.robber_pos
        ax.add_patch(patches.Circle((rc + 0.5, n - 1 - rr + 0.5), 0.3, color="blue"))

        ax.set_title(f"t={self.t}  phase={self.phase}  carrying={bool(self.carrying)}")
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    env = MansionHeistEnv(size=6, slip_prob=0.1, seed=42)
    obs, info = env.reset(seed=42)
    print("loot:", info["loot_pos"], "exit:", info["exit_pos"])
    print("n_states:", env.n_states)
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(a, obs, r, term, trunc)
        if term or trunc:
            break