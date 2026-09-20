# Mansion Heist — Policy Iteration vs. Value Iteration

A custom Gymnasium environment and a from-scratch comparison of **Policy
Iteration (PI)** and **Value Iteration (VI)**, built for a Reinforcement
Learning course assignment.

## The problem

A robber starts at a fixed entrance on an `N x N` grid (default `N=6`). One
guard patrols each column, walking straight up and down between row 0 and
row `N-1` and reversing at the ends; neighboring columns are phase-offset by
half a cycle so adjacent guards always move in opposite directions. This
lets every guard's position be recovered from a single scalar **phase**
variable instead of tracking each guard separately.

The robber must reach a fixed loot cell, then reach a fixed exit cell while
carrying the loot, without being caught. Movement is stochastic: with
probability `slip_prob` (default `0.2`), the intended action is replaced by
a uniformly random *other* action ("bad footing / dim lighting" in the
mansion) — guards are always deterministic.

**State:** `(robber_row, robber_col, phase, carrying_loot)` — `6×6×10×2 =
720` states for the default `N=6`.
**Actions:** `0=up, 1=down, 2=left, 3=right`.
**Reward:** `-0.1` per step, `+10` first time loot is picked up, `+20` for
exiting with the loot (terminal, success), `-20` for being caught
(terminal, failure).
**Episode ends on:** success, capture, or `max_steps` reached (truncation).

Full details and design rationale are in the docstring at the top of
`mansion_heist_env.py`.

## Repository structure

| File | Purpose |
|---|---|
| `mansion_heist_env.py` | The Gymnasium `Env` class (`MansionHeistEnv`), plus a smoke test under `if __name__ == "__main__":`. |
| `estimate_dynamics.py` | Estimates `p(s', r \| s, a)` by running many episodes and averaging observed outcomes; saves the model to `models/dynamics_model.pkl`. |
| `pi_vi.py` | Runs Policy Iteration and Value Iteration on the estimated model; saves both algorithms' policies, value functions, and per-iteration snapshots to `models/pi_vi_results.pkl`. |
| `evaluate.py` | Loads both saved model files, re-evaluates each PI/VI snapshot in the real environment, and produces all comparison plots into `outputs/`. |
| `watch_episode.py` | Loads a converged policy and steps through one live episode with a rendered window, for visual sanity-checking. Not required for the assignment deliverables — a convenience script. |

## Requirements

```bash
pip install gymnasium numpy matplotlib
```

Python 3.9+ recommended.

## How to run

Run the pipeline in this order — each script reads the previous one's
output:

```bash
mkdir -p models outputs

# 1. Estimate the transition model from a random policy (takes a while —
#    n_episodes=400,000 by default; reduce this for a quick test run).
python3 estimate_dynamics.py

# 2. Run Policy Iteration and Value Iteration on the estimated model.
python3 pi_vi.py

# 3. Evaluate both algorithms' snapshots on the real environment and
#    generate all comparison plots.
python3 evaluate.py

# 4. (Optional) Watch the converged PI policy play one episode live.
python3 watch_episode.py
```

### Quick smoke test (no full pipeline)

```bash
python3 mansion_heist_env.py
```
Runs a handful of random actions in the environment and prints the
resulting transitions, to confirm the env behaves as expected before
running the full pipeline.

## Outputs

`estimate_dynamics.py` writes `models/dynamics_model.pkl`:
`P` (transition probabilities), `R` (expected reward), `visited` (which
`(s, a)` pairs were actually sampled), plus the environment's fixed layout
(`loot_pos`, `exit_pos`, `size`, `cycle_length`).

`pi_vi.py` writes `models/pi_vi_results.pkl`: each algorithm's final
`policy` and `V`, plus a list of `snapshots` — one `(policy, elapsed_time,
delta)` tuple per iteration — used by `evaluate.py` to plot progress over
time.

`evaluate.py` writes four plots to `outputs/`:

- **`pi_vs_vi_comparison.png`** — average test-episode reward per
  iteration (both algorithms), and total running time to convergence.
- **`convergence_delta.png`** — value-function delta per iteration
  (log scale). Note: PI's delta is the *final inner policy-evaluation
  sweep's* delta at each outer iteration (since PI evaluates to
  convergence every outer step), while VI's delta is the outer-loop delta
  itself — the two lines are not measuring quite the same thing, which is
  worth calling out when discussing this plot.
- **`outcome_breakdown.png`** — fraction of test episodes ending in
  success / caught / timeout, per iteration, for each algorithm.
- **`coverage_vs_episodes.png`** — state-action coverage of the estimated
  model as a function of how many estimation episodes were run (re-runs
  `estimate_dynamics` at several episode counts, so this step takes a
  while).

## Key design notes

- **Guard phase trick**: collapses what would otherwise be an
  intractable `6^6`-scale joint guard-position state space down to a
  single 10-valued phase variable, since guard motion is fully
  deterministic and periodic.
- **Fixed layout**: the loot cell, exit cell, and robber start are sampled
  once when the environment is constructed and held fixed across every
  episode — this is a single-layout planning problem (PI/VI solve one
  specific mansion optimally), not a generalization study.
- **Slip applies only to the robber**, never to guards — this is required
  for the phase trick to remain valid (a stochastic guard would break the
  single-phase-variable state compression).
- **Exploring starts** are used during dynamics estimation
  (`exploring_starts=True` in `estimate_dynamics.py`) so that
  under-visited regions of the state space (e.g. states with
  `carrying_loot=1`, which a purely random walk from the entrance rarely
  reaches) still get sampled.
- **Unvisited `(s, a)` fallback**: any state-action pair never observed
  during estimation defaults to a self-loop with the ordinary `-0.1` step
  reward, so PI/VI remain well-defined everywhere even with imperfect
  coverage.

## Known limitations / things to double check

- `slip_prob` is hardcoded to `0.2` at the top of `estimate_dynamics.py`,
  `pi_vi.py` (via the estimated `R`/`P` it consumes), and `evaluate.py` —
  if you change one, change all three so the real-environment
  re-evaluation stays consistent with the model PI/VI were trained on.
- `pi_vi.py` defaults to `gamma=0.95`; this isn't stored in
  `dynamics_model.pkl`, so if you re-run with a different discount factor,
  keep track of it separately for your report.
- Full-coverage dynamics estimation (`n_episodes=400_000`) is slow; for
  iterating on code changes, temporarily lower `n_episodes` in
  `estimate_dynamics.py`.