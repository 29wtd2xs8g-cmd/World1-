# toyworld — a toy world for testing when a continual world model should revise itself

A continuous, noisy, nonlinear toy world with **hidden drift** and **graded, irreversible damage**.
It is the testbed (checklist, stage 0, section 1) of a research project on *what should trigger the
revision of a world model*. The agents under study are not in this repository yet: this repository
contains the world, the three drift mechanisms, the regime grid, the ground-truth journal and the
tests that verify them.

## The research in one paragraph

A model-based agent plans with a learned world model. When the environment drifts, the model must
be revised, but *when*? The usual trigger is the model's own prediction error. We study a different
trigger: a **perceptual alert** built from a nociception-like signal (observed harm above a
threshold, coupled to the remaining safety margin). The hypothesis is that when a drift is
**masked in prediction space** (the one-step error stays below the noise) but **accumulates
irreversible damage**, the perceptual trigger reduces cumulative damage and failure rate more than a
prediction-error trigger. Details: [docs/research.md](docs/research.md).

## Install and test

    pip install -r requirements.txt
    python -m pytest -q                        # 28 tests, about 20 s
    PYTHONPATH=. python scripts/calibrate.py   # measures C1 and D = 2*C1
    PYTHONPATH=. python scripts/explore_regimes.py 12   # frozen agent, 8 regimes, before/after drift

## Use

    from World1 import make_env

    env = make_env("R2c", seed=0, n_episodes=20)     # R0, R1, R2d, R2c, R2l, R3d, R3c, R3l
    obs, info = env.reset(seed=0, options={"new_run": True})
    obs, reward, terminated, truncated, info = env.step(action)

- `obs = (x, y, vx, vy, n)`: noisy position, noisy velocity, noisy nociceptive signal.
- `action` in `[-1, 1]^2`: normalised thrust. A hidden command delay `d` may apply.
- `reward` is always 0: the specification defines no reward (goal under a damage constraint).
- All ground truth is in `info["truth"]` (true state, damage, margin, hidden parameters,
  standardised one-step residual of the frozen nominal model, outcome) and `info["drift"]`
  (nature, cause zone k*, onset). **It must never enter an agent's input**; leak tests enforce this.
- A *run* chains `n_episodes` episodes. Each `reset()` starts the next episode; the damage `Ccum`
  restarts at 0 each episode; the drift persists until the end of the run.
  `reset(options={"new_run": True})` starts a new run.
- Scenarios are paired: the noise depends only on `(seed, episode)`, so two agents in the same run
  face exactly the same noise.

## Layout

| Path | Content |
|---|---|
| `World1/config.py` | frozen world parameters (`WorldConfig`) |
| `World1/world.py` | the Gymnasium environment `ToyWorld` |
| `World1/drift.py` | three drift wrappers: `DynamicsDrift`, `CostDrift`, `LatencyDrift` |
| `World1/regimes.py` | the 8-cell regime grid and `make_env` |
| `World1/policies.py` | scripted crosser (frozen reference agent) and damage-free retreat |
| `World1/evaluate.py` | `run_frozen`: runs the frozen agent and collects ground-truth residuals |
| `World1/journal.py` | `TruthLogger`: per-episode ground-truth journal |
| `tests/` | 28 tests: dynamics, geometry, damage, retreat, leaks, every regime |
| `scripts/` | calibration of `D`, exploration of the regimes |
| `docs/` | research design, world specification, regimes and results, status |

## Documentation

1. [docs/research.md](docs/research.md): question, predictions A to E with stopping rules, agents, methodology.
2. [docs/world_spec.md](docs/world_spec.md): equations, symbols, parameters.
3. [docs/regimes.md](docs/regimes.md): the 8-regime grid, adjusted values, measured verification results.
4. [docs/status.md](docs/status.md): what is done, open decisions, next steps, checklist mapping.
