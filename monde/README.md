# toyworld — a toy world for testing when a continual world model should revise itself

A continuous, noisy, nonlinear toy world with **hidden drift** and **graded, irreversible damage**.
It is the testbed (checklist, stage 0, section 1) of a research project on *what should trigger the
revision of a world model*. The repository holds the world, the three drift mechanisms, the regime
grid, the ground-truth journal, a competent baseline agent (learned world model plus planner), the
common trigger interface with its limit triggers (never, always), the two agents under study
(Agent 1, a harm-coupled alert; Agent 2, a CUSUM on prediction error), model revision by
fine-tuning, an oracle reference, the pre-registered protocol and its ledgers, the archived data of
the final study, and the tests that verify all of it.

**Status.** The final study on the test seeds is done and its seeds are spent. Predictions B and C
are not supported; the secondary analysis D11 shows that Agent 1 completes the task more often
(a trade-off, not a ranking). See [docs/results_final.md](docs/results_final.md). Any further
analysis of these seeds is exploratory, and a new study needs fresh seeds and a new pre-registration.

## The research in one paragraph

A model-based agent plans with a learned world model. When the environment drifts, the model must
be revised, but *when*? The usual trigger is the model's own prediction error. We study a different
trigger: a **perceptual alert** built from a nociception-like signal (observed harm above a
threshold, coupled to the remaining safety margin). The hypothesis is that when a drift is
**masked in prediction space** (the one-step error stays below the noise) but **accumulates
irreversible damage**, the perceptual trigger reduces cumulative damage and failure rate more than a
prediction-error trigger. Details: [docs/research.md](docs/research.md).

## Install and test

Use a dedicated virtual environment: the scripts need PyTorch, gymnasium and numpy in the interpreter that runs them.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # a launcher can use another interpreter: PYTHON=/path/to/python bash scripts/run_d1_60.sh 8
    python -m pytest -q                        # 131 tests, about 2 min
    PYTHONPATH=. python scripts/calibrate.py   # measures C1 and D = 2*C1
    PYTHONPATH=. python scripts/explore_regimes.py 12   # scripted agent, 8 regimes, before/after drift
    PYTHONPATH=. python scripts/train_baseline.py       # pretrains the world model (about 1 min)
    PYTHONPATH=. python scripts/eval_baseline.py        # competence check before drift (about 5 min)
    PYTHONPATH=. python scripts/calibrate.py results/baseline.pt 4 2          # C1' and D with the baseline (2 workers)
    PYTHONPATH=. python scripts/verify_regimes_baseline.py results/baseline.pt 6 2   # 8 regimes with the baseline (about 45 min)
    PYTHONPATH=. python scripts/tune_revision.py results/baseline.pt 2        # revision budget, calibration seeds (about 6 min)
    PYTHONPATH=. python scripts/headroom.py run results/baseline.pt 6 2      # oracle and always-adapting, harmful regimes (about 35 min)
    PYTHONPATH=. python scripts/headroom.py summary                          # headroom per regime and criterion D1
    bash scripts/run_d1_60.sh 8                                              # D1 on the 60 validation seeds under DEV-1 (resumable, about 2-3 h on 8 cores)
    PYTHONPATH=. python scripts/calibrate_triggers.py results/baseline.pt 2  # nominal reference of Agents 1 and 2 (about 6 min)
    PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/tune_triggers.py results/baseline.pt 8   # thresholds of Agents 1 and 2, equal budget (480 runs)
    PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/compare_agents.py run results/baseline.pt 8    # Agent 1 against Agent 2, 50 validation seeds
    PYTHONPATH=. python scripts/compare_agents.py summary                    # outcomes first, paired contrasts, verdict of B (after D9)
    PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/replay_detectors.py run results/baseline.pt 8   # frozen trajectories, both detectors at six false-alarm levels (400 runs)
    PYTHONPATH=. python scripts/replay_detectors.py check                    # the replay reproduces the closed-loop runs
    PYTHONPATH=. python scripts/replay_detectors.py summary                  # delay at matched false-alarm rates, verdict of C (after D9)
    PYTHONPATH=. python scripts/final_study.py plan results/baseline.pt      # final study on the test seeds: guards, nothing opened
    PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/final_study.py run results/baseline.pt 8 --open   # opens the test seeds ONCE (ledger), then runs
    PYTHONPATH=. python scripts/final_study.py summary                       # verdicts of B and C, only when all 60 seeds are complete
    PYTHONPATH=. python scripts/secondary_outcomes.py results/test_compare.jsonl --label test --required 60   # secondary analysis D11: non-completion, after the summaries
    PYTHONPATH=. python scripts/make_figures.py --compare results/test_compare.jsonl --replay results/test_replay.jsonl --out figures/test --label test --required 60   # figures, same guard

## Use

    from toyworld import make_env

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
| `toyworld/config.py` | frozen world parameters (`WorldConfig`) |
| `toyworld/world.py` | the Gymnasium environment `ToyWorld` |
| `toyworld/drift.py` | three drift wrappers: `DynamicsDrift`, `CostDrift`, `LatencyDrift` |
| `toyworld/regimes.py` | the 8-cell regime grid and `make_env` |
| `toyworld/policies.py` | scripted crosser (frozen reference agent) and damage-free retreat |
| `toyworld/evaluate.py` | `run_frozen`: runs the frozen agent and collects ground-truth residuals |
| `toyworld/journal.py` | `TruthLogger`: per-episode ground-truth journal |
| `toyworld/seeds.py` | frozen seed lists: calibration, train, validation, test |
| `toyworld/model.py` | `WorldModel` (PyTorch MLP, two heads) and `AnalyticModel` (true equations, oracle) |
| `toyworld/data.py` | pretraining data collected in the nominal world |
| `toyworld/agent.py` | `BaselineAgent`: route choice + CEM model-predictive control |
| `toyworld/triggers.py` | trigger interface: `StepRecord`, `Trigger`, `NeverTrigger`, `AlwaysTrigger` |
| `toyworld/detectors.py` | Agent 1 (`HarmExcessTrigger`, harm excess over remaining margin) and Agent 2 (`CusumTrigger`, prediction error), `Recorder` |
| `toyworld/compare.py` | comparison of Agents 1 and 2: seeds, guarded thresholds, per-seed vectors, contrasts, verdict of B, D9 guard |
| `scripts/final_study.py` | the final study on the test seeds: guards, one recorded opening, resumable run, verdicts without interim look |
| `scripts/secondary_outcomes.py`, `toyworld/secondary.py` | secondary analysis (decision D11): non-completion `1 - success` per regime, same bootstrap, no verdict |
| `scripts/make_figures.py` | six figures (PNG and PDF) from the comparison and replay files; refuses to draw before the required seeds are complete |
| `toyworld/replay.py` | matched false-alarm replay of the two detectors on frozen trajectories, verdict of C |
| `toyworld/nominal.py` | nominal reference: residual scales and threshold grids by false-alarm rate |
| `toyworld/revision.py` | revision by fine-tuning on the window of the last episode (`RevisionParams`) |
| `toyworld/adaptive.py` | `run_agent`: one agent, one trigger (or the oracle), one full run; retreat on alert |
| `toyworld/metrics.py` | evaluation rules: outcome table (failure first), `Ccum` over successes, run totals, paired bootstrap, Holm |
| `toyworld/tuning.py` | `tune`: the same tuning budget for every tunable agent, ledger and equality check |
| `protocol/` | test-seed ledger (one line: the opening of the final study) and tuning ledger, tracked by git |
| `archive/` | exact data of archived runs: the run that motivated a protocol deviation, the validation comparison, the final study |
| `tests/` | 131 tests: dynamics, geometry, damage, retreat, leaks, every regime, world model, agent, triggers, revision, oracle, seeds, metrics, protocol, Agents 1 and 2, comparison, replay, secondary analysis |
| `scripts/` | calibration of `D`, exploration of the regimes, baseline training and evaluation, revision budget, headroom |
| `docs/` | research design, world specification, regimes and results, status |

## Documentation

1. [docs/research.md](docs/research.md): question, predictions A to E with stopping rules, agents, methodology.
2. [docs/world_spec.md](docs/world_spec.md): equations, symbols, parameters.
3. [docs/regimes.md](docs/regimes.md): the 8-regime grid, adjusted values, measured verification results.
4. [docs/agents.md](docs/agents.md): the baseline agent, the references (frozen, oracle, always-adapting), the trigger interface and the revision protocol.
5. [docs/protocol.md](docs/protocol.md): evaluation rules (checklist section 3): metrics, seeds, tuning budget, statistics.
6. [docs/protocol_deviations.md](docs/protocol_deviations.md): dated log of changes made to the protocol after a result was seen.
7. [docs/results_validation.md](docs/results_validation.md): Agent 1 against Agent 2 on the validation seeds (predictions B and C).
8. [docs/results_final.md](docs/results_final.md): the final study on the test seeds (predictions B and C), with the secondary analysis D11.
9. [docs/status.md](docs/status.md): what is done, open decisions, next steps, checklist mapping.
