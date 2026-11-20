# Status

## Done (stage 0, section 1 of the checklist: build the world)

| Checklist item | Where |
|---|---|
| 1.1 toy world | `monde/world.py`, `monde/config.py` |
| 1.2 three drift wrappers | `monde/drift.py` |
| 1.3 verify every regime with a frozen agent | `monde/regimes.py`, `monde/evaluate.py`, `tests/test_regimes.py` |
| 1.4 graded damage, failure at `D` | `monde/world.py`, `tests/test_monde.py` |
| 1.5 damage-free retreat | `monde/policies.py::retreat_action`, retreat tests |
| 1.6 ground-truth journal | `monde/journal.py` (`TruthLogger`) |

Also done: paired seeded noise; leak tests; `D` calibrated (`scripts/calibrate.py`); proportional
timing of drift (`n_episodes` scales `tau` and ramp durations); 28 tests passing.

## Decisions taken

- Run = 20 chained episodes (prototype); failure ends the episode, not the run; `Ccum` restarts each episode; the drift persists to the end of the run.
- The world model predicts `n` (otherwise the prediction-error agent stops all the time).
- Eight-cell regime grid with the adjusted values of [regimes.md](regimes.md).
- Code is authoritative over earlier notes; everything written is in English.

## Open decisions

1. **Origin of the detection delay (prediction C).** Counting from `tau` mixes detection and travel time to the zone; counting from the first entry into `k*` after `tau` is fairer. To decide and justify.
2. **Reward and agent specification.** The world returns reward 0. The goal-under-constraint objective, the planner and the retreat policy of the agents are not specified.
3. **Recurrence over a multi-episode run (prediction E).** "Recurrence of failure" is not yet defined precisely.
4. **Retreat validity.** `retreat_action` works on the start side of the wall only; a retreat from the far side is not defined.

## Known limits

- 20 episodes is a prototype. The final study needs more episodes, pre-training before the run, and proportional drift timing; the regimes (checklist 1.3) must be re-verified at the final size before values are frozen.
- R3l is not a masked drift with the frozen agent.
- The toy world supports conclusions on trigger logic, cause attribution and margin measurement only. Margin size and trigger ranking need standard environments (checklist 6.2).

## Next

1. Decide the open points above, freeze values and thresholds (pre-registration).
2. Measure headroom with learning agents before building the full comparison.
3. Build Agent 1 (perceptual alert) and Agent 2 (prediction error, both predicting `n`).
4. Stage 1 of the checklist; then validation beyond the toy world.

Reading list before agents: stress-gated plasticity (arXiv 2602.18581), the viability-variable paper
(bioRxiv), arXiv 2505.14903.
