# Status

## Done (stage 0, section 1 of the checklist: build the world)

| Checklist item | Where |
|---|---|
| 1.1 toy world | `toyworld/world.py`, `toyworld/config.py` |
| 1.2 three drift wrappers | `toyworld/drift.py` |
| 1.3 verify every regime with a frozen agent | `toyworld/regimes.py`, `toyworld/evaluate.py`, `tests/test_regimes.py` |
| 1.4 graded damage, failure at `D` | `toyworld/world.py`, `tests/test_toyworld.py` |
| 1.5 damage-free retreat | `toyworld/policies.py::retreat_action`, retreat tests |
| 1.6 ground-truth journal | `toyworld/journal.py` (`TruthLogger`) |

Also done: paired seeded noise; leak tests; `D` calibrated (`scripts/calibrate.py`); proportional
timing of drift (`n_episodes` scales `tau` and ramp durations).

## Done (stage 0, section 2: references and oracle, in progress)

| Checklist item | Where |
|---|---|
| 2.1 competent baseline agent before drift | `toyworld/model.py`, `data.py`, `agent.py`; spec in [agents.md](agents.md); criteria met on the 30 validation seeds |

38 tests passing. `D` is calibrated on the baseline and the 8 regimes are verified with it. Remaining in section 2: the three references (2.2), the headroom per regime (2.3), the common trigger interface (2.4).

## Decisions taken

- Run = 20 chained episodes (prototype); failure ends the episode, not the run; `Ccum` restarts each episode; the drift persists to the end of the run.
- The world model predicts `n` (otherwise the prediction-error agent stops all the time).
- Eight-cell regime grid with the adjusted values of [regimes.md](regimes.md).
- Baseline agent: route A preferred before drift (`lambda = 0.1`); competence before drift = success >= 95%, failure <= 1%, mean `Ccum` <= 1.1 * C1, route A in >= 95% of episodes; world model = PyTorch MLP with two heads, pretrained in the nominal world; planner = CEM-MPC on the waypoints of the chosen route (details in [agents.md](agents.md)).
- `D` recalibrated on the baseline agent: `D = 2 * C1'` = 2.981 (`C1'` = 1.4908), frozen before any trigger experiment. Reason: with the first calibration (5.005, scripted crosser) the competent baseline rarely failed under drift. R3d final amplitude raised from 0.25 to 0.35 for the same agent (harm at the threshold otherwise). Regimes re-verified with the baseline ([regimes.md](regimes.md), section 3).
- Detection delay (prediction C) is counted from `tau`, as in the original definition (details in [research.md](research.md)). The delay from the first exposure to the drift is logged as a secondary metric.
- Code is authoritative over earlier notes; everything written is in English.

## Open decisions

1. **Population for prediction C.** The prediction compares medians on failure episodes only; the better agent fails less, so its failure episodes are harder (selection bias). To decide: all post-`tau` episodes with censoring, or paired scenarios.
2. **Retreat policy of the trigger agents.** The baseline has no retreat; what Agents 1 and 2 do once the alert fires (retreat, switch to route B, revise the model) is not specified.
3. **Recurrence over a multi-episode run (prediction E).** "Recurrence of failure" is not yet defined precisely.
4. **Retreat validity.** `retreat_action` works on the start side of the wall only; a retreat from the far side is not defined.

## Known limits

- 20 episodes is a prototype. The final study needs more episodes, pre-training before the run, and proportional drift timing; the regimes (checklist 1.3) must be re-verified at the final size before values are frozen.
- R3l is not a masked drift with the frozen agent.
- The toy world supports conclusions on trigger logic, cause attribution and margin measurement only. Margin size and trigger ranking need standard environments (checklist 6.2).

## Next

1. Checklist 2.4 and 2.2: the three references (frozen, oracle, always-adapting), the headroom per regime, the common trigger interface.
2. Checklist 2.3: the headroom per regime (oracle minus frozen), on the 30 validation seeds.
3. Build Agent 1 (perceptual alert) and Agent 2 (prediction error, both predicting `n`).
4. Stage 1 of the checklist; then validation beyond the toy world.

Reading list before agents: stress-gated plasticity (arXiv 2602.18581), the viability-variable paper
(bioRxiv), arXiv 2505.14903.
