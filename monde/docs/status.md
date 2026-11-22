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
| 2.2 three references: frozen, oracle, always-adapting | `toyworld/adaptive.py`, `revision.py`, `model.py::finetune`; protocol and budget in [agents.md](agents.md), section 9 |
| 2.4 common trigger interface with the never and always triggers | `toyworld/triggers.py`, `tests/test_adaptive.py` |

49 tests passing. `D` is calibrated on the baseline and the 8 regimes are verified with it. Remaining in section 2:
the headroom per regime (2.3), measured by `scripts/headroom.py` (see the results section of [agents.md](agents.md)).

## Decisions taken

- Run = 20 chained episodes (prototype); failure ends the episode, not the run; `Ccum` restarts each episode; the drift persists to the end of the run.
- The world model predicts `n` (otherwise the prediction-error agent stops all the time).
- Eight-cell regime grid with the adjusted values of [regimes.md](regimes.md).
- Baseline agent: route A preferred before drift (`lambda = 0.1`); competence before drift = success >= 95%, failure <= 1%, mean `Ccum` <= 1.1 * C1, route A in >= 95% of episodes; world model = PyTorch MLP with two heads, pretrained in the nominal world; planner = CEM-MPC on the waypoints of the chosen route (details in [agents.md](agents.md)).
- `D` recalibrated on the baseline agent: `D = 2 * C1'` = 2.981 (`C1'` = 1.4908), frozen before any trigger experiment. Reason: with the first calibration (5.005, scripted crosser) the competent baseline rarely failed under drift. R3d final amplitude raised from 0.25 to 0.35 for the same agent (harm at the threshold otherwise). Regimes re-verified with the baseline ([regimes.md](regimes.md), section 3).
- Detection delay (prediction C) is counted from `tau`, as in the original definition (details in [research.md](research.md)). The delay from the first exposure to the drift is logged as a secondary metric.
- Five decisions frozen on 2026-11-22, before the headroom measurement ([research.md](research.md), section 6): D1 headroom criterion (at least 15% post-`tau` damage gap, oracle against frozen, in at least 3 harmful regimes, one of them R3d or R3c); D2 revision recipe (fine-tuning on the window of the last episode, no replay, same budget for all adaptive agents); D3 response to an alert (retreat, revise, retry; start side of the wall only); D4 population of prediction C (all post-`tau` episodes, right-censored delays, medians with bootstrap interval); D5 repeated failure (failures after the first one of the run).
- Code is authoritative over earlier notes; everything written is in English.

## Open decisions

1. **D1 was not met on the first measurement** ([agents.md](agents.md), section 10). Options: (a) keep the baseline and
   the criterion as they are, and apply D1 literally (change the question before building triggers); (b) amend the
   baseline's route rule with a safety buffer (`margin_frac` 0.7, chosen a priori, not tuned), declare it as a post hoc
   amendment, keep the first result in the docs, re-measure the six harmful regimes on the same validation seeds, and
   judge the final conclusions on the untouched test seeds only. Exploratory R3c result with the buffer: headroom +58%.
2. **Role of R3d.** Harmful (+15% damage) but never fatal with any agent, so the failure-rate half of prediction B cannot
   be tested there. Proposal: keep it as a damage-only regime and evaluate failure rate on the others.

Still to define when the agents exist: the alert threshold of each trigger (tuned on validation seeds),
the identification measure of prediction D, and the time-to-goal metric of prediction E over several episodes.

## Known limits

- 20 episodes is a prototype. The final study needs more episodes, pre-training before the run, and proportional drift timing; the regimes (checklist 1.3) must be re-verified at the final size before values are frozen.
- R3l is not a masked drift with the frozen agent.
- The toy world supports conclusions on trigger logic, cause attribution and margin measurement only. Margin size and trigger ranking need standard environments (checklist 6.2).

## Next

1. Checklist 2.3: the headroom per regime (oracle against frozen, criterion D1) on the 30 validation seeds; first run on 6 seeds in [agents.md](agents.md), section 10 (D1 not met, see open decisions).
2. Build Agent 1 (perceptual alert) and Agent 2 (prediction error, both predicting `n`).
3. Stage 1 of the checklist; then validation beyond the toy world.

Reading list before agents: stress-gated plasticity (arXiv 2602.18581), the viability-variable paper
(bioRxiv), arXiv 2505.14903.
