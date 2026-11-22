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
- Taken on 2026-11-22 after the first headroom measurement ([research.md](research.md), D6-D8): regimes read by family, with R3d kept as a regime of damage without mortality (D6); reading rule for `Ccum`, failure and success rates first (D7); 60 paired seeds, 5,000 bootstrap replications, one test-seed opening recorded in a ledger, equal tuning budget per agent (D8, [protocol.md](protocol.md)).
- DEV-1 ([protocol_deviations.md](protocol_deviations.md)): route rule with a safety margin (`margin_frac` 0.7), fixed before any contact with the test seeds, applied to every compared agent. The exact run-1 result (D1 not met, R3c -0.8%) is archived in `archive/2026-11-22_run1_margin1.0/`. Frozen is identical under 0.7 and 1.0 (12 runs).
- Code is authoritative over earlier notes; everything written is in English.

## Open decisions

1. **Tuning budget for the compared agents** ([protocol.md](protocol.md), section 3.3): 6 candidates per agent, validation seeds
   2030-2039, regimes R0, R1, R2c, R3c, 20 episodes (240 runs per agent). To confirm or change.
2. **Reading of checklist 3.1**: `Ccum` over the complete episode (no window), plus run-level totals. To confirm.
3. **D1 on 60 validation seeds**: to run on a machine with more cores ([agents.md](agents.md), section 11). Until then D1 is
   met on 6 seeds only.

Still to define when the agents exist: the alert threshold of each trigger (tuned on validation seeds),
the identification measure of prediction D, and the time-to-goal metric of prediction E over several episodes.

## Known limits

- 20 episodes is a prototype. The final study needs more episodes, pre-training before the run, and proportional drift timing; the regimes (checklist 1.3) must be re-verified at the final size before values are frozen.
- R3l is not a masked drift with the frozen agent.
- The toy world supports conclusions on trigger logic, cause attribution and margin measurement only. Margin size and trigger ranking need standard environments (checklist 6.2).

## Next

1. Checklist 2.3: confirm D1 on the 60 validation seeds (6-seed result under DEV-1: met, R3c +58%, in [agents.md](agents.md), section 11).
2. Build Agent 1 (perceptual alert) and Agent 2 (prediction error, both predicting `n`).
3. Stage 1 of the checklist; then validation beyond the toy world.

Reading list before agents: stress-gated plasticity (arXiv 2602.18581), the viability-variable paper
(bioRxiv), arXiv 2505.14903.
