# Evaluation protocol (checklist section 3)

Rules fixed before the final study. Code: `toyworld/metrics.py`, `toyworld/seeds.py`, `toyworld/tuning.py`;
tests: `tests/test_protocol.py`.

## 3.1 Metrics fixed in advance

**Main metrics.** Failure rate and cumulative damage `Ccum`. Secondary: return, number of revisions, detection delay,
false alarms (cases without change), recurrence, retention outside the zones.

**`Ccum` is measured over the complete episode** (decision D7): from reset to the end of the episode, whether it
ends in success, failure, timeout or retreat. It is never computed on a window after an alert or a detection. A
window would leave the agent no room to correct its error and explore, and would reward shifting damage out of the
window. For the cost of a whole run, `metrics.run_totals` sums damage, steps, failures, retreats and revisions over the
episodes of a run, so a retreat-and-retry never looks free; `metrics.recovery` gives the episodes, steps and damage from
the drift onset to the first success.

**Outcomes come before `Ccum`** (decision D7). A failure caps `Ccum` at about `D` (2.98) and a timeout truncates it, so an
agent that dies early looks thrifty next to one that barely survives at 2.90. Every table and figure therefore shows
the failure and success rates first, then `Ccum` over the successful episodes next to the all-episode mean
(`metrics.outcome_table`). The all-episode mean alone is never interpreted.

**Regimes are read by family** (decision D6).

| Family | Regimes | Main judge | Also reported |
|---|---|---|---|
| lethal | R2d, R2c, R2l, R3c, R3l | irreversible failure rate | damage conditional on success, survival |
| damage without mortality | R3d, R0, R1 | cumulative damage | revision delay and cost, false alarms |

R3d is kept: it is harmful (+15% damage) but a nominal controller absorbs it (`Ccum` 1.6 against `D` = 2.98), so failure
is zero by construction there. A benchmark does not need every regime to be lethal.

## 3.2 Seeds

60 paired seeds for validation and 60 for test (decision D8, more than 50; the first 30 of each list are unchanged).
Every agent plays the same scenarios (noise seeded per `(seed, episode)`), and the raw per-episode results are
written to disk. Comparisons are paired over seeds.

## 3.3 Tuning separated from evaluation, with an equal budget

- Tuning uses validation seeds only. The test seeds are reached only through `seeds.open_test_seeds`, which records the
  access in `protocol/test_seed_ledger.jsonl` (tracked by git) and refuses a second one. The ledger is empty today, and
  a test checks that no other code refers to the test seeds.
- Every tunable agent (the thresholds of Agents 1 and 2) goes through `tuning.tune` with the same budget:

| Item | Value (identical for every agent) |
|---|---|
| Candidate configurations | 6 |
| Tuning seeds | `VALIDATION_SEEDS[30:40]` (10, apart from the headroom seeds 2000-2005) |
| Regimes | R0, R1 (false alarms), R2c, R3c (lethal) |
| Episodes per run | 20 |
| Runs per agent | 6 x 10 x 4 = 240 (4,800 episodes) |
| Score | `metrics.tuning_score`: failure rate after the drift in the lethal regimes + retreat rate in R0 and R1 |

- `tuning.check_equal_budgets` reads `protocol/tuning_ledger.jsonl` and fails if two agents got different budgets or if
  an agent was tuned twice. A reviewer runs it in one line.
- The revision recipe (fine-tuning budget) is common to all adaptive agents (decision D2). Frozen and oracle have nothing
  to tune.

This budget is a proposal fixed before any trigger exists; change it now or never.

## 3.4 Statistics

Paired bootstrap over seeds, **5,000 replications**, 95% percentile intervals (`metrics.paired_bootstrap`), two-sided
p-values corrected for multiple comparisons by Holm (`metrics.holm`) over the family of comparisons announced in
checklist 0.2. With N > 50 seeds the final study reports intervals on every figure.

## 3.5 Report the failures

The report has a section "Where it does not work", even a short one. R3d (no failure to avoid) and R3l (not maskable
with the frozen agent) are already in it.
