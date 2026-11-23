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
| Tuning seeds | `VALIDATION_SEEDS[30:40]` (10 seeds, 2030-2039) |
| Regimes | R0, R1 (false alarms), R2c, R3c (lethal) |
| Episodes per run | 20 |
| Runs per agent | 6 x 10 x 4 = 240 (4,800 episodes) |
| Score | `metrics.tuning_score`: failure rate after the drift in the lethal regimes + retreat rate in R0 and R1 |

- `tuning.check_equal_budgets` reads `protocol/tuning_ledger.jsonl` and fails if two agents got different budgets or if
  an agent was tuned twice. A reviewer runs it in one line.
- The revision recipe (fine-tuning budget) is common to all adaptive agents (decision D2). Frozen and oracle have nothing
  to tune.

- The 60-seed run of criterion D1 contains the tuning seeds, which is harmless: it involves only the references (frozen,
  oracle, always-adapting), which are never tuned. Any comparison of the tuned Agents 1 and 2 on validation seeds leaves
  the ten tuning seeds out (50 seeds left); the final comparison is on the test seeds.
- The candidate thresholds are not hand-picked: each is defined by the false-alarm rate per nominal episode it would
  have had (1, 2, 5, 10, 20 and 40%), measured on nominal runs of calibration seeds ([agents.md](agents.md), section 12).

This budget was confirmed on 2026-11-22, before any trigger was run.

## 3.4 Statistics

Paired bootstrap over seeds, **5,000 replications**, 95% percentile intervals (`metrics.paired_bootstrap`), two-sided
p-values corrected for multiple comparisons by Holm (`metrics.holm`) over the family of comparisons announced in
checklist 0.2. With N > 50 seeds the final study reports intervals on every figure.

## 3.5 Report the failures

The report has a section "Where it does not work", even a short one. R3d (no failure to avoid) and R3l (not maskable
with the frozen agent) are already in it.

## 3.6 Readings of predictions B and C (decision D9)

D9-STATUS: RECORDED (2026-11-23, 16:31 Paris, on the author's go)

The thresholds of predictions B and C ([research.md](research.md), section 3) are fixed here, on the validation seeds, before
any result of the Agent 1 against Agent 2 comparison was read and before any test seed was opened (the ledger of test seeds is
empty). The comparison run on the validation seeds had started when this section was written; its output had not been looked at.
The final verdict is given on the test seeds, with the same rules. `summary` in `scripts/compare_agents.py` and
`scripts/replay_detectors.py` refuse to run unless the line above is present.

**Prediction B (cumulative damage).** Evaluated on R2c and R3c only, the regimes where the drift acts on the cost alone and the
margin is what the alert is about. In each of them, Agent 1 against Agent 2 (paired over seeds):

1. mean `Ccum` at least 15% lower, as a relative gap, over all post-onset episodes (every outcome included);
2. failure rate at least 10 percentage points lower;
3. guard: the success rate of Agent 1 is not more than 5 points below that of Agent 2 (an agent that retreats all the time lowers
   `Ccum` without doing the task). `Ccum` over the successful episodes is reported next to it;
4. both gaps significant: Holm-adjusted two-sided bootstrap p below 0.05, Holm over the two regimes, separately for the `Ccum` gap and
   the failure gap.

B is supported iff the four conditions hold in both regimes. The other regimes (R2d, R2l, R3l, R3d, R0, R1) are reported with the
same tables and no verdict; they feed the section "Where it does not work". Note recorded in advance: in R3c the frozen agent fails
in 14% of post-onset episodes (D1, 60 seeds), so the failure condition has little room there.

**Prediction C (detection).** The delay is compared at equal false-alarm rates, because the tuned thresholds correspond to about 10%
(Agent 1) and 1% (Agent 2) false alarms per nominal episode, and a detector that fires more easily is faster by construction. The
comparison is a replay on the trajectories of the frozen agent (`toyworld/replay.py`, `scripts/replay_detectors.py`): the same
trajectories for both detectors, at the six grid levels of `protocol/nominal_reference.json` (1, 2, 5, 10, 20, 40% nominal
false-alarm rate, the same rates for both agents). The verdict is read at the **5% level**; the curve at the six levels is reported.

- Regimes: the five lethal regimes R2d, R2c, R2l, R3c, R3l for the delay; R3d is reported, not judged; R0 and R1 for false alarms.
- Delay: one value per run, in environment steps from the start of the onset episode to the first alert, right-censored at the end of
  the run (D4); median over seeds; paired bootstrap interval of the median gap (Agent 2 minus Agent 1).
- A regime counts when: the median delay of Agent 1 is at least 10 steps shorter; the Holm-adjusted p of the gap is below 0.05 (Holm
  over the five regimes); and the median of Agent 1 is not censored (fewer than half of its runs end without an alert). A censored
  median of Agent 2 alone leaves the gap a lower bound and does not disqualify the regime.
- C is supported iff the criterion holds in at least 3 of the 5 lethal regimes, and the measured false-alarm rates of the two
  agents differ by at most 5 points, in R0 and in R1 (the matched rates are measured, not assumed).

**Closed-loop numbers.** The delays and false alarms of the runs at the tuned thresholds (`compare_agents.py summary`) are
reported as descriptive; they do not decide C.

**Aggregation.** `Ccum` and the outcome rates are averaged over all episodes from the onset (R0: all episodes), one value per seed;
seeds are the unit of the paired bootstrap.

## 3.7 Procedure of the final study

Decision D10 ([research.md](research.md)): the confirmatory study runs on the 60 test seeds with the agents, the tuned thresholds, the
revision recipe and the rules of section 3.6 exactly as they were on the validation seeds. `scripts/final_study.py` is the only
script that reaches the test seeds, through `seeds.test_seeds_for_study`.

- **One opening, recorded.** The first `run` (with `--open`) writes one line in `protocol/test_seed_ledger.jsonl`. Its purpose
  contains the short hashes of the model, of `protocol/trigger_thresholds.json` and of `protocol/nominal_reference.json`, so the
  ledger shows what was frozen when the seeds were opened. A relaunch after an interruption resumes the same study, with no second
  line; a changed model, threshold or reference cannot resume (it would be another study, and the seeds are spent).
- **Guards before opening.** D9 and D10 recorded, nominal reference computed on the model, both agents tuned once under one budget.
  `final_study.py plan` shows them without opening anything.
- **No interim look.** `check` prints counts only. `summary` prints nothing until all 60 seeds have every job; the verdicts of B and C
  are read once, on the complete data.
- **What is run.** The closed-loop comparison (1,200 jobs, `results/test_compare.jsonl`) and the frozen-trajectory replay (480 jobs,
  `results/test_replay.jsonl`). The frozen reference of the harmful regimes is the frozen agent of the replay.
- **Reported whatever the result.** The verdicts of B and C, the tables of outcomes and the replay curves are published next to the
  validation record ([results_validation.md](results_validation.md)); the post hoc analyses of that record stay labelled exploratory.
