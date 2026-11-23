# Final study: Agent 1 against Agent 2 on the test seeds

Recorded on 2026-11-23. Rules: [protocol.md](protocol.md) sections 3.6 and 3.7 (decisions D9 and D10), secondary analysis D11 ([research.md](research.md)).
Raw data: `archive/2026-11-23_final_study/` (`compare.jsonl`: 1,200 closed-loop jobs; `replay.jsonl`: 480 frozen-trajectory replays;
`summary.txt` and `secondary_outcomes.txt`: the printed outputs of `final_study.py summary` and `secondary_outcomes.py`). Everything below is on the **60 test
seeds** (3000 to 3059), opened once: the ledger `protocol/test_seed_ledger.jsonl` holds one line, dated 2026-11-23, naming the model
(`fd1278e20d501d15`), the tuned thresholds (`e68c8961135274d3`) and the nominal reference (`40fa5759d5c251d7`). The agents, the thresholds,
the revision recipe and the rules are those of the validation comparison ([results_validation.md](results_validation.md)); nothing was tuned
or changed (D10). The verdicts were read once, on the complete data; the secondary analysis was run after them, as registered.

## 1. Verdicts under the rules of D9

| Prediction | Verdict on the test seeds |
|---|---|
| **B** (cumulative damage, R2c and R3c) | **not supported**: in R2c Agent 1 has a higher `Ccum` (0.58 against 0.47) and a higher failure rate (3% against 0%); in R3c the two agents do not differ (`Ccum` 0.95 against 0.91, failure 0.00 against 0.01) |
| **C** (detection, at equal false-alarm rate, 5% level) | **not supported**: the delay criterion holds in 1 of the 5 lethal regimes (R3c), and 3 are required; the matched false-alarm clause holds |

These are the verdicts of the validation seeds, unchanged. Predictions A, D and E were not evaluated.

## 2. Prediction B (closed loop, tuned thresholds)

Outcomes (post-onset episodes; R0 and R1 over all episodes). The `frozen` rows of the harmful regimes are the frozen agent of the replay
(the `always` row exists in R0 and R1 only):

```
regime family      kind          | failure success timeout retreat | Ccum|succ Ccum all | revis/run false al/ep
R0     damage_only frozen        |    0.00    1.00    0.00    0.00 |      1.49     1.49 |       0.0       0.000
R0     damage_only agent1_alert  |    0.00    1.00    0.00    0.00 |      1.49     1.49 |       0.9       0.045
R0     damage_only agent2_cusum  |    0.00    1.00    0.00    0.00 |      1.49     1.49 |       0.2       0.008
R0     damage_only always        |    0.00    1.00    0.00    0.00 |      1.49     1.49 |      20.0       0.000
R1     damage_only frozen        |    0.00    1.00    0.00    0.00 |      1.40     1.40 |       0.0       0.000
R1     damage_only agent1_alert  |    0.00    1.00    0.00    0.00 |      1.41     1.41 |       0.8       0.037
R1     damage_only agent2_cusum  |    0.00    1.00    0.00    0.00 |      1.40     1.40 |       0.3       0.013
R1     damage_only always        |    0.00    1.00    0.00    0.00 |      1.43     1.43 |      20.0       0.000
R2d    lethal      frozen        |    0.59    0.41    0.00    0.00 |      2.67     2.87 |       0.0       0.000
R2d    lethal      agent1_alert  |    0.45    0.55    0.00    0.00 |      2.22     2.47 |       2.3       0.063
R2d    lethal      agent2_cusum  |    0.42    0.44    0.00    0.14 |      2.32     2.50 |       5.1       0.011
R2c    lethal      frozen        |    1.00    0.00    0.00    0.00 |       nan     3.12 |       0.0       0.000
R2c    lethal      agent1_alert  |    0.03    0.90    0.00    0.07 |      0.38     0.58 |       1.6       0.063
R2c    lethal      agent2_cusum  |    0.00    0.80    0.00    0.20 |      0.40     0.47 |       4.4       0.011
R2l    lethal      frozen        |    0.79    0.19    0.01    0.00 |      2.35     2.89 |       0.0       0.000
R2l    lethal      agent1_alert  |    0.52    0.39    0.03    0.06 |      1.94     2.45 |       2.5       0.063
R2l    lethal      agent2_cusum  |    0.02    0.03    0.00    0.94 |      1.85     0.50 |      12.4       0.011
R3d    damage_only frozen        |    0.00    1.00    0.00    0.00 |      1.65     1.65 |       0.0       0.000
R3d    damage_only agent1_alert  |    0.00    1.00    0.00    0.00 |      1.63     1.63 |       1.1       0.063
R3d    damage_only agent2_cusum  |    0.00    1.00    0.00    0.00 |      1.64     1.64 |       0.3       0.011
R3c    lethal      frozen        |    0.13    0.87    0.00    0.00 |      2.20     2.31 |       0.0       0.000
R3c    lethal      agent1_alert  |    0.00    0.97    0.00    0.02 |      0.90     0.95 |       3.4       0.063
R3c    lethal      agent2_cusum  |    0.01    0.82    0.00    0.17 |      0.77     0.91 |       5.1       0.011
R3l    lethal      frozen        |    0.50    0.50    0.00    0.00 |      1.80     2.42 |       0.0       0.000
R3l    lethal      agent1_alert  |    0.28    0.70    0.00    0.02 |      1.70     2.08 |       1.9       0.063
R3l    lethal      agent2_cusum  |    0.00    0.22    0.00    0.77 |      1.54     0.78 |      11.8       0.011
```

Paired contrasts, Agent 1 against Agent 2 (positive = Agent 1 better):

```
Agent 1 against Agent 2 (positive = Agent 1 better); Holm over the lethal regimes (failure) and the harmful ones (Ccum)
regime |          failure gap (points)  Holm p |           Ccum gap (relative)  Holm p | success A1 - A2 | B
R2d    |     -0.032 [ -0.100,  +0.038]   0.400 |     +0.011 [ -0.075,  +0.100]   0.810 |          +0.109 | no verdict
R2c    |     -0.032 [ -0.070,  -0.009]   0.000 |     -0.224 [ -0.421,  -0.107]   0.000 |          +0.100 | not met
R2l    |     -0.501 [ -0.569,  -0.434]   0.000 |     -3.911 [ -4.790,  -3.205]   0.000 |          +0.359 | no verdict
R3d    |     +0.000 [ +0.000,  +0.000]     nan |     +0.007 [ -0.001,  +0.014]   0.233 |          +0.002 | no verdict
R3c    |     +0.005 [ -0.002,  +0.012]   0.393 |     -0.042 [ -0.100,  +0.016]   0.295 |          +0.155 | not met
R3l    |     -0.280 [ -0.326,  -0.234]   0.000 |     -1.661 [ -1.952,  -1.410]   0.000 |          +0.474 | no verdict

PREDICTION B (R2c and R3c, Holm over the two; Ccum on all post-onset episodes): NOT SUPPORTED on the validation seeds
  Holm p, failure gap: R2c 0.000  R3c 0.196   Ccum gap: R2c 0.000  R3c 0.148
```

- R2c: `Ccum` is about 22% higher with Agent 1 (0.58 against 0.47; gap -0.224 [-0.421, -0.107]), and failure is 3% against 0% (gap -0.032
  [-0.070, -0.009]). Agent 2 is better on both criteria, so the regime goes the wrong way for B. Agent 1 succeeds in 90% of the episodes
  against 80% (+10.0 points); Agent 2 retreats in 20% against 7%.
- R3c: `Ccum` is 0.95 against 0.91 (gap -0.042 [-0.100, +0.016], Holm p = 0.148 over the two regimes) and failure is 0.00 against 0.01
  (gap +0.005 [-0.002, +0.012], Holm p = 0.196). Neither gap reaches the thresholds of D9 (at least 15% on `Ccum`, at least 10 points on
  failure). Agent 1 succeeds in 97% of the episodes against 82%; Agent 2 retreats in 17% against 2%. The frozen agent fails in 13% of the
  episodes here, so the failure criterion had little room, as noted in D9.
- Agent 1 succeeds more often than Agent 2 in every lethal regime (+10.0 points in R2c to +47.4 in R3l).
- Frozen agent, for reference: failure 0.59 (R2d), 1.00 (R2c), 0.79 (R2l), 0.13 (R3c), 0.50 (R3l).

## 3. Prediction C (replay on frozen trajectories, equal nominal false-alarm rates)

Delay in steps from the start of the onset episode to the first alert, right-censored at the end of the run; gap = Agent 2 median minus
Agent 1 median (positive = Agent 1 faster). `*` marks the verdict level (5%).

```
regime level | median A1  cens | median A2  cens |           gap A2 - A1 [95% CI] |  FA A1  FA A2
R0       1%  |                 |                 |                                |  0.007  0.003
R0       2%  |                 |                 |                                |  0.012  0.013
R0       5%* |                 |                 |                                |  0.035  0.048
R0      10%  |                 |                 |                                |  0.064  0.082
R0      20%  |                 |                 |                                |  0.158  0.159
R0      40%  |                 |                 |                                |  0.349  0.412

R1       1%  |                 |                 |                                |  0.005  0.006
R1       2%  |                 |                 |                                |  0.010  0.022
R1       5%* |                 |                 |                                |  0.024  0.067
R1      10%  |                 |                 |                                |  0.049  0.102
R1      20%  |                 |                 |                                |  0.128  0.222
R1      40%  |                 |                 |                                |  0.304  0.556

R2d      1%  |       149  0.00 |        70  0.00 |       -79.5 [  -89.5,    -8.0] |  0.011  0.007
R2d      2%  |       148  0.00 |        66  0.00 |       -82.0 [  -91.0,   -12.0] |  0.020  0.020
R2d      5%* |       122  0.00 |        60  0.00 |       -62.0 [  -89.5,   -15.5] |  0.037  0.059
R2d     10%  |        86  0.00 |        59  0.00 |       -26.5 [  -87.0,   -15.5] |  0.068  0.081
R2d     20%  |        80  0.00 |        57  0.00 |       -22.5 [  -84.0,   -17.0] |  0.164  0.151
R2d     40%  |        80  0.00 |        51  0.00 |       -28.5 [  -82.0,   -20.5] |  0.333  0.414

R2c      1%  |        47  0.00 |        43  0.00 |        -4.0 [   -5.0,    -3.0] |  0.011  0.007
R2c      2%  |        47  0.00 |        42  0.00 |        -5.0 [   -5.0,    -3.5] |  0.020  0.020
R2c      5%* |        46  0.00 |        42  0.00 |        -4.0 [   -4.0,    -3.0] |  0.037  0.059
R2c     10%  |        45  0.00 |        42  0.00 |        -3.0 [   -4.0,    -2.0] |  0.068  0.081
R2c     20%  |        44  0.00 |        42  0.00 |        -2.5 [   -3.0,    -2.0] |  0.164  0.151
R2c     40%  |        43  0.00 |        42  0.00 |        -1.0 [   -2.0,    -1.0] |  0.333  0.414

R2l      1%  |       270  0.00 |         7  0.00 |      -262.5 [ -294.0,  -137.0] |  0.011  0.007
R2l      2%  |       268  0.00 |         6  0.00 |      -261.5 [ -281.0,  -137.0] |  0.020  0.020
R2l      5%* |       244  0.00 |         6  0.00 |      -238.5 [ -279.0,  -136.5] |  0.037  0.059
R2l     10%  |       232  0.00 |         5  0.00 |      -227.0 [ -276.0,  -135.0] |  0.068  0.081
R2l     20%  |       168  0.00 |         4  0.00 |      -164.5 [ -250.0,  -121.0] |  0.164  0.151
R2l     40%  |       122  0.00 |         3  0.00 |      -119.5 [ -197.0,  -101.0] |  0.333  0.414

R3d      1%  |      1184  0.83 |      1192  0.97 |        +7.0 [   +0.0,  +101.0] |  0.011  0.007
R3d      2%  |      1104  0.73 |      1192  0.90 |       +88.0 [   +1.5,  +187.0] |  0.020  0.020
R3d      5%* |       860  0.33 |      1092  0.63 |      +231.5 [  +44.0,  +549.0] |  0.037  0.059
R3d     10%  |       570  0.22 |       988  0.43 |      +417.0 [  +67.0,  +536.0] |  0.068  0.081
R3d     20%  |       364  0.07 |       516  0.12 |      +152.0 [ -106.0,  +368.5] |  0.164  0.151
R3d     40%  |       166  0.00 |       155  0.00 |       -11.5 [ -110.5,   +38.5] |  0.333  0.414

R3c      1%  |       249  0.00 |       246  0.00 |        -3.0 [   -6.5,   +56.0] |  0.011  0.007
R3c      2%  |       248  0.00 |       244  0.00 |        -3.5 [   -7.0,   +60.5] |  0.020  0.020
R3c      5%* |       162  0.00 |       242  0.00 |       +80.0 [  +43.5,   +83.0] |  0.037  0.059
R3c     10%  |       159  0.00 |       240  0.00 |       +80.5 [   +0.5,   +84.5] |  0.068  0.081
R3c     20%  |       154  0.00 |       156  0.00 |        +2.0 [   -1.5,    +7.0] |  0.164  0.151
R3c     40%  |       145  0.00 |       142  0.00 |        -3.0 [  -83.0,   +75.5] |  0.333  0.414

R3l      1%  |       521  0.00 |        98  0.00 |      -423.5 [ -514.5,  -373.0] |  0.011  0.007
R3l      2%  |       512  0.00 |        64  0.00 |      -448.0 [ -471.5,  -348.0] |  0.020  0.020
R3l      5%* |       382  0.00 |        61  0.00 |      -321.5 [ -416.5,  -303.0] |  0.037  0.059
R3l     10%  |       364  0.00 |        60  0.00 |      -305.0 [ -325.0,  -211.0] |  0.068  0.081
R3l     20%  |       260  0.00 |        56  0.00 |      -204.0 [ -217.5,  -135.0] |  0.164  0.151
R3l     40%  |       162  0.00 |        46  0.00 |      -116.0 [ -132.0,  -102.0] |  0.333  0.414
```

```
Prediction C at the 5% level (*), Holm over the five lethal regimes:
  R2d: not met   (Holm p = 0.000)
  R2c: not met   (Holm p = 0.000)
  R2l: not met   (Holm p = 0.000)
  R3c: MET   (Holm p = 0.016)
  R3l: not met   (Holm p = 0.000)
  regimes met: 1 of 5; measured false-alarm rates matched in R0 and R1: yes
  PREDICTION C: NOT SUPPORTED on the validation seeds

R3d is reported, not judged. The closed-loop delays of `compare_agents.py summary` (tuned thresholds, unequal false-alarm rates) are descriptive only.
```

- Agent 2 is faster in R2d (62 steps at 5%), R2l (238.5), R3l (321.5) and, by 4 steps, in R2c, with Holm p = 0.000 in each.
- Agent 1 is faster in R3c only: median 162 against 242 steps at 5% (gap +80.0 [+43.5, +83.0], Holm p = 0.016), the one regime where the
  criterion holds. The advantage depends on the level: +80.5 at 10%, but within 4 steps at 1%, 2%, 20% and 40% (-3.0, -3.5, +2.0, -3.0).
  On the validation seeds the same gap (+80.5) had an interval that included zero (Holm p = 0.138).
- C needs at least 3 of the 5 lethal regimes; 1 of 5 hold, so C is not supported.
- R3d is reported, not judged: at 5% Agent 1 alerts earlier (median 860 against 1,092 steps; 33% and 63% of the runs censored).
- False alarms, matched clause: at the 5% level the rates are 0.035 (Agent 1) and 0.048 (Agent 2) in R0, and 0.024 and 0.067 in R1, a gap
  of 4.3 points against a tolerance of 5, so the clause holds (on the validation seeds it failed, with 5.4 points in R1). In R1 Agent 2
  still raises more false alarms than Agent 1 at every level (0.556 against 0.304 at 40%). This is descriptive: no prediction was
  registered on it.

Closed-loop delays and false alarms at the tuned thresholds (about 10% nominal false alarms for Agent 1, 1% for Agent 2), descriptive only:

```
Closed-loop detection, DESCRIPTIVE ONLY (tuned thresholds, unequal false-alarm rates; the verdict of C is in replay_detectors.py): delay in steps from the onset episode, censored at the end of the run
regime | median A1  cens | median A2  cens |           gap A2 - A1 [95% CI]
R2d    |       156  0.18 |        68  0.00 |    -87.500 [-182.500, -75.988]
R2c    |        45  0.00 |        43  0.00 |      -2.000 [ -3.000,  -2.000]
R2l    |       346  0.15 |         7  0.00 |  -339.500 [-538.050, -256.000]
R3d    |       966  0.45 |      1192  0.92 |  +225.500 [+143.988, +610.000]
R3c    |       162  0.00 |       246  0.00 |     +83.500 [ +5.000, +87.000]
R3l    |       446  0.20 |        91  0.00 |  -355.000 [-455.500, -256.988]

False alarms per episode (Agent 1 / Agent 2): R0 0.045/0.008  R1 0.037/0.013  R2d 0.063/0.011  R2c 0.063/0.011  R2l 0.063/0.011  R3d 0.063/0.011  R3c 0.063/0.011  R3l 0.063/0.011
Calm-regime gap (R0 and R1, Agent 2 minus Agent 1): -0.030
```

## 4. Behaviour in the latency regimes (R2l, R3l)

Agent 2 retreats in 94% of the post-onset episodes in R2l and in 77% in R3l, and revises 12.4 and 11.8 times per run (Agent 1: 2.5 and
1.9). Its failure rate is the lowest (0.02 in R2l, 0.00 in R3l), but it completes the task in 3% and 22% of the episodes. Agent 1 fails in
52% (R2l) and 28% (R3l) of the episodes and succeeds in 39% and 70%. The failure rate alone favours Agent 2 and the success rate favours
Agent 1; neither agent solves these regimes. The run-level analyses of section 5 of the validation record (`Ccum` by outcome, cost of a whole
run, runs with no success) were not recomputed on the test seeds.

## 5. Secondary analysis (decision D11): non-completion of the task

Registered in [research.md](research.md) (D11) before the test summary was read; run after it. It changes no verdict and no definition:
`failure` keeps its registered meaning (irreversible failure), and a retreat is not called a failure. Non-completion is
`1 - success = failure + retreat + timeout`.

```
Non-completion of the task = 1 - success = failure + retreat + timeout. SECONDARY ANALYSIS registered as D11 before the test summary was read.
Not part of the verdicts of B and C; `failure` keeps its registered definition (irreversible failure).
60 paired seeds, post-onset episodes; gap = Agent 2 minus Agent 1 (positive = Agent 1 completes more often);
95% paired bootstrap interval, Holm over the five lethal regimes (R3d reported, not judged).

regime | Agent 1: fail retreat timeout | non-compl | Agent 2: fail retreat timeout | non-compl |     gap (points) [95% CI]  Holm p | reading
R2d    |          0.45   0.00    0.00 |      0.45 |          0.42   0.14    0.00 |      0.56 |   +10.9 [  +5.2,  +16.8]  0.000 | Agent 1 completes more
R2c    |          0.03   0.07    0.00 |      0.10 |          0.00   0.20    0.00 |      0.20 |   +10.0 [  +5.3,  +13.7]  0.000 | Agent 1 completes more
R2l    |          0.52   0.06    0.03 |      0.61 |          0.02   0.94    0.00 |      0.97 |   +35.9 [ +29.6,  +42.3]  0.000 | Agent 1 completes more
R3d    |          0.00   0.00    0.00 |      0.00 |          0.00   0.00    0.00 |      0.00 |    +0.2 [  +0.0,   +0.5]      - | not judged
R3c    |          0.00   0.02    0.00 |      0.03 |          0.01   0.17    0.00 |      0.18 |   +15.5 [ +12.6,  +18.3]  0.000 | Agent 1 completes more
R3l    |          0.28   0.02    0.00 |      0.30 |          0.00   0.77    0.00 |      0.78 |   +47.4 [ +41.8,  +53.1]  0.000 | Agent 1 completes more
```

- Under the registered definition (irreversible failure only), Agent 2 has fewer failures in R2l (0.02 against 0.52) and R3l (0.00 against
  0.28). The breakdown of outcomes shows that this comes from a systematic retreat (94% and 77% of the episodes), which brings its success
  rate down to 3% and 22%.
- On non-completion Agent 1 is lower in all five lethal regimes, with Holm p = 0.000 (below the resolution of 5,000 replications): R2d 0.45
  against 0.56, R2c 0.10 against 0.20, R2l 0.61 against 0.97, R3c 0.03 against 0.18, R3l 0.30 against 0.78. In R2d the gap was not
  significant on the validation seeds (+5.1 points, Holm p = 0.119); here it is +10.9 [+5.2, +16.8].
- The two readings disagree, and neither agent is better on both. The failure rate rewards a retreat; the non-completion rate penalises
  it, although a retreat is a safe abort and a failure is not. Which one matters depends on the cost of not delivering, which this study
  does not define. Both are reported, with the registered one as the verdict.
- The retreats come from the alert-retreat-revise recipe shared by both agents (decisions D2 and D3), so this analysis cannot separate the
  effect of the trigger from the effect of the recipe.

## 6. Validation and test side by side

Failure and non-completion rates, Agent 1 / Agent 2 (validation seeds, then test seeds):

| Regime | Failure, validation | Failure, test | Non-completion, validation | Non-completion, test |
|---|---|---|---|---|
| R2d | 0.50 / 0.42 | 0.45 / 0.42 | 0.51 / 0.56 | 0.45 / 0.56 |
| R2c | 0.03 / 0.00 | 0.03 / 0.00 | 0.10 / 0.17 | 0.10 / 0.20 |
| R2l | 0.50 / 0.03 | 0.52 / 0.02 | 0.68 / 0.98 | 0.61 / 0.97 |
| R3c | 0.01 / 0.01 | 0.00 / 0.01 | 0.03 / 0.16 | 0.03 / 0.18 |
| R3l | 0.27 / 0.00 | 0.28 / 0.00 | 0.32 / 0.80 | 0.30 / 0.78 |

The pattern of the validation seeds is reproduced. Two differences: in R2d, Agent 2 had significantly fewer failures on the validation
seeds (Holm p = 0.040) and does not on the test seeds (gap -0.032 [-0.100, +0.038], Holm p = 0.400), while the non-completion gap
became significant; and in R3c the detection advantage of Agent 1 now has an interval that excludes zero, with the same size.

## 7. Where it does not work (protocol 3.5)

- **Dynamics and latency drifts (R2d, R2l, R3l).** Agent 1 reads the nociceptive channel only, so it cannot alert before the agent feels
  harm; a drift that changes the kinematics is visible to a prediction-error detector from the first steps. At equal false-alarm rates
  Agent 2 is faster by 62, 238.5 and 321.5 steps at 5%.
- **R3d.** No failure to avoid in either agent (0.00); both are censored in many runs at the low false-alarm levels.
- **Cost drifts (R2c, R3c).** The detection advantage of Agent 1 is limited to R3c and to the 5% and 10% levels. The damage and failure
  criteria of B are not met in either regime; in R2c Agent 1 is worse on both.
- **Open hypothesis, not tested.** In the latency regimes Agent 2 cycles through alert, retreat, revision and retry without repairing its
  model. The test seeds are spent: a diagnostic of the shared recipe, a second deviation (DEV-2) or a new agent needs fresh seeds or the
  validation seeds, labelled exploratory.

## 8. Notes on the record

- The verdict lines printed by `final_study.py summary` end with "on the validation seeds" although the data are the test seeds (the
  header line says 60 paired comparison seeds). See INC-3 in [protocol_deviations.md](protocol_deviations.md).
