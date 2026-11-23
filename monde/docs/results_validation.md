# Validation comparison: Agent 1 against Agent 2

Recorded on 2026-11-23. Code and rules: [agents.md](agents.md) section 13, [protocol.md](protocol.md) sections 3.1 and 3.6 (decision D9).
Raw data: `archive/2026-11-23_validation_comparison/` (`compare.jsonl`: closed-loop runs; `replay.jsonl`: frozen trajectories
replayed with both detectors). Everything below is on the **validation seeds** (the 50 that tuned nothing); the test seeds
are untouched (`protocol/test_seed_ledger.jsonl` is empty).

## 1. Verdicts under the rules of D9

D9 was committed (`6b84f89`, tag `d9-recorded`) before any number of this comparison was read; see INC-2 in
[protocol_deviations.md](protocol_deviations.md) for the order of events.

| Prediction | Verdict on the validation seeds |
|---|---|
| **B** (cumulative damage, R2c and R3c) | **not supported**: in neither regime does Agent 1 have a lower `Ccum` or a lower failure rate than Agent 2 |
| **C** (detection, at equal false-alarm rate, 5% level) | **not supported**: 0 of the 5 lethal regimes; the matched false-alarm clause also fails (R1) |

Predictions A, D and E were not computed.

## 2. Prediction B (closed loop, tuned thresholds)

Outcomes first (post-onset episodes; R0 and R1 over all episodes; the frozen, oracle and always-adapting rows of the harmful
regimes need the D1 rerun and are not in this table):

```
regime family      kind          | failure success timeout retreat | Ccum|succ Ccum all | revis/run false al/ep
R0     damage_only frozen        |    0.00    1.00    0.00    0.00 |      1.50     1.50 |       0.0       0.000
R0     damage_only agent1_alert  |    0.00    1.00    0.00    0.00 |      1.50     1.50 |       0.8       0.039
R0     damage_only agent2_cusum  |    0.00    0.99    0.00    0.01 |      1.48     1.48 |       0.5       0.027
R0     damage_only always        |    0.00    1.00    0.00    0.00 |      1.49     1.49 |      20.0       0.000
R1     damage_only frozen        |    0.00    1.00    0.00    0.00 |      1.41     1.41 |       0.0       0.000
R1     damage_only agent1_alert  |    0.00    1.00    0.00    0.00 |      1.42     1.42 |       0.6       0.031
R1     damage_only agent2_cusum  |    0.00    0.99    0.00    0.01 |      1.41     1.41 |       0.6       0.032
R1     damage_only always        |    0.00    1.00    0.00    0.00 |      1.43     1.43 |      20.0       0.000
R2d    lethal      agent1_alert  |    0.50    0.49    0.00    0.01 |      2.27     2.55 |       2.5       0.059
R2d    lethal      agent2_cusum  |    0.42    0.44    0.00    0.14 |      2.31     2.53 |       5.1       0.017
R2c    lethal      agent1_alert  |    0.03    0.90    0.00    0.08 |      0.38     0.58 |       1.7       0.059
R2c    lethal      agent2_cusum  |    0.00    0.83    0.00    0.17 |      0.39     0.47 |       4.4       0.017
R2l    lethal      agent1_alert  |    0.50    0.32    0.12    0.07 |      2.06     2.34 |       3.0       0.059
R2l    lethal      agent2_cusum  |    0.03    0.02    0.01    0.94 |      1.79     0.50 |      12.9       0.017
R3d    damage_only agent1_alert  |    0.00    1.00    0.00    0.00 |      1.65     1.65 |       1.0       0.059
R3d    damage_only agent2_cusum  |    0.00    1.00    0.00    0.00 |      1.66     1.66 |       0.3       0.017
R3c    lethal      agent1_alert  |    0.01    0.97    0.00    0.02 |      0.93     0.98 |       3.9       0.059
R3c    lethal      agent2_cusum  |    0.01    0.84    0.00    0.15 |      0.74     0.91 |       5.0       0.017
R3l    lethal      agent1_alert  |    0.27    0.68    0.03    0.02 |      1.77     2.07 |       2.2       0.059
R3l    lethal      agent2_cusum  |    0.00    0.20    0.00    0.80 |      1.53     0.76 |      12.3       0.017
```

Paired contrasts, Agent 1 against Agent 2 (positive = Agent 1 better):

```
Agent 1 against Agent 2 (positive = Agent 1 better); Holm over the lethal regimes (failure) and the harmful ones (Ccum)
regime |          failure gap (points)  Holm p |           Ccum gap (relative)  Holm p | success A1 - A2 | B
R2d    |     -0.075 [ -0.139,  -0.011]   0.040 |     -0.006 [ -0.082,  +0.071]   1.000 |          +0.051 | no verdict
R2c    |     -0.027 [ -0.070,  -0.003]   0.012 |     -0.230 [ -0.469,  -0.089]   0.000 |          +0.068 | not met
R2l    |     -0.465 [ -0.538,  -0.393]   0.000 |     -3.726 [ -4.782,  -2.931]   0.000 |          +0.298 | no verdict
R3d    |     +0.000 [ +0.000,  +0.000]     nan |     +0.001 [ -0.008,  +0.009]   1.000 |          +0.003 | no verdict
R3c    |     -0.000 [ -0.014,  +0.012]   0.962 |     -0.086 [ -0.135,  -0.041]   0.000 |          +0.134 | not met
R3l    |     -0.268 [ -0.315,  -0.224]   0.000 |     -1.732 [ -2.013,  -1.495]   0.000 |          +0.487 | no verdict

PREDICTION B (R2c and R3c, Holm over the two; Ccum on all post-onset episodes): NOT SUPPORTED on the validation seeds
  Holm p, failure gap: R2c 0.008  R3c 0.962   Ccum gap: R2c 0.000  R3c 0.000
```

- R2c: `Ccum` is 23% higher with Agent 1 (0.58 against 0.47), failure 3% against 0%.
- R3c: `Ccum` is 9% higher (0.98 against 0.91), failure 1% for both. Holm p = 0.962 on the failure gap.
- Agent 1 succeeds more often in every lethal regime (+5.1 points in R2d to +48.7 in R3l); Agent 2 retreats in 17% (R2c) and
  15% (R3c) of the post-onset episodes against 8% and 2% for Agent 1.

## 3. Prediction C (replay on frozen trajectories, equal nominal false-alarm rates)

Delay in steps from the start of the onset episode to the first alert, right-censored at the end of the run; gap = Agent 2
median minus Agent 1 median (positive = Agent 1 faster). `*` marks the verdict level (5%).

```
regime level | median A1  cens | median A2  cens |           gap A2 - A1 [95% CI] |  FA A1  FA A2
R0       1%  |                 |                 |                                |  0.004  0.014
R0       2%  |                 |                 |                                |  0.007  0.021
R0       5%* |                 |                 |                                |  0.033  0.053
R0      10%  |                 |                 |                                |  0.058  0.087
R0      20%  |                 |                 |                                |  0.154  0.172
R0      40%  |                 |                 |                                |  0.340  0.395

R1       1%  |                 |                 |                                |  0.001  0.020
R1       2%  |                 |                 |                                |  0.005  0.038
R1       5%* |                 |                 |                                |  0.025  0.079
R1      10%  |                 |                 |                                |  0.044  0.123
R1      20%  |                 |                 |                                |  0.128  0.240
R1      40%  |                 |                 |                                |  0.297  0.544

R2d      1%  |       152  0.00 |        75  0.00 |       -77.0 [  -91.5,   +66.5] |  0.000  0.014
R2d      2%  |       152  0.00 |        68  0.00 |       -84.0 [ -100.5,    -4.5] |  0.003  0.020
R2d      5%* |       151  0.00 |        64  0.00 |       -87.0 [ -104.5,   -10.0] |  0.042  0.056
R2d     10%  |       150  0.00 |        62  0.00 |       -88.5 [ -105.5,   -12.5] |  0.076  0.096
R2d     20%  |        78  0.00 |        59  0.00 |       -18.5 [  -94.5,   -15.0] |  0.184  0.192
R2d     40%  |        76  0.00 |        54  0.00 |       -22.0 [  -96.0,   -19.0] |  0.370  0.418

R2c      1%  |        47  0.00 |        43  0.00 |        -4.0 [   -5.0,    -4.0] |  0.000  0.014
R2c      2%  |        47  0.00 |        43  0.00 |        -4.0 [   -4.5,    -4.0] |  0.003  0.020
R2c      5%* |        46  0.00 |        43  0.00 |        -3.0 [   -4.0,    -3.0] |  0.042  0.056
R2c     10%  |        46  0.00 |        43  0.00 |        -3.0 [   -4.0,    -2.0] |  0.076  0.096
R2c     20%  |        45  0.00 |        43  0.00 |        -2.0 [   -3.0,    -2.0] |  0.184  0.192
R2c     40%  |        44  0.00 |        42  0.00 |        -2.0 [   -2.0,    -1.0] |  0.370  0.418

R2l      1%  |       284  0.00 |         6  0.00 |      -277.5 [ -374.5,  -139.0] |  0.000  0.014
R2l      2%  |       284  0.00 |         6  0.00 |      -278.0 [ -375.5,  -139.5] |  0.003  0.020
R2l      5%* |       258  0.00 |         5  0.00 |      -252.5 [ -347.0,  -128.0] |  0.042  0.056
R2l     10%  |       151  0.00 |         4  0.00 |      -146.5 [ -327.5,  -119.0] |  0.076  0.096
R2l     20%  |       151  0.00 |         4  0.00 |      -147.0 [ -326.0,  -118.5] |  0.184  0.192
R2l     40%  |       130  0.00 |         3  0.00 |      -126.5 [ -235.5,  -100.0] |  0.370  0.418

R3d      1%  |      1202  0.76 |      1290  0.90 |       +87.5 [   -5.5,  +188.0] |  0.000  0.014
R3d      2%  |      1194  0.70 |      1199  0.72 |        +5.5 [  -93.0,  +118.0] |  0.003  0.020
R3d      5%* |      1036  0.40 |      1146  0.60 |      +109.0 [  -71.0,  +291.0] |  0.042  0.056
R3d     10%  |       777  0.24 |      1030  0.46 |      +253.5 [ -191.0,  +503.5] |  0.076  0.096
R3d     20%  |       522  0.06 |       362  0.08 |      -160.5 [ -222.5,    -3.0] |  0.184  0.192
R3d     40%  |       166  0.00 |       150  0.00 |       -15.5 [ -116.5,   +65.0] |  0.370  0.418

R3c      1%  |       251  0.00 |       248  0.00 |        -3.5 [   -6.0,   +60.5] |  0.000  0.014
R3c      2%  |       224  0.00 |       246  0.00 |       +22.5 [   -6.0,   +74.5] |  0.003  0.020
R3c      5%* |       163  0.00 |       244  0.00 |       +80.5 [   -4.5,   +84.5] |  0.042  0.056
R3c     10%  |       159  0.00 |       238  0.00 |       +79.5 [   -4.0,   +86.0] |  0.076  0.096
R3c     20%  |       155  0.00 |       153  0.00 |        -2.0 [   -6.5,    +2.5] |  0.184  0.192
R3c     40%  |       145  0.00 |       144  0.00 |        -0.5 [  -83.0,   +67.0] |  0.370  0.418

R3l      1%  |       542  0.02 |       138  0.00 |      -404.5 [ -550.0,  -381.0] |  0.000  0.014
R3l      2%  |       535  0.02 |        64  0.00 |      -471.5 [ -583.5,  -439.5] |  0.003  0.020
R3l      5%* |       404  0.02 |        61  0.00 |      -343.5 [ -443.5,  -322.0] |  0.042  0.056
R3l     10%  |       388  0.02 |        60  0.00 |      -328.5 [ -426.5,  -310.5] |  0.076  0.096
R3l     20%  |       272  0.00 |        56  0.00 |      -215.5 [ -314.0,  -147.5] |  0.184  0.192
R3l     40%  |       161  0.00 |        43  0.00 |      -118.0 [ -130.0,   -47.0] |  0.370  0.418
```

```
Prediction C at the 5% level (*), Holm over the five lethal regimes:
  R2d: not met   (Holm p = 0.000)
  R2c: not met   (Holm p = 0.000)
  R2l: not met   (Holm p = 0.000)
  R3c: not met   (Holm p = 0.138)
  R3l: not met   (Holm p = 0.000)
  regimes met: 0 of 5; measured false-alarm rates matched in R0 and R1: NO
  PREDICTION C: NOT SUPPORTED on the validation seeds

R3d is reported, not judged. The closed-loop delays of `compare_agents.py summary` (tuned thresholds, unequal false-alarm rates) are descriptive only.
```

- Agent 2 is faster in R2d, R2l and R3l (dynamics and latency drifts), and by 2 to 4 steps in R2c, where both alert on entering
  the zone. Agent 1 is faster only in R3c (163 against 244 steps at 5%; 159 against 238 at 10%), with an interval that includes
  zero (Holm p = 0.138); the gap is gone at the 20% level (155 against 153) and negative at 1%.
- R3d is reported, not judged: most runs are censored at the low levels.
- False alarms, matched clause: in R0 the rates at the 5% level are 0.033 (Agent 1) and 0.053 (Agent 2); in R1 they are 0.025 and
  0.079, a gap of 5.4 points against a tolerance of 5. The clause fails because Agent 2 raises more false alarms than Agent 1 in
  R1, at every level (0.123 against 0.044 at 10%, 0.544 against 0.297 at 40%). This does not change the verdict (0 of 5), and it
  is descriptive: no prediction was registered on it.

Closed-loop delays and false alarms at the tuned thresholds (about 10% nominal false alarms for Agent 1, 1% for Agent 2), descriptive only:

```
Closed-loop detection, DESCRIPTIVE ONLY (tuned thresholds, unequal false-alarm rates; the verdict of C is in replay_detectors.py): delay in steps from the onset episode, censored at the end of the run
regime | median A1  cens | median A2  cens |           gap A2 - A1 [95% CI]
R2d    |       244  0.18 |        72  0.00 |   -171.500 [-373.500, -79.000]
R2c    |        46  0.00 |        43  0.00 |      -3.000 [ -3.000,  -2.000]
R2l    |       423  0.18 |         6  0.00 |  -416.500 [-511.000, -268.000]
R3d    |      1094  0.48 |      1254  0.84 |   +160.000 [+59.500, +404.000]
R3c    |       162  0.00 |       248  0.00 |     +85.500 [ +0.000, +89.000]
R3l    |       529  0.24 |       114  0.00 |  -415.500 [-571.000, -345.000]

False alarms per episode (Agent 1 / Agent 2): R0 0.039/0.027  R1 0.031/0.032  R2d 0.059/0.017  R2c 0.059/0.017  R2l 0.059/0.017  R3d 0.059/0.017  R3c 0.059/0.017  R3l 0.059/0.017
Calm-regime gap (R0 and R1, Agent 2 minus Agent 1): -0.005
```

## 4. Behaviour in the latency regimes (R2l, R3l)

Agent 2 alerts within a few steps of the onset (median first alert at step 5 in R2l) and then alerts again in almost every
episode: 607 of 646 post-onset episodes end in a retreat in R2l (522 in R3l), and it revises in 641 of 646 (608 in R3l), with
12 to 13 revisions per run. Its failure rate is the lowest (0.03 in R2l, 0.00 in R3l), but it completes the task in 2% and 20%
of the episodes, and 76% of its runs have no success at all after the onset in R2l (10% for Agent 1). Agent 1 fails in 50% (R2l)
and 27% (R3l) of the episodes and succeeds in 32% and 68%. The failure rate alone favours Agent 2 and the success rate favours
Agent 1; neither agent solves these regimes.

## 5. Exploratory analyses (post hoc)

These were computed **after** the verdicts were read, on the validation seeds, to check whether retreats distort the reading of
`Ccum`. They are not part of any verdict and carry no pre-registered threshold. `Ccum` by outcome, mean over post-onset episodes
of the 50 seeds (n = episodes):

| Regime | Agent | failure | retreat | success |
|---|---|---|---|---|
| R2c | Agent 1 | n=19, 3.10 | n=47, 2.03 | n=580, 0.38 |
| R2c | Agent 2 | n=0 | n=108, 0.84 | n=538, 0.40 |
| R3c | Agent 1 | n=6, 3.01 | n=11, 2.66 | n=629, 0.93 |
| R3c | Agent 2 | n=5, 3.00 | n=97, 1.69 | n=544, 0.74 |

A retreat episode is not free (0.84 to 2.66): an alert comes after some damage. Agent 2 alerts earlier within the episode (mean
alert step of its retreats 40 against 46 in R2c, 44 against 54 in R3c), so its retreats cost less. Conditioning on success does
not favour Agent 1 (0.38 against 0.40 in R2c; 0.93 against 0.74 in R3c).

Cost of a whole run from the onset (`metrics.run_totals`) and cost up to the first success (`metrics.recovery`), mean over the
50 seeds:

| Regime | Agent | `Ccum` of the run | failures | retreats | revisions | `Ccum` to first success | runs with no success |
|---|---|---|---|---|---|---|---|
| R2c | Agent 1 | 7.5 | 0.4 | 0.9 | 1.3 | 3.5 | 0.02 |
| R2c | Agent 2 | 6.1 | 0.0 | 2.2 | 4.3 | 1.3 | 0.00 |
| R3c | Agent 1 | 12.6 | 0.1 | 0.2 | 3.5 | 1.5 | 0.00 |
| R3c | Agent 2 | 11.6 | 0.1 | 1.9 | 4.9 | 1.5 | 0.00 |
| R2l | Agent 1 | 30.0 | 6.3 | 0.8 | 2.6 | 10.1 | 0.10 |
| R2l | Agent 2 | 6.4 | 0.4 | 12.1 | 12.8 | 5.4 | 0.76 |
| R3l | Agent 1 | 26.8 | 3.5 | 0.2 | 1.8 | 1.5 | 0.00 |
| R3l | Agent 2 | 9.6 | 0.0 | 10.4 | 12.2 | 2.2 | 0.06 |
| R2d | Agent 1 | 32.8 | 6.3 | 0.2 | 2.1 | 7.1 | 0.00 |
| R2d | Agent 2 | 32.7 | 5.4 | 1.8 | 5.0 | 6.1 | 0.00 |

The non-support of B is therefore not an artefact of how retreats are counted: on the run-level cost, Agent 2 is lower in R2c and
nearly equal in R3c, and the cost to the first success is equal in R3c.

(The last column of this table was corrected on 2026-11-23: an earlier version showed 0.00, 0.80 and 0.10 for R2c Agent 1, R2l Agent 2 and R3l Agent 2, which were the exact values 0.02, 0.76 and 0.06 rounded to one decimal.)

## 5.1 Secondary analysis (decision D11): non-completion of the task

Post hoc on the validation seeds (computed after the verdicts were read, motivated by the retreats of section 4); registered in
[research.md](research.md) (D11) for the test seeds before their summary is read. It changes no verdict and no definition:
`failure` keeps its registered meaning (irreversible failure), and a retreat is not called a failure (it is reversible and costs
damage; a failure ends the episode). Non-completion is `1 - success = failure + retreat + timeout`.

```
Non-completion of the task = 1 - success = failure + retreat + timeout. POST HOC (computed after the verdicts were read), on the validation seeds.
Not part of the verdicts of B and C; `failure` keeps its registered definition (irreversible failure).
50 paired seeds, post-onset episodes; gap = Agent 2 minus Agent 1 (positive = Agent 1 completes more often);
95% paired bootstrap interval, Holm over the five lethal regimes (R3d reported, not judged).

regime | Agent 1: fail retreat timeout | non-compl | Agent 2: fail retreat timeout | non-compl |     gap (points) [95% CI]  Holm p | reading
R2d    |          0.50   0.01    0.00 |      0.51 |          0.42   0.14    0.00 |      0.56 |    +5.1 [  -1.3,  +11.6]  0.119 | no difference
R2c    |          0.03   0.08    0.00 |      0.10 |          0.00   0.17    0.00 |      0.17 |    +6.8 [  +1.6,  +10.9]  0.023 | Agent 1 completes more
R2l    |          0.50   0.07    0.12 |      0.68 |          0.03   0.94    0.01 |      0.98 |   +29.8 [ +23.3,  +36.5]  0.000 | Agent 1 completes more
R3d    |          0.00   0.00    0.00 |      0.00 |          0.00   0.00    0.00 |      0.00 |    +0.3 [  +0.0,   +0.7]      - | not judged
R3c    |          0.01   0.02    0.00 |      0.03 |          0.01   0.15    0.00 |      0.16 |   +13.4 [ +10.3,  +16.5]  0.000 | Agent 1 completes more
R3l    |          0.27   0.02    0.03 |      0.32 |          0.00   0.80    0.00 |      0.80 |   +48.7 [ +43.8,  +53.8]  0.000 | Agent 1 completes more
```

- Under the registered definition (irreversible failure only), Agent 2 has fewer failures in R2l (0.03 against 0.50) and R3l (0.00
  against 0.27). The breakdown of outcomes shows that this comes from a systematic retreat (94% and 80% of the episodes), which brings its
  success rate down to 2% and 20%.
- On non-completion Agent 1 is lower in R2c (0.10 against 0.17), R2l (0.68 against 0.98), R3c (0.03 against 0.16) and R3l (0.32 against
  0.80), with Holm p below 0.05; in R2d the gap (0.51 against 0.56) is not significant (Holm p = 0.119).
- The two readings disagree, and neither agent is better on both. The failure rate rewards a retreat; the non-completion rate penalises
  it, although a retreat is a safe abort and a failure is not. Which one matters depends on the cost of not delivering, which this study
  does not define. Both are reported, with the registered one as the verdict.
- The retreats come from the alert-retreat-revise recipe shared by both agents (decisions D2 and D3), so this analysis cannot separate
  the effect of the trigger from the effect of the recipe (section 6).

## 6. Where it does not work (protocol 3.5)

- **Dynamics and latency drifts (R2d, R2l, R3l).** Agent 1 reads the nociceptive channel only, so it cannot alert before the
  agent feels harm; a drift that changes the kinematics is visible to a prediction-error detector from the first steps.
- **R3d.** No failure to avoid; both agents are censored most of the time at the low false-alarm levels.
- **Cost drifts (R2c, R3c).** The detection advantage of Agent 1 is limited to R3c and is not significant after correction; the
  damage and failure criteria of B are not met.
- **Open hypothesis, not tested.** In the latency regimes Agent 2 cycles through alert, retreat, revision and retry without
  repairing its model: the loss on the revision window falls to about 0.36 of its value, but the alert returns in the next episode.
  A possible cause is the recipe shared by all agents (decisions D2 and D3: fine-tuning on the short window of a retreat
  episode, no replay). This is a limit of the common protocol, not of one detector, and it is not tested here.

## 7. Decision D10 (2026-11-23, after reading this comparison)

The confirmatory study goes to the test seeds with the agents, the thresholds, the revision recipe and the rules of D9 unchanged
(option A). The analyses of section 5 stay exploratory, and the hypothesis of section 6 stays open. The verdicts of B and C on
the test seeds will be reported whatever they are, next to this validation record.

Decision D11 (2026-11-23): the analysis of section 5.1 is registered as a secondary analysis for the test seeds, next to the verdicts.
