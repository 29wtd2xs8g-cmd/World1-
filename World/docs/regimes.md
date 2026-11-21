# Regime grid (checklist 0.4 x 0.5) and verification

Eight cells: a control, a harmless surprise, and abrupt/slow harmful drifts in each of the three natures.
`X` = 10%: a harmful regime must raise the mean cumulative damage by at least X% over R0.
Masking criterion: the one-step residual of the frozen nominal model stays below the noise
(SNR < 1) at the moment the damage exceeds R0 by X%.

## 1. Cells and exact parameters (as coded in `toyworld/regimes.py`)

Durations are fractions of `n_episodes`, so the study can scale: at 20 episodes, `tau` in {5..10};
at 100 episodes, `tau` in {25..50}.

| Regime | Nature | Parameters | What it tests | Automatic criterion (frozen model) |
|---|---|---|---|---|
| **R0** no change | none | `theta`, `psi`, `d` nominal | false alarms, nominal stability | mean `Ccum` stable, no drift active |
| **R1** harmless but surprising | dynamics (phantom zone) | `k* = Z3`, jump `psi3`: 0 -> 1.0 m/s^2, `theta3 = 0` | resistance to a surprise with no damage; useless revisions | in-zone residual above noise; `Ccum` and failure rate not worse than R0 |
| **R2d** harmful abrupt | dynamics | `k* = Z1`, `e1 = (-1, 0)`, jump `psi1`: 0 -> 1.0 m/s^2 | speed of detection, avoidance before `D` | `Ccum` above R0 by more than X% |
| **R2c** harmful abrupt | cost | `k* = Z1`, jump `theta1`: 1 -> 3 d.u./s | same, plus the wrong remedy (revising the world model when the cost changed) | a crossing of the baseline through A costs `3*C1' > D`; failure rate above R0 |
| **R2l** harmful abrupt | latency | jump `d`: 0 -> **10** steps | same | one-step error large; `Ccum` above R0 |
| **R3d** harmful slow, masked | dynamics | `k* = Z1`, ramp `psi1`: 0 -> **0.35** m/s^2 over **25%** of the run | hidden errors that eat the margin `b[t]` | per-step SNR < 1 when `Ccum` exceeds R0 by X% |
| **R3c** harmful slow, masked | cost | `k* = Z1`, ramp `theta1`: 1 -> **2** d.u./s over **50%** of the run | same | `dtheta * phi / sigma_n < 1` when `Ccum` exceeds R0 by X% |
| **R3l** harmful slow | latency | steps: `d` 0 -> **10**, +1 step per episode (at 20 episodes) over **50%** of the run | same | **not maskable** (see below) |

Bold values differ from the first version of the grid (see section 2).
`D` = 2.981 (calibrated on the baseline agent, see [world_spec.md](world_spec.md)); `C1'` = 1.4908.

## 2. Adjustments after measurement

The first values did not reach the X = 10% harm or the masking condition with the frozen scripted agent
(`scripts/explore_regimes.py`, 12 seeds x 20 episodes). They were changed before any trigger experiment:

| Cell | First value | Now | Reason |
|---|---|---|---|
| R2l | `d`: 0 -> 3 | 0 -> 10 | `d = 3` gives only +4% damage |
| R3d | `psi1` final 0.1 over 5 episodes | 0.35 over 25% of the run | scripted agent: 0.1 gave +7%, 0.25 about +17%. Baseline agent (feedback compensates part of the force): 0.25 gave only +10%, 0.35 gives +15% with the residual still below the noise |
| R3c | `theta1` 1 -> 3 over 5 episodes | 1 -> 2 over 50% of the run | +0.1 d.u./s per episode = `sigma_n` at 20 episodes |
| R3l | `d` 0 -> 3, +1 every 2 episodes | 0 -> 10, +1 per episode | latency harm only appears for large `d` |

Sensitivity of the extra damage (scripted agent): `psi1` 0.1 -> +7%, 0.2 -> +14%, 0.3 -> +23%;
latency `d` = 3 -> +4%, 8 -> +12.7%, 10 -> +25%.

R3d final value with the baseline agent (`scripts/sweep_target.py`, 6 seeds, damage once the ramp is over):

| `psi1` final | Damage over R0 | Residual excess in Z1 | Verdict |
|---|---|---|---|
| 0.25 | +10% to +13% | 0.3 to 0.8 | harm at the threshold |
| **0.35** | +11% to +18% | 0.4 to 0.97 | harmful and masked |
| 0.50 | +18% to +30% | 1.0 to 1.3 | visible, not masked |

Harm and visibility grow together: this is the trade-off that makes the regime interesting.

## 3. Measured results with the baseline agent (frozen model, 6 validation seeds x 20 episodes, `D` = 2.981)

`scripts/verify_regimes_baseline.py`. "After tau" averages all episodes from the drift episode on. Failing
episodes stop at `D`, so `Ccum` understates the harm of the strong regimes.

| Regime | `Ccum` after tau (x R0) | Failure after tau | Residual excess | Verdict |
|---|---|---|---|---|
| R0 | 1.51 | 0% | 0 | control |
| R1 | 0.94x | 0% | 2.18 in Z3 | surprising, harmless |
| R2d | 1.93x | 68% | 2.43 in Z1 | harmful, visible |
| R2c | 2.07x | 100% | 0.00 | harmful, invisible to the prediction of `x, v` |
| R2l | 1.95x | 87% | 3.99 | harmful, visible |
| R3d (0.35)* | 1.11x, up to 1.18x | 0% | 0.4 to 0.97 in Z1 | harmful, masked: first +10% at relative episode 4 (excess 0.77) |
| R3c | 1.57x, from 1.17x (rel 1) to 1.97x (rel 10) | 12% (60% at rel 10) | 0.00 | harmful, masked: first +10% at relative episode 1 |
| R3l | 1.61x | 49% | 1.3 at rel 0, 3.4 on average | harmful but **not maskable**: first +10% at rel 1 with excess 1.79 |

\* measured in the sweep above, same seeds. `R3d` at 0.25 in the main run gave 1.07x.

Residual excess `= sqrt(max(0, mean |r|^2 - 2))`, with `r` the standardised one-step residual of the
frozen nominal model; 0 without drift, grows with the signal. Six seeds give an exploration, not
confidence intervals: the final study uses at least 30 seeds.

## 4. Earlier measurements with the scripted agent (12 seeds x 20 episodes, `D` = 5.005)

These remain the basis of `tests/test_regimes.py`, which uses the scripted agent (fast, no PyTorch).
The tests measure damage without failure cutoff, so `Ccum` is not capped at `D`.

| Regime | Result |
|---|---|
| R0 | `Ccum` about 2.52, failure rate 0 |
| R1 | `Ccum` unchanged; residual excess inside Z3 goes from 0 to about 2.2 |
| R2d | after `tau`: `Ccum` 2.01x R0, residual excess 2.75 |
| R2c | after `tau`: `Ccum` 2.04x R0, residual excess 0 |
| R2l | after `tau`: `Ccum` 1.15x R0, residual excess 4.92 |
| R3d (0.25, before the second adjustment) | first +10% damage at relative episode 3 (1.12x R0), residual excess 0.51 (masked) |
| R3c | first +10% damage at relative episode 1 (1.11x R0); cost SNR below 1 |
| R3l | +10% damage at relative episode 4, residual excess 3.68: not maskable |

**Finding on R3l.** Latency only becomes harmful when the one-step error is already very visible: with both
agents the cell is not a masked drift. Per the grid's own note ("if a cell cannot be masked, write it
rather than force it"), it is kept and documented, not forced.

**Finding on R2c.** The cost drift leaves trajectories unchanged: prediction error on `x, v` is blind to it.
If the world model also predicts `n`, prediction error sees it. This is why both Agent 2 variants must
predict `n` (decided) and why the first regimes are a fair test.

All of the above that concerns the scripted agent is enforced by `tests/test_regimes.py`.
