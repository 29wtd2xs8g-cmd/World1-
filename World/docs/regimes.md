# Regime grid (checklist 0.4 x 0.5) and verification

Eight cells: a control, a harmless surprise, and abrupt/slow harmful drifts in each of the three natures.
`X` = 10%: a harmful regime must raise the mean cumulative damage by at least X% over R0.
Masking criterion: the one-step residual of the frozen nominal model stays below the noise
(SNR < 1) at the moment the damage exceeds R0 by X%.

## 1. Cells and exact parameters (as coded in `monde/regimes.py`)

Durations are fractions of `n_episodes`, so the study can scale: at 20 episodes, `tau` in {5..10};
at 100 episodes, `tau` in {25..50}.

| Regime | Nature | Parameters | What it tests | Automatic criterion (frozen model) |
|---|---|---|---|---|
| **R0** no change | none | `theta`, `psi`, `d` nominal | false alarms, nominal stability | mean `Ccum` stable, no drift active |
| **R1** harmless but surprising | dynamics (phantom zone) | `k* = Z3`, jump `psi3`: 0 -> 1.0 m/s^2, `theta3 = 0` | resistance to a surprise with no damage; useless revisions | in-zone residual above noise; `Ccum` and failure rate not worse than R0 |
| **R2d** harmful abrupt | dynamics | `k* = Z1`, `e1 = (-1, 0)`, jump `psi1`: 0 -> 1.0 m/s^2 | speed of detection, avoidance before `D` | `Ccum` above R0 by more than X% |
| **R2c** harmful abrupt | cost | `k* = Z1`, jump `theta1`: 1 -> 3 d.u./s | same, plus the wrong remedy (revising the world model when the cost changed) | a crossing through A costs `3*C1 > D`; failure rate above R0 |
| **R2l** harmful abrupt | latency | jump `d`: 0 -> **10** steps | same | one-step error large; `Ccum` above R0 |
| **R3d** harmful slow, masked | dynamics | `k* = Z1`, ramp `psi1`: 0 -> **0.25** m/s^2 over **25%** of the run | hidden errors that eat the margin `b[t]` | per-step SNR < 1 when `Ccum` exceeds R0 by X% |
| **R3c** harmful slow, masked | cost | `k* = Z1`, ramp `theta1`: 1 -> **2** d.u./s over **50%** of the run | same | `dtheta * phi / sigma_n < 1` when `Ccum` exceeds R0 by X% |
| **R3l** harmful slow | latency | steps: `d` 0 -> **10**, +1 step per episode (at 20 episodes) over **50%** of the run | same | **not maskable with this agent** (see below) |

Bold values differ from the first version of the grid (see section 2).

## 2. Adjustments after measurement

The first values did not reach the X = 10% harm or the masking condition with the frozen agent
(`scripts/explore_regimes.py`, 12 seeds x 20 episodes). They were changed once, before any learning agent:

| Cell | First value | Now | Reason |
|---|---|---|---|
| R2l | `d`: 0 -> 3 | 0 -> 10 | `d = 3` gives only +4% damage |
| R3d | `psi1` final 0.1 over 5 episodes | 0.25 over 25% of the run | 0.1 gives +7%; 0.25 about +17% with SNR < 1 inside the zone |
| R3c | `theta1` 1 -> 3 over 5 episodes | 1 -> 2 over 50% of the run | +0.1 d.u./s per episode = `sigma_n` at 20 episodes |
| R3l | `d` 0 -> 3, +1 every 2 episodes | 0 -> 10, +1 per episode | latency harm only appears for large `d` |

Sensitivity of the extra damage (frozen agent): `psi1` 0.1 -> +7%, 0.2 -> +14%, 0.3 -> +23%;
latency `d` = 3 -> +4%, 8 -> +12.7%, 10 -> +25%.

## 3. Measured results (frozen scripted agent, 12 seeds x 20 episodes)

| Regime | Result |
|---|---|
| R0 | `Ccum` about 2.52, failure rate 0 |
| R1 | `Ccum` unchanged; residual excess inside Z3 goes from 0 to about 2.2 (surprising, harmless) |
| R2d | after `tau`: `Ccum` 2.01x R0, failure rate 1.0, residual excess 2.75 |
| R2c | after `tau`: `Ccum` 2.04x R0, failure rate 1.0, residual excess 0 (invisible to prediction of `x, v`) |
| R2l | after `tau`: `Ccum` 1.15x R0, failure rate 0.3, residual excess 4.92 |
| R3d | first +10% damage at relative episode 3 (1.12x R0), residual excess 0.51 (masked) |
| R3c | first +10% damage at relative episode 1 (1.11x R0); cost SNR below 1 |
| R3l | +10% damage at relative episode 4, residual excess 3.68: **not maskable** |

Residual excess `= sqrt(max(0, mean |r|^2 - 2))`, with `r` the standardised one-step residual of the
frozen nominal model; 0 without drift, grows with the signal.

**Finding on R3l.** Latency only becomes harmful when the one-step error is already very visible, so
this cell cannot be a masked drift with this agent. Per the grid's own note ("if a cell cannot be
masked, write it rather than force it"), it is kept and documented, not forced.

**Finding on R2c.** The cost drift leaves trajectories unchanged: prediction error on `x, v` is blind to it.
If the world model also predicts `n`, prediction error sees it. This is why both Agent 2 variants must
predict `n` (decided) and why the first regimes are a fair test.

All of the above is enforced by `tests/test_regimes.py`.
