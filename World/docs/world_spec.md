# World specification (checklist 0.3)

The code in `toyworld/` is authoritative: where this page and an earlier note differ, the code wins.

## 1. Principle

The world provides ground truth (the instant, nature and cause of drift are known) and a non-trivial
dynamics to predict (continuous, nonlinear, noisy). It supports: irreversible graded damage with
failure when cumulative damage reaches `D`; three natures of hidden drift, one at a time; tunable
masking of the drift; two routes crossing different zones; a damage-free retreat.

## 2. Dynamics, damage, observation

Point mass in the plane, time step `dt`, Euler-Maruyama noise:

    x[t+1] = x[t] + dt * v[t+1]
    v[t+1] = v[t] + dt * ( a_max * u[t-d] - gamma * |v[t]| * v[t] + w[t](x[t]) )
                  + sqrt(dt) * sigma * eps[t],             eps[t] ~ N(0, I2)

Zones `k` have a smooth, compactly supported profile (equal to 1 at the centre, 0 outside radius `rho_k`):

    phi_k(x) = max(0, 1 - |x - z_k|^2 / rho_k^2)

Hidden force (dynamics drift) and graded damage (cost drift):

    w[t](x) = sum_k psi_k(t) * phi_k(x) * e_k             e_k: fixed unit vector
    c[t]    = sum_k theta_k(t) * phi_k(x[t])              damage units per second (d.u./s)
    Ccum[t] = sum_{u<=t} c[u] * dt                        cumulative damage
    b[t]    = D - Ccum[t]                                 remaining margin

Agent observation (it never sees `theta`, `psi`, `d`, `tau`, `k*`):

    o[t] = ( x[t] + eta, v[t] + eta', n[t] ),   eta ~ N(0, sigma_o^2 I)
    n[t] = max(0, c[t] + eps_n),                eps_n ~ N(0, sigma_n^2)       nociceptive signal

Episode ends at the first of: **failure** `Ccum[t] >= D` (irreversible); **success**
`|x[t] - g| <= r_g` with `Ccum[t] < D`; **timeout** `t = T - 1`.

Collisions: the wall `x = 5` can only be crossed through an opening; arena borders clip the position
and cancel the outward velocity.

**Damage-free retreat.** The start `x0` and a disc of radius `r0` around it are outside every zone.
Retreat = cancel the velocity, then return to `x0`. Damage suffered during the retreat is counted.
`retreat_action` is valid on the start side of the wall only.

## 3. Symbols and visibility

"Visible" = may enter an agent's input. Everything marked "no" is checked by leak tests.

| Symbol | Meaning | Unit | Visible |
|---|---|---|---|
| `t`, `dt`, `T` | step index, step duration, horizon | step, s, step | yes |
| `x[t]`, `v[t]` | position, velocity | m, m/s | yes, noisy (`sigma_o`) |
| `u[t]` in [-1,1]^2 | normalised command chosen by the agent | none | yes |
| `a_max` | maximum acceleration | m/s^2 | yes |
| `gamma` | quadratic drag | 1/m | no |
| `sigma`, `eps[t]` | process noise and its draw | m s^(-3/2), none | no |
| `z_k`, `rho_k`, `phi_k(x)` | zone centre, radius, profile | m, m, none | no |
| `theta_k(t)` | hidden damage rate of zone k (cost drift) | d.u./s | no |
| `psi_k(t)`, `e_k` | hidden force and direction (dynamics drift) | m/s^2, none | no |
| `d` | command delay (latency drift) | step | no |
| `c[t]` | instantaneous damage | d.u./s | no (only through `n[t]`) |
| `n[t]`, `sigma_n` | nociceptive signal and its noise | d.u./s | yes |
| `Ccum[t]` | cumulative damage | d.u. | no (estimated by summing `n`) |
| `D` | 2.981 d.u. = 2 * C1' | `C1'` = 1.4908 = damage of a direct crossing through A, no drift, by the baseline agent (200 episodes on the calibration seeds, 95% CI +-0.011) |
| `b[t]` | remaining margin `D - Ccum[t]` | d.u. | no (estimated) |
| `tau` | drift onset (`t_onset`) | step | no |
| nature, `k*` | kind of drift, true cause zone | -, index | no |
| `g`, `r_g`; `x0`, `r0` | goal and success radius; start and clear radius | m | yes |
| `sigma_o` | observation noise | m, m/s | no |

## 4. Parameter values (frozen in `toyworld/config.py`)

| Parameter | Value | Note |
|---|---|---|
| `dt`, `T` | 0.1 s, 500 steps | enough for a crossing, a detour and a retreat |
| Arena | 10 m x 10 m | |
| Wall | `x = 5`, openings A (`y = 5`, direct) and B (`y = 1.5`, longer), half-width 0.5 | the two routes |
| `x0`, `r0` | (1, 1), 1.5 | outside every zone |
| `g`, `r_g` | (9, 9), 0.5 | other side of the wall |
| `a_max`, `gamma` | 1 m/s^2, 0.5 1/m | terminal speed `sqrt(a_max/gamma)` ~ 1.41 m/s |
| `sigma` | 0.1 | per-step velocity noise ~ 0.03 m/s |
| `sigma_o`, `sigma_n` | 0.02, 0.1 | noisy but readable sensors |
| Zones | K = 3, `rho` = 1 | Z1 at (5, 5) in opening A; Z2 at (5, 1.5) in opening B; Z3 at (3, 3) on the approach: phantom zone |
| `e_k` | Z1, Z2: (-1, 0); Z3: (0, 1) | hidden force directions |
| Nominal `theta` | (1.0, 0.3, 0) d.u./s | route B costs less but is longer |
| `D` | 5.005 d.u. = 2 * C1 | `C1` = 2.5027 = damage of a direct crossing through A, no drift (scripted agent, 200 episodes, 95% CI +-0.015) |
| Run | 20 episodes (prototype), `tau` start of an episode in {5..10} | proportional to `n_episodes`: `tau` in [25%, 50%] of the run |

`D` was first calibrated at 5.005 = 2 * 2.5027 with the scripted crosser. The competent baseline agent crosses
faster (1.49 per crossing), so that `D` left it a margin of 3.3 times its damage and failures became rare under drift
(3 to 5% for R2c and R2d). `D` was recalibrated on the baseline, which is the reference of all comparisons, before any
trigger experiment. The scripted crosser (2.50 per crossing) now only serves to test the world.

Changes relative to the first draft of this specification, made after measurement or design review:
`T` 300 -> 500; Z3 (2, 8) -> (3, 3), because off-route the phantom zone was never visited and R1 was inert;
Gaussian profile -> compact support, because a Gaussian leaves no damage-free point;
latency and ramp amplitudes: see [regimes.md](regimes.md).

## 5. One drift per episode, form in time

A single drift is active in a run, so the cause `k*` is unique. It starts at `tau`, in zone `k*`.
Forms: **jump** (value changes at `tau`), **ramp** (linear), **steps** (+1 level every fixed duration).
Drift time is global to the run: `t_global = episode * T + t`.
