# Agents: baseline (checklist 2.1), references and trigger interface (2.2, 2.4)

A competent agent *before* the drift. Without it, any gap after the drift cannot be interpreted.
The same agent, with its model never updated, becomes the **frozen** reference of checklist 2.2.

## 1. Decisions

| Decision | Choice |
|---|---|
| Objective | reach `g` without reaching `D`: plan on time + `lambda` * predicted damage, under a margin constraint |
| Route before drift | route A (direct) is preferred; B is the fallback. Tested: A chosen in at least 95% of pre-drift episodes |
| Competence before drift | success >= 95%, failure <= 1%, mean `Ccum` <= 1.1 * C1 (C1 = 2.5027), on the validation seeds |
| World model | PyTorch MLP, pretrained in the nominal world, predicts velocity change and `n` |
| Planner | CEM model-predictive control tracking the waypoints of the chosen route |

Why A must be preferred: every harmful dynamics or cost drift hits Z1 (opening A), and the phantom
zone Z3 is only on the way to A. An agent that preferred B would never meet these drifts.

## 2. Objective

The world returns reward 0; the objective lives in the agent's planning cost.

    route cost   J_r = L_r / v_c + lambda * Dhat_r              (time + damage), lambda = 0.1
    rollout cost J   = sum_h |x_h - wp|^2 + lambda * sum_h nhat(x_h) * dt + 0.01 * sum_h |u_h|^2

`L_r` is the length of the polyline of route `r`, `v_c = 0.9 m/s` the cruise speed, `Dhat_r` the
damage predicted by the model along the route. A route is allowed if `Dhat_r < margin_frac * bhat`, with the estimated
margin `bhat = D - sum n * dt` and `margin_frac` = 0.7 for every agent since protocol deviation DEV-1
([protocol_deviations.md](protocol_deviations.md)); the first protocol used 1.0 (the whole margin). Route choice happens
at the start of every episode (the margin `bhat` is then `D`, because `Ccum` restarts at 0), so route A needs a
predicted damage below 0.7 x 2.981 = 2.087; the learned model predicts 1.48.

## 3. World model (`toyworld/model.py`)

One model, two MLP heads. Everything it sees is observed (noisy); nothing comes from the ground truth.

| Head | Input | Output |
|---|---|---|
| dynamics | position (2), velocity (2), the last `K = 12` commands `u[t], ..., u[t-K+1]` (24) | `v[t+1] - v[t]` (2) |
| nociception | position (2) | `n[t]` (1) |

- `K = 12` covers a command delay up to 11 steps. Without the command history the delay `d` cannot be learned.
- The position update is kinematic (`x[t+1] = x[t] + dt * v[t+1]`): the model does not learn it.
- The nociception head takes only the position: damage is a spatial field. This is a modelling choice.
- Each head: 2 hidden layers of 128 (dynamics) and 64 (nociception) units, ReLU. Inputs and targets are standardised with statistics of the pretraining set.
- Training: Adam, MSE, 30 epochs, batch 512, learning rate 1e-3.

## 4. Pretraining data (`toyworld/data.py`)

Collected in the **nominal** world (`theta` nominal, `psi = 0`, `d = 0`), with a large `D` so that
exploration is never cut short by failure.

- 2000 segments of 100 steps, each starting from a random state (position anywhere in the arena, random velocity). This is a privilege of the data-collection stage; the agent in evaluation always starts at `x0`.
- Commands: piecewise-constant random thrust (held 5 to 20 steps). This covers the inside of every zone, both openings and the wall.
- Seeds for pretraining are disjoint from validation and test seeds (`toyworld/seeds.py`).

## 5. Planner (`toyworld/agent.py`)

- **Route choice.** Waypoints of route A: (4.2, 5), (5.8, 5), g. Route B: (4.2, 1.5), (5.8, 1.5), g. `J_r` is computed with the model's nociception head on points sampled along the polyline; the floor of `n` (its value at `x0`, which is damage-free) is subtracted.
- **Tracking.** CEM over command sequences: horizon 20 steps, 256 samples, 25 elites, 4 iterations. The model rolls the state forward. The first command is applied, the solution is shifted for the next step. The waypoint advances when the agent is within 0.3 m.
- Fixed random seed per run, so evaluations are reproducible.

## 6. Competence check (`scripts/eval_baseline.py`)

Run on the validation seeds, regime R0 (no drift): success rate, failure rate, mean `Ccum`, share of
episodes through route A. All four criteria must hold. Hyperparameters may be tuned on the
validation seeds only; the test seeds are not used at this stage.

**Re-run with `margin_frac` 0.7 (DEV-1, 2026-11-22)**, same 30 validation seeds x 3 episodes, regime R0: success 100%,
failure 0%, mean `Ccum` 1.50, route A 100%, 98 steps per episode, all four criteria met. The numbers are those of the
first protocol: before any drift the buffer changes nothing, because route A predicts 1.48 against a threshold of 2.087.

## 7. Known issues and limits

- **The nociceptive signal has a positive floor.** `n = max(0, c + eps_n)` with `eps_n ~ N(0, sigma_n^2)`, so when `c = 0`, `E[n] = sigma_n / sqrt(2 pi) = 0.04`. The estimate `sum n * dt` drifts upward by about 0.004 per step, about 0.5 at the end of a 130-step episode, 10% of `D`. Any margin-coupled trigger (Agent 1) must debias it.
- **Prediction A.** A plain MLP has no latent variables: "working memory" reduces to the command window `K`. The latent-variable measure is reserved for the second model type (checklist 6.5).
- **Wall.** The wall collision is a discontinuity that a MLP fits only approximately; waypoints placed on both sides of each opening keep the agent away from it.
- **Frozen after pretraining.** The baseline never updates its model; revision belongs to Agents 1 and 2.

## 8. Results

**Pretraining** (`scripts/train_baseline.py`): 200,000 samples, 47 s on CPU. Holdout RMSE: velocity
change 0.072 (noise floor about 0.042; the excess comes from wall and border collisions), `n` 0.064
(noise floor about 0.06 outside zones, 0.1 inside).

**Competence before drift** (`scripts/eval_baseline.py`, 30 validation seeds x 3 episodes, regime R0):

| Criterion | Required | Measured |
|---|---|---|
| success | >= 95% | 100% |
| failure | <= 1% | 0% |
| mean `Ccum` | <= 2.75 (1.1 * C1) | 1.50 |
| route A | >= 95% | 100% |

Episodes last 98 steps on average (the scripted crosser needs 132). With the analytic model the same
planner gives `Ccum` 1.51, so the learned model costs nothing measurable here.

**Frozen baseline under drift**: see [regimes.md](regimes.md), section 3 (6 validation seeds x 20 episodes, `D` = 2.981).
A first exploration with `D` = 5.005 (3 seeds) showed that the baseline rarely failed under drift (3 to 5% for R2c
and R2d), which led to the recalibration of `D` on this agent.

**Calibration of `D`** (`scripts/calibrate.py model.pt 4 2`): 200 episodes on the calibration seeds, route A in 100%,
`C1'` = 1.4908 (std 0.081, 95% CI +-0.011), `D = 2 * C1'` = 2.981.

## 9. References, revision and trigger interface (checklist 2.2 and 2.4)

**One agent, several triggers.** Agents 1 and 2 will be the baseline agent with a different trigger; the references
are the same agent at the two ends of the trigger scale, plus an oracle.

| Reference | Planning model | Revision | Code |
|---|---|---|---|
| frozen | pretrained network | never | `NeverTrigger` |
| oracle | `AnalyticModel` with the true current `theta`, `psi`, `d` (privileged read of the world) | never | `run_agent(..., oracle=True)` |
| always-adapting | pretrained network | at the end of every episode | `AlwaysTrigger` |

The oracle is the same planner with a correct model; the noise stays, so it is not an optimal controller. Its role
is the headroom (D1): the most any trigger can gain, to a first approximation.

**Trigger interface** (`toyworld/triggers.py`). A trigger reads a `StepRecord` and answers two questions:
`step(rec)` (alert now?) and `revise_at_end(alerted)` (fine-tune at the end of this episode?). A record holds only
`t`, the observation `o[t]`, the model's predicted `n` at the observed position, the one-step velocity residual of the
agent's own model, and the margin estimate `b_hat = D - dt * sum(n)` (floor of `n` not removed). It holds nothing
from the ground truth; a test checks the field list.

**Run protocol** (`toyworld/adaptive.py`, decisions D2 and D3 of [research.md](research.md), section 6):

1. The first alert of an episode on the start side of the wall makes the agent retreat to the clear disc around `x0`;
   the episode closes with outcome `retreat`. Past the wall the alert is only recorded.
2. At the end of an episode, if the trigger asks for it, both networks are fine-tuned on that episode's window
   (`toyworld/revision.py`): `steps` Adam steps on minibatches drawn from the window, standardisation buffers frozen,
   no replay. The budget is the same for every adaptive agent.
3. The next episode is the retry, from `x0`, with a fresh margin estimate. Each run revises its own copy of the
   pretrained model.

With `NeverTrigger` the runner reproduces `run_frozen` to the last digit (test), so earlier frozen results stay valid.

**Choice of the budget** (`scripts/tune_revision.py`, calibration seeds 1000-1001 only, always-adapting agent,
10 episodes per run, post-`tau` episodes). Rule fixed before looking: lowest mean post-`tau` `Ccum` over R2d and R2c.

| Setting (steps, lr) | R2d `Ccum` | R2c `Ccum` | R2c failure |
|---|---|---|---|
| frozen | 2.95 | 3.10 | 1.00 |
| **(50, 1e-3)** | 2.94 | **0.84** | 0.17 |
| (150, 1e-3) | 2.89 | 2.67 | 0.83 |
| (150, 3e-3) | 2.73 | 2.67 | 0.83 |

Chosen: 50 steps, batch 128, lr 1e-3. The sample is small (2 seeds, about 6 post-`tau` episodes each), so only the
direction is reliable: more steps on a window that covers little of the arena hurt on R2c, a sign of forgetting
(no replay, by decision D2).

**Limits.** The window covers the route of one episode, so the nociception head can forget zones that were not
visited (for example Z2 on route B) and mislead the next route choice. This is a consequence of D2 and is measured, not
corrected, here.

## 10. Headroom per regime (checklist 2.3), run 1: first protocol (`margin_frac` 1.0)

Archived exactly in `archive/2026-11-22_run1_margin1.0/` (deviation DEV-1). `scripts/headroom.py`: 6 validation seeds (2000-2005), 20 episodes per run, post-`tau` episodes, paired scenarios.
Frozen results are those of `results/regimes_baseline.json` (the Never-trigger runner reproduces them exactly).
Interval: bootstrap over seeds (2000 draws); with 6 seeds it is indicative only.

| Regime | `Ccum` frozen | oracle | always | Failure frozen / oracle / always | Headroom (oracle) | Gain of always-adapting |
|---|---|---|---|---|---|---|
| R2d | 2.91 | 2.79 | 2.86 | 0.69 / 0.44 / 0.62 | +4.3% [+3.3, +5.2] | +1.8% |
| R2c | 3.12 | 0.39 | 0.62 | 1.00 / 0.00 / 0.08 | +87.4% [+87.2, +87.6] | +80.2% |
| R2l | 2.94 | 2.09 | 1.72 | 0.86 / 0.00 / 0.39 | +28.7% [+25.6, +31.5] | +41.3% |
| R3d | 1.62 | 1.65 | 1.63 | 0.00 / 0.00 / 0.00 | -1.6% [-2.2, -1.0] | -0.3% |
| R3c | 2.35 | 2.37 | 1.91 | 0.11 / 0.15 / 0.05 | -0.8% [-1.3, -0.3] | +18.5% |
| R3l | 2.42 | 1.90 | 1.83 | 0.48 / 0.00 / 0.22 | +21.5% [+19.5, +24.0] | +24.5% |

**Criterion D1 (research.md, section 6): not met.** (The R3d frozen reference of this run was stale: corrected to +1.5% in DEV-1, which changes no verdict.) Three regimes reach 15% (R2c, R2l, R3l), none of them in {R3d, R3c}.

**Why** (from `scripts/headroom.py detail`):

1. **R3d is harmful but never fatal.** All three agents cross by route A and reach the goal in 100% of the episodes
   (`Ccum` 1.6 against `D` = 2.98). The masked force costs about +15% damage; knowing it changes nothing because the
   planner already compensates by feedback. No trigger can gain on failures here.
2. **R3c: the baseline's route rule has no safety buffer.** Route A is allowed while its predicted damage is below
   the whole remaining margin. The oracle therefore switches to B only when `theta` reaches `D / 1.49` and then
   fails as often as the frozen agent (15% against 11%). The headroom is nil because of the rule, not because
   knowing the cost is useless.
3. **`Ccum` alone misleads.** A failure caps `Ccum` at about `D` (R2d: failures 0.69 -> 0.44, `Ccum` only -4%), and a
   timeout lowers it (R2l, always-adapting: 16% success, 39% failure, 45% timeouts). The always-adapting agent can
   beat the oracle on `Ccum` (R2l, R3c) without being better: the oracle is not an upper bound. Outcomes (success,
   failure, timeout) must be reported next to `Ccum` in every comparison.

**Exploratory, post hoc** (not part of the pre-registered measurement): `AgentParams.margin_frac = 0.7`, chosen
a priori and not tuned (a route is allowed if its predicted damage is below 0.7 of the margin), R3c only, same 6
seeds (`headroom.py run model.pt 6 2 0.7 R3c`). Frozen is unchanged (its predictions stay nominal). Result:

| R3c, `margin_frac` 0.7 | `Ccum` | Failure | Route A | Steps (success) |
|---|---|---|---|---|
| frozen | 2.35 | 0.11 | 1.00 | 98 |
| oracle | 0.98 | 0.00 | 0.39 | 107 |
| always-adapting | 1.01 | 0.01 | 0.41 | 107 |

Headroom +58% [+54, +62]; the always-adapting agent already gets +57%. The price is 9 more steps per crossing, because
route B is longer. Whether to adopt this buffer is an open decision ([status.md](status.md)).

## 11. Headroom per regime under the amended protocol (`margin_frac` 0.7, DEV-1)

Same 6 validation seeds and protocol as section 10. Every agent, frozen included, ran with the current code and the same rule
(`margin_frac` 0.7); results in `results/headroom_rho0.7.jsonl`, printed by `scripts/headroom.py summary 0.7`. Outcomes first
([protocol.md](protocol.md), section 3.1). Means over seeds, post-`tau` episodes; intervals: bootstrap over seeds, 5,000 replications.

**Outcomes** (frozen / oracle / always-adapting).

| Regime | Failure | Success | Timeout (always) | `Ccum` over successes |
|---|---|---|---|---|
| R2d | 0.69 / 0.44 / 0.66 | 0.31 / 0.56 / 0.34 | 0.00 | 2.65 / 2.60 / 2.25 |
| R2c | 1.00 / 0.00 / 0.08 | 0.00 / 1.00 / 0.92 | 0.00 | none / 0.39 / 0.40 |
| R2l | 0.86 / 0.00 / 0.41 | 0.14 / 1.00 / 0.16 | 0.43 | 2.50 / 2.09 / 2.17 |
| R3c | 0.11 / 0.00 / 0.01 | 0.89 / 1.00 / 0.99 | 0.00 | 2.26 / 0.98 / 0.98 |
| R3l | 0.48 / 0.00 / 0.18 | 0.52 / 1.00 / 0.67 | 0.15 | 1.87 / 1.90 / 1.71 |
| R3d (no mortality) | 0 / 0 / 0 | 1 / 1 / 1 | 0 | 1.67 / 1.65 / 1.63 |

**Headroom of the oracle.**

| Regime | Failure, frozen minus oracle (points) | D1: `Ccum` over all episodes |
|---|---|---|
| R2d | +24.8 [+17.5, +30.4] | +4.3% [+3.3, +5.2] |
| R2c | +100.0 [+100.0, +100.0] | +87.4% [+87.2, +87.6] |
| R2l | +85.8 [+70.5, +97.4] | +28.7% [+25.7, +31.4] |
| R3c | +11.4 [+4.9, +19.4] | **+58.1%** [+53.6, +62.0] |
| R3l | +48.1 [+40.6, +55.0] | +21.5% [+19.5, +24.0] |
| R3d | 0 | +1.5% [+0.9, +2.1] |

**Criterion D1 is met on these 6 seeds**: four regimes reach 15% (R2c, R2l, R3c, R3l), R3c among them. With 6 seeds the
intervals only describe seed-to-seed spread; the verdict has to be confirmed on the 60 validation seeds (command below).

**Reading notes.**

1. On R3c the oracle now leaves route A for B in 60% of the post-`tau` episodes and avoids every failure; the always-adapting
   agent obtains almost the same (+57.0%, failure 0.01).
2. The always-adapting `Ccum` gain on R2l (+39.6%) is mostly timeouts (43% of the episodes neither succeed nor fail): do not read it
   as a better outcome. This is what the outcome-first rule is for.
3. R2d keeps a small headroom in `Ccum` (+4.3%) because failures cap it, but a large one in failures (+24.8 points).
4. Frozen is identical under 0.7 and 1.0 (12 runs recomputed under both, identical episode by episode).

5. Revision budget re-checked under 0.7 (`scripts/tune_revision.py`, calibration seeds 1000-1001, R2d and R2c, rule: lowest mean
   post-`tau` `Ccum`). (50 steps, lr 1e-3) stays the best: mean 1.79 (R2d 2.73, R2c 0.84), against 2.06 for (150, 3e-3), 2.68 for
   (150, 1e-3) and 3.03 without revision. The always-adapting reference therefore does not need to be redone.

**To confirm D1 on 60 seeds** (about 11 minutes per seed on 2 cores, so run it on a machine with more workers):

    PYTHONPATH=. python scripts/headroom.py run results/baseline.pt 60 <workers> 0.7 all
    PYTHONPATH=. python scripts/headroom.py summary 0.7
