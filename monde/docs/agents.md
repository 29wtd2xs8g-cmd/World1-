# Baseline agent (checklist 2.1)

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
damage predicted by the model along the route. A route is rejected if `Dhat_r` exceeds the estimated
margin `bhat = D - sum n * dt`. Route choice happens at the start of every episode (the margin
`bhat` is then `D`, because `Ccum` restarts at 0).

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

**Frozen baseline under drift** (`scripts/explore_baseline.py`, 3 validation seeds x 20 episodes, model never updated):

| Regime | `Ccum` before -> after `tau` | Failure rate after `tau` |
|---|---|---|
| R2d | 1.51 -> 3.37 | 5% |
| R2c | 1.51 -> 4.49 | 3% |
| R2l | 1.51 -> 4.33 | 70% |
| R3c | 1.51 -> 2.34 | 0% |

R1, R3d and R3l have not been run with this agent. 3 seeds is an exploration, not a result.

**Consequence for `D`.** `D = 5.005` was calibrated on the scripted crosser (2.50 per crossing). The
baseline damages 40% less per crossing, so abrupt harmful drifts triple its damage without reaching
`D`. See [status.md](status.md), open decision 1.
