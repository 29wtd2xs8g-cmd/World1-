# Research design

## 1. Question (checklist 0.1)

Should an agent's failure be seen as a failure of **perception** rather than of **prediction**?
Can perception format an alert signal that prevents hidden errors from accumulating?

> When environment drifts are masked in prediction space but accumulate irreversible risk, does an
> alert signal formed at the perceptual level (coupled to the remaining safety margin) reduce
> cumulative damage and failure rate more than a trigger based on prediction error?

Justification: the world model can predict with confidence in a state space that no longer reflects
what matters.

## 2. Idea

A continual world model (model-based RL, planning by MPC) must decide **when to open a revision**.

| | Agent 1 (perceptual alert) | Agent 2 (prediction error) |
|---|---|---|
| Trigger | observed harm: a nociception-like signal `n` crossing a negativity threshold, coupled to the remaining margin `b[t] = D - Ccum[t]` | the one-step prediction error of the world model (raw, or via a change detector such as CUSUM) |
| Sees drift when | the damage becomes visible in `n` | the dynamics change by more than the prediction noise |
| Expected weakness | misses harmless surprises by design (that is the point) | blind when the drift is masked in prediction space; reacts to harmless surprises |

Both agents are given a world model that **predicts `n`** as well as position and velocity (decided:
otherwise the prediction-error agent is blind to cost drift by construction and keeps stopping).
Further variants of Agent 2 (value-based trigger, viability margin) are possible baselines.

Why a continuous, stochastic, nonlinear world: the prediction problem must be non-trivial (drag,
process noise, inertia), so that prediction error is not zero by construction and the agent that
triggers on it is a real opponent. The idea must not depend on a discrete grid.

## 3. Predictions and stopping rules (checklist 0.2)

All thresholds are frozen before the first experiment. "Validation seeds" are separate from test seeds.

| | Prediction | Stopping rule |
|---|---|---|
| **A. Resources** (CPU time and FLOPs per episode; working memory = context window + number of active latent variables) | Agent 1 uses at least 20% less active working memory and 10% fewer FLOPs than Agent 2 | If Agent 1 consumes more than Agent 2 despite its perceptual path, stop and check the implementation of the alert path |
| **B. Cumulative damage** (main metric: `Ccum`) | Mean `Ccum` of Agent 1 at least 15% lower than Agent 2 on validation seeds, and failure rate at least 10% lower | If the relative gap in `Ccum` is below 5%, conclude the gain is not clear enough in this regime and move to the next regime |
| **C. Detection** (delay `dt_det = t_alert - t_onset`) | On failure episodes, the median detection delay of Agent 1 is at least 10 environment steps shorter than Agent 2's, while keeping the false-alarm rate per episode 5% lower | If Agent 1 is faster only by blowing up false alarms, stop evaluating this setting and recalibrate the alert threshold on validation |
| **D. Identification** (finding the cause, not only detecting a problem: which zone/path leads to the error) | At equal exploration budget, Agent 1 identifies the true cause in at least 15% of episodes versus at most 5% for Agent 2, or with a median identification delay at least 5 steps shorter | If neither agent exceeds 40% identification on validation, check that the signal is observable |
| **E. Recovering the goal** (reach the target without triggering a continuing failure loop) | Agent 1 reaches the target at least 15 steps before Agent 2, with a lower recurrence rate of failure | If Agent 1 is faster but with higher exploration damage and a much higher cost (more than 15% of the context window), stop and tighten the margin constraint |

**Definition of the detection delay (decided).** `dt_det = t_alert - tau`, in global environment steps
of the run, where `tau` is the drift onset (`t_onset`). An alert before `tau` is a false alarm, not a
negative delay. A drift that is never alerted gives a censored delay (missing, not zero).
This delay includes the travel time to the affected zone `k*`: an agent cannot feel anything before
entering it. It is therefore also logged from the exposure time, as a secondary metric:
`dt_exp = t_alert - t_exp`, with `t_exp` the first step >= `tau` where the drift acts on the agent
(`phi_k*(x) > 0` for dynamics and cost; first nonzero command for latency).

## 4. Methodology

- **Pre-registration**: parameter values, thresholds and metrics are frozen before experiments.
- **Ground truth**: onset `tau`, nature and cause `k*` of every drift are known and logged, never given to agents.
- **Paired scenarios**: noise is seeded per `(run seed, episode)`; agents are compared on identical scenarios.
- **Seeds**: at least 30 paired seeds; validation seeds are separate from test seeds. Episodes per run and number of seeds are distinct quantities.
- **Leak tests**: automatic tests check that the agent's input contains none of the hidden symbols (`theta`, `psi`, `d`, `tau`, `k*`, regime, seed).
- **Frozen reference agent**: a scripted agent that adapts to nothing verifies the world before any learning agent exists.
- **Beyond the toy world** (checklist section 6): the toy world supports conclusions on trigger logic, accuracy of cause attribution and margin measurement; the size of the margin and the ranking of triggers must be confirmed on standard environments.

## 5. Regimes

Three **natures** of drift (what changes), crossed with the **form** in time (abrupt, slow/masked),
plus controls. See [regimes.md](regimes.md).

| Nature | Hidden parameter | Expected remedy |
|---|---|---|
| Dynamics | force `psi[k*]` | revise the world model |
| Cost | damage rate `theta[k*]` | revise the cost measure |
| Latency | command delay `d` | align time |

Masking is a tunable signal-to-noise ratio (SNR): dynamics SNR `= psi * phi * sqrt(dt) / sigma`,
cost SNR `= dtheta * phi / sigma_n`. Below 1, the one-step error sees almost nothing while the effect
accumulates in the trajectory and in the damage.

## 6. Decisions frozen before the headroom measurement (2026-11-22)

All five were taken before any trigger experiment and before the headroom (checklist 2.3) was measured.

**D1. Headroom criterion (decision B of checklist 7).** The headroom of a regime is
`(Ccum_frozen - Ccum_oracle) / Ccum_frozen`, with `Ccum` the mean cumulative damage per episode over the
post-`tau` episodes, on paired validation seeds. The project continues only if the headroom is at least 15% in
at least 3 harmful regimes (R2d, R2c, R2l, R3d, R3c, R3l), at least one of them in {R3d, R3c}. Otherwise the
question changes before any trigger is built. Rationale: Agent 1 cannot beat Agent 2 by more than the gap between
the oracle and the frozen agent, and prediction B asks for 15%. The always-adapting reference is reported next to
it, but it is not part of the criterion.

**D2. Revision recipe.** Both networks are fine-tuned on the window of the last episode (all its transitions,
observed quantities only), at the end of that episode, without replay of older data. The gradient budget and the
hyperparameters are the same for every adaptive agent: `steps`, `batch`, `lr` of `RevisionParams`
(`toyworld/revision.py`). They were chosen on the calibration seeds (`scripts/tune_revision.py`), never on
validation or test seeds.

**D3. Response to an alert.** The first alert of an episode makes the agent retreat to a damage-free point, then
the model is revised (D2), then the agent retries in the next episode with a fresh margin estimate. The response
is identical for every trigger, otherwise a gap would measure the response and not the trigger. The retreat is
defined on the start side of the wall; past the wall the agent keeps going to the goal, which is the nearest exit
of the zone, and the alert is only recorded.

**D4. Population of prediction C.** All post-`tau` episodes of paired seeds. A drift never alerted gives a delay
censored on the right at the end of the run. Agents are compared by the median delay with a bootstrap
confidence interval. This replaces "failure episodes only" in the table of section 3, which is biased: the better
agent fails less and is compared on harder cases.

**D5. Repeated failure (prediction E).** The number of failures after the first failure of the run.

**First measurement of D1 (2026-11-22, 6 validation seeds, route margin 1.0).** Not met: R2c, R2l and R3l reach 15%, R3d
and R3c do not (R3c -0.8%). Causes and numbers in [agents.md](agents.md), section 10; the exact data are archived.

**Measurement under the amended baseline (DEV-1, `margin_frac` 0.7, same 6 seeds).** Met: R2c +87%, R3c +58%, R2l +29%,
R3l +22% (R2d +4%, R3d +1.5%). Numbers in [agents.md](agents.md), section 11. This is a post hoc amendment, declared as such
in [protocol_deviations.md](protocol_deviations.md). **Confirmed on the 60 validation seeds (2026-11-23): D1 is met**
(R2c +87%, R3c +59%, R2l +30%, R3l +23%; R2d +5%, R3d +1.6%; [agents.md](agents.md), section 11.1). The final conclusions
are judged on the test seeds only.

**D6. Regimes are read by family (2026-11-22).** R3d stays as a regime of damage without mortality: harmful (+15%
damage) but absorbed by the nominal controller (`Ccum` 1.6 against `D` = 2.98). On R2d, R2c, R2l, R3c and R3l the main
judge is the irreversible failure rate (with damage conditional on success and survival); on R3d, R0 and R1 the
judges are cumulative damage, revision cost and false alarms, since failure is zero by construction.

**D7. Reading rule for `Ccum` (2026-11-22).** `Ccum` is measured over the complete episode, never on a window. It is
never interpreted without the failure and success rates, which come first in every table; `Ccum` is reported over the
successful episodes next to the all-episode mean, because a failure caps it at about `D` and a timeout truncates it.
Details in [protocol.md](protocol.md), section 3.1.

**D8. Statistics and seeds (2026-11-22).** 60 paired seeds for validation and for test (more than 50), 5,000 bootstrap
replications over seeds, Holm correction over the announced comparisons, one tuning budget for every tunable agent,
test seeds opened once through a recorded ledger. Details in [protocol.md](protocol.md), sections 3.2 to 3.4.

**Amendment of the baseline (DEV-1, 2026-11-22).** After the first measurement of D1, the route rule received a safety
buffer (`margin_frac` 0.7, for every agent). See [protocol_deviations.md](protocol_deviations.md).
