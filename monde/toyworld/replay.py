"""Detector-level replay at matched false-alarm rates (decision D9; docs/protocol.md, section 3.6).

Prediction C compares how fast the two detectors notice a harmful drift. In the closed-loop runs of
`scripts/compare_agents.py` the two agents sit at very different false-alarm rates (their tuned thresholds correspond to about
10% and 1% of nominal episodes), and a detector that fires more easily is faster by construction. So the delay is compared
here at EQUAL false-alarm rates, on the same trajectories:

  1. the frozen agent plays each regime and seed once, with a `Recorder` (it never alerts and never revises, so its
     trajectory does not depend on any detector);
  2. both statistics are computed along every recorded episode, and the first step at which each exceeds each of the six
     threshold-grid values of `protocol/nominal_reference.json` is kept. The grid is defined by the nominal false-alarm rate
     (1, 2, 5, 10, 20, 40% of nominal episodes), so level i has the same nominal rate for both agents;
  3. the delay, the censoring and the false alarms are then read with the same functions as the closed-loop runs
     (`metrics.detection`, `metrics.false_alarms`), at every level.

The verdict of prediction C is read at the level of `VERDICT_RATE`; the other levels give the delay-against-false-alarm curve.
Nothing here uses the tuned thresholds, the tuning seeds or the test seeds.
"""
import json
from pathlib import Path

import numpy as np

from .compare import AGENTS, COMPARISON_SEEDS, HARMFUL, holm_adjust
from .detectors import CusumTrigger, HarmExcessTrigger
from .metrics import LETHAL, detection, false_alarms, paired_bootstrap_stat
from .nominal import FALSE_ALARM_RATES, NominalReference

REPLAY_REGIMES = ("R0", "R1", "R2d", "R2c", "R2l", "R3d", "R3c", "R3l")
VERDICT_RATE = 0.05
VERDICT_LEVEL = FALSE_ALARM_RATES.index(VERDICT_RATE)

# D9, prediction C (docs/protocol.md, section 3.6)
C_REGIMES = LETHAL                 # R2d, R2c, R2l, R3c, R3l; R3d is reported, not judged
C_MIN_DELAY_STEPS = 10.0           # Agent 1 median delay at least this much shorter than Agent 2's
C_MIN_REGIMES = 3                  # the prediction is supported if the criterion holds in at least 3 of the 5
C_MAX_CENSORED = 0.5               # a median is censored when at least half of the runs never alerted
C_FALSE_ALARM_TOLERANCE = 0.05     # matched false-alarm rates: measured rates within this, in R0 and in R1
ALPHA = 0.05


def grid(ref: NominalReference):
    return {"agent1_alert": list(ref.agent1_thresholds), "agent2_cusum": list(ref.agent2_thresholds)}


def _statistic_trigger(agent, ref: NominalReference):
    """The statistic of an agent with no threshold (it never alerts)."""
    if agent == "agent1_alert":
        return HarmExcessTrigger()
    if agent == "agent2_cusum":
        return CusumTrigger(ref.sigma_v, ref.sigma_n)
    raise ValueError(agent)


def first_crossings(trigger, episode, thresholds):
    """For each threshold, the step `rec.t` of the first record of `episode` whose statistic exceeds it (None if never).
    Same rule as `StatisticTrigger.step`: alert when the statistic is strictly above the threshold."""
    trigger.reset_episode()
    out = [None] * len(thresholds)
    for rec in episode:
        s = trigger.statistic(rec)
        for j, th in enumerate(thresholds):
            if out[j] is None and s > th:
                out[j] = int(rec.t)
    return out


def replay_run(rows, episodes, ref: NominalReference):
    """`rows`: per-episode rows of a frozen run; `episodes`: its recorded `StepRecord`s. Returns what is stored."""
    if len(rows) != len(episodes):
        raise ValueError("rows and recorded episodes differ in length")
    th = grid(ref)
    cross = {a: [first_crossings(_statistic_trigger(a, ref), ep, th[a]) for ep in episodes] for a in AGENTS}
    return {"rows": rows, "cross": cross}


def load_replay(path):
    """(regime, seed) -> {"rows", "cross"}, from a jsonl written by `scripts/replay_detectors.py`."""
    out = {}
    path = Path(path)
    if path.exists():
        for r in map(json.loads, path.read_text().splitlines()):
            out[(r["regime"], r["seed"])] = {"rows": r["rows"], "cross": r["cross"]}
    return out


def rows_at(entry, agent, level):
    """The frozen rows of a run, with the alerts that `agent` would have raised at grid level `level`."""
    return [{**r, "alert_step": c[level]} for r, c in zip(entry["rows"], entry["cross"][agent])]


def level_vectors(replay, agent, regime, level, seeds):
    delays, censored, fa, n = [], [], 0, 0
    for s in seeds:
        rows = rows_at(replay[(regime, s)], agent, level)
        if regime in HARMFUL:
            d = detection(rows)
            delays.append(d["delay"])
            censored.append(float(d["censored"]))
        a, k = false_alarms(rows)
        fa, n = fa + a, n + k
    return {"delay": np.array(delays, float), "censored": np.array(censored, float), "fa": fa, "fa_n": n}


def matched_contrast(replay, regime, level, seeds, reps=5000):
    """Agent 1 against Agent 2 on one regime at one grid level, paired over seeds (positive = Agent 1 faster)."""
    va = level_vectors(replay, "agent1_alert", regime, level, seeds)
    vb = level_vectors(replay, "agent2_cusum", regime, level, seeds)
    rate = lambda v: v["fa"] / v["fa_n"] if v["fa_n"] else float("nan")
    out = {"regime": regime, "level": level, "nominal_rate": FALSE_ALARM_RATES[level], "n": len(seeds),
           "false_alarm_a": rate(va), "false_alarm_b": rate(vb)}
    if regime in HARMFUL:
        gap = lambda x, y: float(np.median(y) - np.median(x))
        out.update(delay_a=float(np.median(va["delay"])), delay_b=float(np.median(vb["delay"])),
                   censored_a=float(va["censored"].mean()), censored_b=float(vb["censored"].mean()),
                   delay_gap=paired_bootstrap_stat(va["delay"], vb["delay"], gap, reps))
    return out


def c_regime_met(c, holm_p):
    """One lethal regime: median delay of Agent 1 at least 10 steps shorter, Holm-adjusted p below 0.05, and Agent 1's
    median not censored (at least half of its runs alerted). A censored median of Agent 2 alone only makes the gap a lower
    bound, so it does not disqualify the regime."""
    return bool(c["delay_gap"]["value"] >= C_MIN_DELAY_STEPS and holm_p < ALPHA and c["censored_a"] < C_MAX_CENSORED)


def verdict_c(contrasts):
    """Prediction C (D9). `contrasts`: matched contrasts at the verdict level, for the lethal regimes and for R0 and R1.
    Supported iff the criterion holds in at least 3 of the 5 lethal regimes (Holm over the five) and the measured false-alarm
    rates of the two agents are within 5 points of each other in R0 and in R1."""
    lethal = [c for c in contrasts if c["regime"] in C_REGIMES]
    calm = [c for c in contrasts if c["regime"] in ("R0", "R1")]
    if {c["regime"] for c in lethal} != set(C_REGIMES) or {c["regime"] for c in calm} != {"R0", "R1"}:
        raise ValueError("verdict C needs the five lethal regimes and R0 and R1")
    adj = holm_adjust(lethal, "delay_gap", C_REGIMES)
    met = {c["regime"]: c_regime_met(c, adj[c["regime"]]) for c in lethal}
    matched = all(abs(c["false_alarm_b"] - c["false_alarm_a"]) <= C_FALSE_ALARM_TOLERANCE for c in calm)
    n_met = sum(met.values())
    return {"met": met, "holm_p": adj, "n_met": n_met, "false_alarms_matched": matched,
            "supported": bool(n_met >= C_MIN_REGIMES and matched)}


def complete_seeds(replay, seeds=COMPARISON_SEEDS):
    """The seeds of `seeds` that have every regime in the replay."""
    return [s for s in seeds if all((g, s) in replay for g in REPLAY_REGIMES)]
