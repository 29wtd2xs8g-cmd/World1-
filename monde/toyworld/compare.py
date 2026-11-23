"""Comparison of Agent 1 and Agent 2 on validation seeds (checklist 4; docs/agents.md, section 13).

Pure functions: the seeds of the comparison, the guarded loading of the tuned thresholds, per-seed vectors, paired
contrasts with bootstrap intervals, and the literal verdicts of predictions B and C. `scripts/compare_agents.py` runs and
prints.

The comparison never uses the tuning seeds (they chose the thresholds) and never the test seeds.
"""
import json
from pathlib import Path

import numpy as np

from .detectors import CusumTrigger, HarmExcessTrigger
from .metrics import (LETHAL, detection, false_alarms, holm, outcome_table, paired_bootstrap_stat, post_tau)
from .nominal import NominalReference
from .seeds import VALIDATION_SEEDS
from .tuning import BUDGET, LEDGER, check_equal_budgets

THRESHOLDS = Path(__file__).resolve().parent.parent / "protocol" / "trigger_thresholds.json"
COMPARISON_SEEDS = tuple(s for s in VALIDATION_SEEDS if s not in BUDGET.seeds)      # 50 seeds
REGIMES = ("R0", "R1", "R2d", "R2c", "R2l", "R3d", "R3c", "R3l")
HARMFUL = ("R2d", "R2c", "R2l", "R3d", "R3c", "R3l")
AGENTS = ("agent1_alert", "agent2_cusum")


def load_thresholds(path=THRESHOLDS, ledger=LEDGER):
    """The tuned thresholds. Refuses to continue unless both agents were tuned, once each, under the same budget."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: tune the agents first (scripts/tune_triggers.py)")
    chosen = json.loads(path.read_text())
    missing = [a for a in AGENTS if a not in chosen]
    if missing:
        raise ValueError(f"no tuned threshold for {missing}")
    if check_equal_budgets(ledger) is None:
        raise ValueError("the tuning ledger is empty")
    tuned = {json.loads(l)["agent"] for l in Path(ledger).read_text().splitlines() if l.strip()}
    if not set(AGENTS) <= tuned:
        raise ValueError(f"the tuning ledger does not list both agents (found {sorted(tuned)})")
    return {a: float(chosen[a]["threshold"]) for a in AGENTS}


def make_trigger(agent, thresholds, ref: NominalReference):
    if agent == "agent1_alert":
        return HarmExcessTrigger(thresholds[agent])
    if agent == "agent2_cusum":
        return CusumTrigger(ref.sigma_v, ref.sigma_n, thresholds[agent])
    raise ValueError(agent)


def load_runs(path):
    """(kind, regime, seed) -> rows, from a jsonl written by `scripts/compare_agents.py` or `scripts/headroom.py`."""
    runs = {}
    path = Path(path)
    if path.exists():
        for r in map(json.loads, path.read_text().splitlines()):
            runs[(r["kind"], r["regime"], r["seed"])] = r["rows"]
    return runs


def _scored(rows):
    """Episodes the outcome rates are read on: after the onset where there is one, every episode otherwise (R0)."""
    return post_tau(rows) or rows


def seed_vectors(runs, kind, regime, seeds):
    """One value per seed (same order as `seeds`) for the metrics of the comparison."""
    out = {"failure": [], "success": [], "timeout": [], "retreat": [], "ccum": [], "ccum_success": [], "delay": [],
           "censored": [], "fa": [], "fa_n": [], "revisions": []}
    for s in seeds:
        rows = runs[(kind, regime, s)]
        t = outcome_table(_scored(rows))
        out["failure"].append(t["failure"])
        out["success"].append(t["success"])
        out["timeout"].append(t["timeout"])
        out["retreat"].append(t["retreat"])
        out["ccum_success"].append(t["ccum_success"])
        out["ccum"].append(t["ccum_all"])
        d = detection(rows) if regime in HARMFUL else None
        out["delay"].append(np.nan if d is None else d["delay"])
        out["censored"].append(np.nan if d is None else float(d["censored"]))
        fa, n = false_alarms(rows)
        out["fa"].append(fa)
        out["fa_n"].append(n)
        out["revisions"].append(sum(bool(r.get("revised")) for r in rows))
    return {k: np.array(v, float) for k, v in out.items()}


def calm_false_alarm_gap(runs, a, b, seeds):
    """Agent `b` rate minus agent `a` rate of false alarms per episode, averaged over R0 and R1 (positive = `a` quieter)."""
    gaps = []
    for g in ("R0", "R1"):
        va, vb = seed_vectors(runs, a, g, seeds), seed_vectors(runs, b, g, seeds)
        gaps.append(vb["fa"].sum() / vb["fa_n"].sum() - va["fa"].sum() / va["fa_n"].sum())
    return float(np.mean(gaps))


def contrast(runs, a, b, regime, seeds, reps=5000):
    """Agent `a` against agent `b` on one regime, paired over seeds. Signs: positive = `a` better."""
    va, vb = seed_vectors(runs, a, regime, seeds), seed_vectors(runs, b, regime, seeds)
    mean_gap = lambda x, y: float(y.mean() - x.mean())
    rel_gap = lambda x, y: float(1.0 - x.mean() / y.mean()) if y.mean() > 0 else 0.0
    med_gap = lambda x, y: float(np.median(y) - np.median(x))
    fa_rate = lambda v: float(v["fa"].sum() / v["fa_n"].sum()) if v["fa_n"].sum() else float("nan")
    out = {"regime": regime, "a": a, "b": b, "n": len(seeds),
           "failure_a": float(va["failure"].mean()), "failure_b": float(vb["failure"].mean()),
           "ccum_a": float(va["ccum"].mean()), "ccum_b": float(vb["ccum"].mean()),
           "failure_gap": paired_bootstrap_stat(va["failure"], vb["failure"], mean_gap, reps),
           "ccum_gap_rel": paired_bootstrap_stat(va["ccum"], vb["ccum"], rel_gap, reps),
           "false_alarm_a": fa_rate(va), "false_alarm_b": fa_rate(vb),
           "revisions_a": float(va["revisions"].mean()), "revisions_b": float(vb["revisions"].mean())}
    if regime in HARMFUL:
        out.update(delay_a=float(np.median(va["delay"])), delay_b=float(np.median(vb["delay"])),
                   censored_a=float(va["censored"].mean()), censored_b=float(vb["censored"].mean()),
                   delay_gap=paired_bootstrap_stat(va["delay"], vb["delay"], med_gap, reps))
    return out


def holm_adjust(contrasts, key, regimes):
    """Holm-adjusted p-values of one family (e.g. the failure gaps over the lethal regimes), in the order of `regimes`."""
    sel = [c for c in contrasts if c["regime"] in regimes]
    adj = holm([c[key]["p"] for c in sel])
    return {c["regime"]: p for c, p in zip(sel, adj)}


# Provisional readings of the thresholds of predictions B and C (docs/research.md section 3). "X% lower" is read as a
# relative gap for the damage and as percentage points for the failure rate; decision D9 has to confirm it.
B_MIN_CCUM_REL, B_MIN_FAILURE_POINTS = 0.15, 0.10
C_MIN_DELAY_STEPS, C_MIN_FALSE_ALARM_POINTS = 10.0, 0.05


def verdict_b(c):
    """Prediction B, literal: mean Ccum at least 15% lower AND failure rate at least 10 points lower (lethal regimes).
    In a regime without mortality only the damage criterion applies."""
    ok_c = c["ccum_gap_rel"]["value"] >= B_MIN_CCUM_REL
    if c["regime"] not in LETHAL:
        return ok_c
    return ok_c and c["failure_gap"]["value"] >= B_MIN_FAILURE_POINTS


def verdict_c(c, calm_false_alarm_gap):
    """Prediction C, literal: median delay at least 10 steps shorter, with a false-alarm rate lower by at least 5 points
    (`calm_false_alarm_gap` = Agent 2 rate minus Agent 1 rate on R0 and R1)."""
    return (c["delay_gap"]["value"] >= C_MIN_DELAY_STEPS) and (calm_false_alarm_gap >= C_MIN_FALSE_ALARM_POINTS)
