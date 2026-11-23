"""Comparison of Agent 1 and Agent 2 on validation seeds (checklist 4; docs/agents.md, section 13).

Pure functions: the seeds of the comparison, the guarded loading of the tuned thresholds, per-seed vectors, paired
contrasts with bootstrap intervals, and the verdict of prediction B (decision D9). The verdict of prediction C is read at
matched false-alarm rates in `replay.py`. `scripts/compare_agents.py` runs and prints.

The comparison never uses the tuning seeds (they chose the thresholds) and never the test seeds.
"""
import json
from pathlib import Path

import numpy as np

from .detectors import CusumTrigger, HarmExcessTrigger
from .metrics import detection, false_alarms, holm, outcome_table, paired_bootstrap_stat, post_tau
from .nominal import REFERENCE, NominalReference, model_hash
from .seeds import VALIDATION_SEEDS
from .tuning import BUDGET, LEDGER, check_equal_budgets

THRESHOLDS = Path(__file__).resolve().parent.parent / "protocol" / "trigger_thresholds.json"
PROTOCOL = Path(__file__).resolve().parent.parent / "docs" / "protocol.md"
RESEARCH = Path(__file__).resolve().parent.parent / "docs" / "research.md"
D9_MARKER = "D9-STATUS: RECORDED"
D10_MARKER = "**D10. Confirmatory study on the test seeds, nothing changed"
STUDY_NAME = "final study: Agent 1 against Agent 2 on the test seeds (rules D9, decision D10)"
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


def require_d9_recorded(path=PROTOCOL):
    """The readings of predictions B and C (decision D9) must be written in protocol.md before any verdict is computed, so
    that no result can have shaped them. The line `D9-STATUS: RECORDED` of section 3.6 is the author's go."""
    path = Path(path)
    if not path.exists() or D9_MARKER not in path.read_text():
        raise RuntimeError("decision D9 is not recorded in docs/protocol.md (section 3.6): no summary before it is")


def require_d10_recorded(path=RESEARCH):
    """The final study is decided (D10) in docs/research.md before the test seeds are opened."""
    path = Path(path)
    if not path.exists() or D10_MARKER not in path.read_text():
        raise RuntimeError("decision D10 is not recorded in docs/research.md: the test seeds stay closed")


def study_purpose(model_path, thresholds_path=THRESHOLDS, reference_path=REFERENCE):
    """The text written in the test-seed ledger. It names the model, the tuned thresholds and the nominal reference the study
    runs with (short hashes), so that the ledger shows what was frozen when the test seeds were opened."""
    return (f"{STUDY_NAME}; model {model_hash(model_path)}; thresholds {model_hash(thresholds_path)}; "
            f"nominal reference {model_hash(reference_path)}")


def require_complete(seeds_found, required):
    """The final study is read once, on all its seeds: no interim look at partial results (it would be optional stopping)."""
    if len(seeds_found) != required:
        raise RuntimeError(f"incomplete: {len(seeds_found)} of {required} seeds have every job; no result is printed before the end")


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
           "success_a": float(va["success"].mean()), "success_b": float(vb["success"].mean()),
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


# Readings of the thresholds of prediction B (decision D9, docs/protocol.md section 3.6). "X% lower" is a relative gap for
# the damage and percentage points for the failure rate. Prediction C is read in `replay.py` (matched false-alarm rates).
B_REGIMES = ("R2c", "R3c")          # drift that acts on the cost only: the margin is what the alert is about
B_MIN_CCUM_REL, B_MIN_FAILURE_POINTS = 0.15, 0.10
B_MAX_SUCCESS_DROP = 0.05           # guard: Agent 1 may not succeed more than 5 points less often than Agent 2
ALPHA = 0.05


def b_regime_met(c, holm_p_ccum, holm_p_failure):
    """One regime of prediction B: mean Ccum (all post-onset episodes) at least 15% lower, failure rate at least 10 points
    lower, success rate not more than 5 points lower (retreating all the time lowers Ccum without solving the task), and both
    gaps significant after Holm correction over the regimes of B."""
    return bool(c["ccum_gap_rel"]["value"] >= B_MIN_CCUM_REL
                and c["failure_gap"]["value"] >= B_MIN_FAILURE_POINTS
                and c["success_a"] - c["success_b"] >= -B_MAX_SUCCESS_DROP
                and holm_p_ccum < ALPHA and holm_p_failure < ALPHA)


def verdict_b(contrasts):
    """Prediction B (D9): supported iff the criterion holds in both R2c and R3c (Holm over the two). The other regimes are
    reported without a verdict."""
    sel = [c for c in contrasts if c["regime"] in B_REGIMES]
    if {c["regime"] for c in sel} != set(B_REGIMES):
        raise ValueError("verdict B needs the contrasts of R2c and R3c")
    hc = holm_adjust(sel, "ccum_gap_rel", B_REGIMES)
    hf = holm_adjust(sel, "failure_gap", B_REGIMES)
    met = {c["regime"]: b_regime_met(c, hc[c["regime"]], hf[c["regime"]]) for c in sel}
    return {"met": met, "holm_ccum": hc, "holm_failure": hf, "supported": all(met.values())}
