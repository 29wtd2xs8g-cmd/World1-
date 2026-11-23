"""Evaluation rules (checklist 3.1 and 3.4; docs/protocol.md). Every function reads per-episode records as produced by
`adaptive.run_agent` (keys: regime, seed, episode, tau_ep, ccum, outcome, steps, ...).

Reading rules, fixed before the final study:
  1. `ccum` is the damage over the COMPLETE episode, from reset to its end (success, failure, timeout or retreat);
     never a window after an alert or a detection.
  2. Outcomes come first. A failure caps `ccum` at about D and a timeout truncates it, so `ccum` alone makes an agent
     that dies early look thrifty. Every table lists failure and success rates before any `ccum`, and `ccum` is
     reported over the successful episodes (`ccum_success`) next to the all-episode mean.
  3. Regimes are read by family: where failure is possible (`LETHAL`), the failure rate is the main judge; where it is
     zero by construction (`DAMAGE_ONLY`), the judges are damage, revision cost and false alarms.
"""
import numpy as np

from .seeds import BOOTSTRAP_REPS

OUTCOMES = ("success", "failure", "timeout", "retreat")
LETHAL = ("R2d", "R2c", "R2l", "R3c", "R3l")
DAMAGE_ONLY = ("R0", "R1", "R3d")


def family(regime):
    if regime in LETHAL:
        return "lethal"
    if regime in DAMAGE_ONLY:
        return "damage_only"
    raise ValueError(f"unknown regime: {regime}")


def post_tau(rows):
    """Post-drift episodes of a run (the episode of onset included)."""
    return [r for r in rows if r["tau_ep"] is not None and r["episode"] >= r["tau_ep"]]


def outcome_table(rows):
    """Outcome rates first, then `ccum` over all episodes and over the successful ones."""
    n = len(rows)
    out = {"n": n}
    for k in OUTCOMES:
        out[k] = float(np.mean([r["outcome"] == k for r in rows])) if n else float("nan")
    ok = [r for r in rows if r["outcome"] == "success"]
    out["ccum_all"] = float(np.mean([r["ccum"] for r in rows])) if n else float("nan")
    out["ccum_success"] = float(np.mean([r["ccum"] for r in ok])) if ok else float("nan")
    out["steps_success"] = float(np.mean([r["steps"] for r in ok])) if ok else float("nan")
    return out


def run_totals(rows):
    """One run (episodes of one seed): what the whole run cost, so that a retreat-and-retry never looks free."""
    return {"ccum_run": float(sum(r["ccum"] for r in rows)), "steps_run": int(sum(r["steps"] for r in rows)),
            "failures": int(sum(r["outcome"] == "failure" for r in rows)),
            "retreats": int(sum(r["outcome"] == "retreat" for r in rows)),
            "revisions": int(sum(bool(r.get("revised")) for r in rows))}


def recovery(rows):
    """Cost of getting back to the goal after the drift: episodes, steps and damage from the onset episode up to and
    including the first success. `censored` is True when the run ends without a success after the onset."""
    ep = steps = 0
    dmg = 0.0
    for r in post_tau(rows):
        ep, steps, dmg = ep + 1, steps + r["steps"], dmg + r["ccum"]
        if r["outcome"] == "success":
            return {"episodes": ep, "steps": steps, "ccum": dmg, "censored": False}
    return {"episodes": ep, "steps": steps, "ccum": dmg, "censored": True}


def repeated_failures(rows):
    """Decision D5: the number of failures after the first failure of the run."""
    f = sum(r["outcome"] == "failure" for r in rows)
    return max(0, f - 1)


def paired_bootstrap(a, b=None, reps=BOOTSTRAP_REPS, seed=0):
    """Paired bootstrap over seeds. `a`, `b`: one value per seed, same seed order (b = None: `a` is already the
    per-seed difference). Returns the mean difference a - b, its 95% percentile interval and a two-sided p-value."""
    d = np.asarray(a, float) - (0.0 if b is None else np.asarray(b, float))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2.0 * min((means <= 0).mean(), (means >= 0).mean()))
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(p), "n": len(d), "reps": reps}


def paired_bootstrap_stat(a, b, stat, reps=BOOTSTRAP_REPS, seed=0):
    """Paired bootstrap over seeds of any statistic `stat(a_sample, b_sample)` (a median difference, a relative gap).
    The null value is 0. Returns the statistic, its 95% percentile interval and a two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (reps, len(a)))
    vals = np.array([stat(a[i], b[i]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2.0 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {"value": float(stat(a, b)), "lo": float(lo), "hi": float(hi), "p": float(p), "n": len(a), "reps": reps}


def detection(rows):
    """Decision D4 (prediction C): delay of the first alert after the drift onset, in environment steps counted from
    the start of the onset episode (the drift acts from step 0 of that episode). A run with no alert after the onset
    gets a delay censored on the right at the end of the run (`censored` True). None when the run has no onset."""
    elapsed = 0
    post = post_tau(rows)
    if not post:
        return None
    for r in post:
        if r["alert_step"] is not None:
            return {"delay": elapsed + int(r["alert_step"]), "censored": False}
        elapsed += int(r["steps"])
    return {"delay": elapsed, "censored": True}


def false_alarms(rows):
    """(alerts, episodes) over the episodes where an alert is a false alarm: every episode of R0 and R1 (nothing
    harmful to detect) and the episodes before the onset in the other regimes."""
    if rows and rows[0]["regime"] in ("R0", "R1"):
        sel = rows
    else:
        sel = [r for r in rows if r["tau_ep"] is not None and r["episode"] < r["tau_ep"]]
    return int(sum(r["alert_step"] is not None for r in sel)), len(sel)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order (family of the comparisons announced in 0.2)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def tuning_score(rows):
    """The scalar every tunable agent minimises on the tuning budget (same for all agents, `tuning.py`):
    failure rate over the post-drift episodes of the lethal regimes, plus the retreat rate over the episodes of the
    regimes without harm (R0, R1), where every alert is a false alarm. Equal weights; lower is better."""
    lethal = [r for r in rows if r["regime"] in LETHAL and r["tau_ep"] is not None and r["episode"] >= r["tau_ep"]]
    calm = [r for r in rows if r["regime"] in ("R0", "R1")]
    fail = np.mean([r["outcome"] == "failure" for r in lethal]) if lethal else 0.0
    false_alarm = np.mean([r["outcome"] == "retreat" for r in calm]) if calm else 0.0
    return float(fail + false_alarm)
