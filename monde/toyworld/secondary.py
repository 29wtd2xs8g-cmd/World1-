"""Secondary analysis: non-completion of the task, next to the registered failure rate (decision D11).

The registered failure rate (prediction B) counts irreversible failures only. An agent can lower it by retreating instead of
crossing: in the validation comparison Agent 2 has fewer failures than Agent 1 in R2l and R3l, but 94% and 80% of its
post-onset episodes end in a retreat and only 2% and 20% in a success. The non-completion rate

    non-completion = 1 - success = failure + retreat + timeout

is read here, per regime, with the same paired bootstrap as the verdicts. It is a secondary analysis: it changes no verdict, it
does not redefine `failure`, and it is never used to decide prediction B or C. On the validation seeds it was computed after
the verdicts were read (post hoc). It is registered for the test seeds before their summary is read (docs/research.md, D11).
"""
import numpy as np

from .compare import AGENTS, REGIMES, holm_adjust, seed_vectors
from .metrics import LETHAL, paired_bootstrap

SECONDARY_REGIMES = ("R2d", "R2c", "R2l", "R3d", "R3c", "R3l")      # R3d has no mortality: reported, not in the Holm family
HOLM_FAMILY = LETHAL                                                # the five regimes where an irreversible failure exists


def complete_run_seeds(runs, required):
    """The seeds that have both agents on every regime; refuses to go on with fewer than `required` (no interim look)."""
    found = {s for (_, _, s) in runs}
    seeds = sorted(s for s in found if all((k, g, s) in runs for k in AGENTS for g in REGIMES))
    if len(seeds) < required:
        raise ValueError(f"{len(seeds)} complete seeds of {required} required: nothing is read before the study is complete")
    return seeds


def non_completion(runs, kind, regime, seeds):
    """One value per seed: the share of post-onset episodes that do not end in a success."""
    return 1.0 - seed_vectors(runs, kind, regime, seeds)["success"]


def secondary_contrast(runs, regime, seeds, reps=5000):
    """Agent 1 against Agent 2 on one regime, paired over seeds. Positive gap = Agent 1 completes the task more often."""
    a, b = AGENTS
    va, vb = seed_vectors(runs, a, regime, seeds), seed_vectors(runs, b, regime, seeds)
    nc_a, nc_b = 1.0 - va["success"], 1.0 - vb["success"]
    out = {"regime": regime, "n": len(seeds)}
    for tag, v, nc in (("a", va, nc_a), ("b", vb, nc_b)):
        out.update({f"failure_{tag}": float(v["failure"].mean()), f"retreat_{tag}": float(v["retreat"].mean()),
                    f"timeout_{tag}": float(v["timeout"].mean()), f"success_{tag}": float(v["success"].mean()),
                    f"noncompletion_{tag}": float(nc.mean())})
    out["noncompletion_gap"] = paired_bootstrap(nc_b, nc_a, reps)       # mean of (Agent 2 - Agent 1), positive = Agent 1 better
    return out


def secondary_table(runs, seeds, reps=5000):
    """Contrasts of the six harmful regimes and the Holm-adjusted p-values over the five lethal ones."""
    contrasts = [secondary_contrast(runs, g, seeds, reps) for g in SECONDARY_REGIMES]
    adj = holm_adjust(contrasts, "noncompletion_gap", HOLM_FAMILY)
    for c in contrasts:
        c["holm_p"] = adj.get(c["regime"])
    return contrasts


def reading(c):
    """One word per regime: which agent completes the task more often, at the 5% level (Holm) and by the sign of the gap."""
    if c["holm_p"] is None:
        return "not judged"
    g = c["noncompletion_gap"]
    if c["holm_p"] >= 0.05:
        return "no difference"
    return "Agent 1 completes more" if g["mean"] > 0 else "Agent 2 completes more"
