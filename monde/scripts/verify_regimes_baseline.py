"""Checklist 1.3 with the frozen baseline agent: does each regime do what it says (X = 10% harm, masking)?
Runs the 8 regimes on validation seeds (20 episodes per run), in parallel, and writes per-episode records
to results/regimes_baseline.json, then prints a summary.
Usage: PYTHONPATH=. python scripts/verify_regimes_baseline.py model.pt [n_seeds] [workers]"""
import json
import multiprocessing as mp
import sys

import numpy as np

from toyworld.evaluate import run_frozen
from toyworld.seeds import VALIDATION_SEEDS

REGIMES = ["R0", "R1", "R2d", "R2c", "R2l", "R3d", "R3c", "R3l"]
K_STAR = {"R1": 2, "R2d": 0, "R2c": 0, "R3d": 0, "R3c": 0}
MODEL = None


def _init(path):
    global MODEL
    from toyworld.model import WorldModel
    MODEL = WorldModel.load(path)


def _job(args):
    regime, seed = args
    from toyworld.agent import BaselineAgent
    rows = run_frozen(regime, seeds=[seed], n_episodes=20, policy_factory=lambda: BaselineAgent(MODEL))
    out = []
    for r in rows:
        zone = {k: [float((np.asarray(a) ** 2).sum()) for a, _ in v] for k, v in r["resid_zone"].items()}
        out.append({"regime": regime, "seed": seed, "episode": r["episode"], "tau_ep": r["tau_ep"],
                    "ccum": r["ccum"], "outcome": r["outcome"], "steps": r["steps"], "route": r["route"],
                    "r2_all": float((r["resid_all"] ** 2).sum()), "n_all": int(len(r["resid_all"])),
                    "r2_zone": {k: float(sum(v)) for k, v in zone.items()},
                    "n_zone": {k: len(v) for k, v in zone.items()}})
    return out


def excess(recs, zone=None):
    if zone is None:
        s, n = sum(r["r2_all"] for r in recs), sum(r["n_all"] for r in recs)
    else:
        s, n = sum(r["r2_zone"][zone] for r in recs), sum(r["n_zone"][zone] for r in recs)
    return float(np.sqrt(max(0.0, s / max(n, 1) - 2.0))) if n else float("nan")


if __name__ == "__main__":
    path = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    jobs = [(reg, s) for reg in REGIMES for s in VALIDATION_SEEDS[:n_seeds]]
    with mp.Pool(workers, initializer=_init, initargs=(path,)) as pool:
        out = pool.map(_job, jobs, chunksize=1)
    recs = [r for o in out for r in o]
    json.dump(recs, open("results/regimes_baseline.json", "w"))
    base = np.mean([r["ccum"] for r in recs if r["regime"] == "R0"])
    print(f"R0 mean Ccum {base:.2f}; failure {np.mean([r['outcome'] == 'failure' for r in recs if r['regime'] == 'R0']):.2f}")
    for reg in REGIMES[1:]:
        rs = [r for r in recs if r["regime"] == reg]
        post = [r for r in rs if r["episode"] >= r["tau_ep"]]
        print(f"\n{reg}: after tau Ccum {np.mean([r['ccum'] for r in post]):.2f} ({np.mean([r['ccum'] for r in post]) / base:.2f}x R0), "
              f"failure {np.mean([r['outcome'] == 'failure' for r in post]):.2f}, "
              f"route A {np.mean([r['route'] == 'A' for r in post]):.2f}, excess all {excess(post):.2f}"
              + (f", excess in k* {excess(post, K_STAR[reg]):.2f}" if reg in K_STAR else ""))
        for rel in range(0, 11):
            at = [r for r in rs if r["episode"] - r["tau_ep"] == rel]
            if at:
                print(f"   rel {rel:2d}: Ccum {np.mean([r['ccum'] for r in at]):.2f} ({np.mean([r['ccum'] for r in at]) / base:.2f}x) "
                      f"fail {np.mean([r['outcome'] == 'failure' for r in at]):.2f} "
                      + (f"excess k* {excess(at, K_STAR[reg]):.2f}" if reg in K_STAR else f"excess {excess(at):.2f}"))
