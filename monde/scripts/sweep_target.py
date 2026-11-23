"""Sensitivity of one regime to its final value, with the frozen baseline agent (checklist 6.4 preview).
Patches the target of the regime, runs 6 validation seeds x 20 episodes, prints harm and masking by relative episode.
Usage: PYTHONPATH=. python scripts/sweep_target.py model.pt regime target [n_seeds]"""
import sys

import numpy as np

import verify_regimes_baseline as v
from toyworld import regimes
from toyworld.seeds import VALIDATION_SEEDS

path, reg, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 6
cls, k_star, form, _, frac = regimes.REGIMES[reg]
regimes.REGIMES[reg] = (cls, k_star, form, target, frac)
v._init(path)
recs = [r for s in VALIDATION_SEEDS[:n_seeds] for r in v._job((reg, s))]
base = 1.51                                             # R0 mean Ccum of the baseline (scripts/verify_regimes_baseline.py)
post = [r for r in recs if r["episode"] >= r["tau_ep"]]
print(f"{reg} target {target}: after tau Ccum {np.mean([r['ccum'] for r in post]):.2f} "
      f"({np.mean([r['ccum'] for r in post]) / base:.2f}x), failure {np.mean([r['outcome'] == 'failure' for r in post]):.2f}")
for rel in range(0, 11):
    at = [r for r in recs if r["episode"] - r["tau_ep"] == rel]
    ks = v.K_STAR.get(reg)
    print(f"   rel {rel:2d}: Ccum {np.mean([r['ccum'] for r in at]) / base:.2f}x fail {np.mean([r['outcome'] == 'failure' for r in at]):.2f} "
          f"excess {'k* ' + format(v.excess(at, ks), '.2f') if ks is not None else format(v.excess(at), '.2f')}")
