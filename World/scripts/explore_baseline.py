"""Frozen baseline agent (learned model, never updated) in the drift regimes: Ccum and failures before/after tau.
Usage: PYTHONPATH=. python scripts/explore_baseline.py model.pt n_seeds regime [regime ...]"""
import sys

import numpy as np

from toyworld.agent import BaselineAgent
from toyworld.evaluate import run_frozen
from toyworld.model import WorldModel
from toyworld.seeds import VALIDATION_SEEDS

model = WorldModel.load(sys.argv[1])
seeds = VALIDATION_SEEDS[:int(sys.argv[2])]
for reg in sys.argv[3:]:
    rows = run_frozen(reg, seeds=seeds, n_episodes=20, policy_factory=lambda: BaselineAgent(model))
    pre = [r for r in rows if r["tau_ep"] is None or r["episode"] < r["tau_ep"]]
    post = [r for r in rows if r["tau_ep"] is not None and r["episode"] >= r["tau_ep"]]
    f = lambda rs: (np.mean([x["ccum"] for x in rs]), np.mean([x["outcome"] == "failure" for x in rs]),
                    np.mean([x["outcome"] == "success" for x in rs]), np.mean([x["steps"] for x in rs]))
    a = f(pre)
    line = f"{reg:4s} before: Ccum {a[0]:.2f} fail {a[1]:.2f} success {a[2]:.2f} steps {a[3]:.0f}"
    if post:
        b = f(post)
        line += f" | after: Ccum {b[0]:.2f} fail {b[1]:.2f} success {b[2]:.2f} steps {b[3]:.0f}"
    print(line, flush=True)
