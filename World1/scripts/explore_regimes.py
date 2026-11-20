"""Explore the 8 regimes with the frozen agent: Ccum and failures before/after the drift.
Usage: PYTHONPATH=. python scripts/explore_regimes.py [n_seeds]"""
import sys
import numpy as np
from monde.evaluate import run_frozen

SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
for reg in ["R0", "R1", "R2d", "R2c", "R2l", "R3d", "R3c", "R3l"]:
    rows = run_frozen(reg, seeds=SEEDS, n_episodes=20)
    pre = [r for r in rows if r["tau_ep"] is None or r["episode"] < r["tau_ep"]]
    post = [r for r in rows if r["tau_ep"] is not None and r["episode"] >= r["tau_ep"]]
    f = lambda rs: (np.mean([x["ccum"] for x in rs]), np.mean([x["outcome"] == "failure" for x in rs]),
                    np.mean([x["outcome"] == "success" for x in rs]), np.mean([x["steps"] for x in rs]))
    a = f(pre)
    line = f"{reg:4s} before: Ccum {a[0]:.2f} fail {a[1]:.2f} success {a[2]:.2f} steps {a[3]:.0f}"
    if post:
        b = f(post)
        line += f" | after: Ccum {b[0]:.2f} fail {b[1]:.2f} success {b[2]:.2f} steps {b[3]:.0f}"
    print(line)
