"""Competence check of the baseline agent before drift (docs/agents.md, section 6; checklist 2.1).
Regime R0 on the validation seeds. Exit code 0 only if all four criteria hold.
Usage: PYTHONPATH=. python scripts/eval_baseline.py [model.pt] [n_seeds] [n_episodes]"""
import sys
import time

import numpy as np

from toyworld.agent import BaselineAgent
from toyworld.evaluate import run_frozen
from toyworld.model import WorldModel
from toyworld.seeds import VALIDATION_SEEDS

C1 = 2.5027
path = sys.argv[1] if len(sys.argv) > 1 else "results/baseline.pt"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30     # the 30 documented seeds
n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
model = WorldModel.load(path)
t0 = time.time()
rows = run_frozen("R0", seeds=VALIDATION_SEEDS[:n_seeds], n_episodes=n_episodes,
                  policy_factory=lambda: BaselineAgent(model))
succ = np.mean([r["outcome"] == "success" for r in rows])
fail = np.mean([r["outcome"] == "failure" for r in rows])
ccum = np.mean([r["ccum"] for r in rows])
routeA = np.mean([r["route"] == "A" for r in rows])
steps = np.mean([r["steps"] for r in rows])
checks = {"success >= 95%": succ >= 0.95, "failure <= 1%": fail <= 0.01,
          "mean Ccum <= 1.1*C1": ccum <= 1.1 * C1, "route A >= 95%": routeA >= 0.95}
print(f"{len(rows)} episodes in {time.time() - t0:.0f} s | success {succ:.1%} failure {fail:.1%} "
      f"Ccum {ccum:.2f} (limit {1.1 * C1:.2f}) route A {routeA:.1%} steps {steps:.0f}")
for k, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + k)
sys.exit(0 if all(checks.values()) else 1)
