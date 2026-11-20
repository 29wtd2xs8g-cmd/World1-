"""Measure C1 (damage of a direct crossing through opening A, no drift) and propose D = 2 * C1.
Run once, then copy D into monde/config.py before any experiment (checklist 0.2).
Usage: PYTHONPATH=. python scripts/calibrate.py"""
import numpy as np
from World import WorldConfig
from World.evaluate import run_frozen

rows = run_frozen("R0", seeds=range(1000, 1050), n_episodes=4, route="A", config=WorldConfig(D=1e9))
ok = [r["ccum"] for r in rows if r["outcome"] == "success"]
c1 = float(np.mean(ok))
print(f"successful episodes: {len(ok)}/{len(rows)}")
print(f"C1 = {c1:.4f} damage units (std {np.std(ok):.4f}, 95% CI +-{1.96*np.std(ok)/np.sqrt(len(ok)):.4f})")
print(f"D  = 2*C1 = {2*c1:.4f} damage units")
