"""Measure C1 (damage of one direct crossing through opening A, no drift) and propose D = 2 * C1.
Run once, then copy D into toyworld/config.py before any experiment (checklist 0.2).

Reference agent: the baseline agent (docs/agents.md) when a model file is given, the scripted crosser otherwise.
No failure cutoff (D = 1e9), calibration seeds only.
Usage: PYTHONPATH=. python scripts/calibrate.py [model.pt] [n_episodes_per_seed] [workers]"""
import multiprocessing as mp
import sys

import numpy as np

from toyworld import WorldConfig
from toyworld.evaluate import run_frozen
from toyworld.seeds import CALIBRATION_SEEDS

MODEL = None


def _init(path):
    global MODEL
    if path:
        from toyworld.model import WorldModel
        MODEL = WorldModel.load(path)


def _one_seed(args):
    seed, n_episodes = args
    factory = None
    if MODEL is not None:
        from toyworld.agent import BaselineAgent
        factory = lambda: BaselineAgent(MODEL)
    rows = run_frozen("R0", seeds=[seed], n_episodes=n_episodes, config=WorldConfig(D=1e9), policy_factory=factory)
    return [(r["outcome"], r["ccum"], r["route"]) for r in rows]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "scripted" else None
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    jobs = [(s, n_ep) for s in CALIBRATION_SEEDS]
    if workers > 1:
        with mp.Pool(workers, initializer=_init, initargs=(path,)) as pool:
            out = pool.map(_one_seed, jobs)
    else:
        _init(path)
        out = [_one_seed(j) for j in jobs]
    rows = [r for seed_rows in out for r in seed_rows]
    ok = [c for o, c, _ in rows if o == "success"]
    routes = {k: sum(1 for _, _, r in rows if r == k) for k in ("A", "B")}
    c1 = float(np.mean(ok))
    print(f"agent: {'baseline ' + path if path else 'scripted crosser'}; routes {routes}")
    print(f"successful episodes: {len(ok)}/{len(rows)}")
    print(f"C1 = {c1:.4f} damage units (std {np.std(ok):.4f}, 95% CI +-{1.96 * np.std(ok) / np.sqrt(len(ok)):.4f})")
    print(f"D  = 2*C1 = {2 * c1:.4f} damage units")
