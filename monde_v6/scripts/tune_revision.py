"""Choice of the revision budget (decision D2), on the CALIBRATION seeds only (never on validation/test seeds).
Runs the always-adapting agent for a few (steps, lr) settings on abrupt drifts and prints the post-tau damage.
Usage: PYTHONPATH=. python scripts/tune_revision.py model.pt [workers]"""
import multiprocessing as mp
import sys

import numpy as np

from toyworld.adaptive import run_agent
from toyworld.regimes import onset_episode
from toyworld.revision import RevisionParams
from toyworld.seeds import CALIBRATION_SEEDS
from toyworld.triggers import AlwaysTrigger, NeverTrigger

MODEL = None
SETTINGS = [None, (50, 1e-3), (150, 1e-3), (150, 3e-3)]      # None = frozen
REGIMES = ["R2d", "R2c"]
SEEDS = CALIBRATION_SEEDS[:2]
N_EP = 10


def _init(path):
    global MODEL
    from toyworld.model import WorldModel
    MODEL = WorldModel.load(path)


def _job(args):
    setting, regime, seed = args
    if setting is None:
        rows = run_agent(regime, seed, MODEL, NeverTrigger(), n_episodes=N_EP)
    else:
        rows = run_agent(regime, seed, MODEL, AlwaysTrigger(), n_episodes=N_EP,
                         revision=RevisionParams(steps=setting[0], lr=setting[1]))
    return setting, regime, seed, rows


if __name__ == "__main__":
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    jobs = [(s, r, sd) for s in SETTINGS for r in REGIMES for sd in SEEDS]
    with mp.Pool(workers, initializer=_init, initargs=(sys.argv[1],)) as pool:
        out = pool.map(_job, jobs, chunksize=1)
    for s in SETTINGS:
        for r in REGIMES:
            post = [e for st, rg, sd, rows in out if st == s and rg == r for e in rows if e["episode"] >= e["tau_ep"]]
            lb = [e["loss_before"] for st, rg, sd, rows in out if st == s and rg == r for e in rows if e["loss_before"]]
            la = [e["loss_after"] for st, rg, sd, rows in out if st == s and rg == r for e in rows if e["loss_after"]]
            print(f"{str(s):14s} {r}: post-tau Ccum {np.mean([e['ccum'] for e in post]):.2f}, "
                  f"failure {np.mean([e['outcome'] == 'failure' for e in post]):.2f}, route A {np.mean([e['route'] == 'A' for e in post]):.2f}"
                  + (f", window loss {np.mean(lb):.3f} -> {np.mean(la):.3f}" if lb else ""))
