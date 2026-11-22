"""Nominal reference of Agents 1 and 2 (docs/agents.md, section 12): run the frozen agent in R0 on calibration seeds,
compute the residual scales and the six-candidate threshold grid of each agent, write protocol/nominal_reference.json.

  PYTHONPATH=. python scripts/calibrate_triggers.py results/baseline.pt [workers]
"""
import multiprocessing as mp
import sys

from toyworld.model import WorldModel
from toyworld.nominal import NOMINAL_EPISODES, NOMINAL_SEEDS, build_reference, model_hash, record_nominal

MODEL = None


def _init(path):
    global MODEL
    MODEL = WorldModel.load(path)


def _job(seed):
    return record_nominal(MODEL, seeds=[seed])


if __name__ == "__main__":
    path = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with mp.Pool(workers, initializer=_init, initargs=(path,)) as pool:
        parts = pool.map(_job, NOMINAL_SEEDS)
    episodes = [ep for part in parts for ep in part]
    ref = build_reference(None, episodes, model_hash(path))
    ref.save()
    print(f"{len(episodes)} nominal episodes, {sum(map(len, episodes))} steps")
    print(f"sigma_v {ref.sigma_v:.4f}  sigma_n {ref.sigma_n:.4f}")
    print("false-alarm rate  ", *[f"{r:8.2f}" for r in ref.rates])
    print("agent 1 threshold ", *[f"{t:8.4f}" for t in ref.agent1_thresholds])
    print("agent 2 threshold ", *[f"{t:8.3f}" for t in ref.agent2_thresholds])
