"""Tune the threshold of Agent 1 and Agent 2 under the same budget (docs/protocol.md, section 3.3).

Each agent tries the six thresholds of its nominal grid (`protocol/nominal_reference.json`) on the tuning budget
(`toyworld/tuning.py`: validation seeds 2030-2039, regimes R0, R1, R2c, R3c, 20 episodes), scored by
`metrics.tuning_score`. The chosen thresholds go to `protocol/trigger_thresholds.json`; every tuning is logged in
`protocol/tuning_ledger.jsonl`. An agent can be tuned once only.

  PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/tune_triggers.py results/baseline.pt [workers]
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

from toyworld.adaptive import run_agent
from toyworld.detectors import CusumTrigger, HarmExcessTrigger
from toyworld.metrics import tuning_score
from toyworld.model import WorldModel
from toyworld.nominal import NominalReference, model_hash
from toyworld.tuning import BUDGET, LEDGER, tune

OUT = Path("protocol/trigger_thresholds.json")
MODEL, REF = None, None


def _init(path):
    global MODEL, REF
    MODEL, REF = WorldModel.load(path), NominalReference.load()


def make_trigger(agent, threshold):
    if agent == "agent1_alert":
        return HarmExcessTrigger(threshold)
    return CusumTrigger(REF.sigma_v, REF.sigma_n, threshold)


def _job(args):
    agent, threshold, regime, seed, n_ep = args
    return run_agent(regime, seed, MODEL, make_trigger(agent, threshold), n_episodes=n_ep)


def main(path, workers):
    ref = NominalReference.load()
    if ref.model_sha256 != model_hash(path):
        sys.exit("the nominal reference was computed on another model: rerun scripts/calibrate_triggers.py")
    done = {json.loads(l)["agent"] for l in LEDGER.read_text().splitlines() if l.strip()} if LEDGER.exists() else set()
    grids = {"agent1_alert": ref.agent1_thresholds, "agent2_cusum": ref.agent2_thresholds}
    if done & set(grids):
        sys.exit(f"already tuned: {sorted(done & set(grids))}. An agent is tuned once; see protocol/tuning_ledger.jsonl")
    chosen = json.loads(OUT.read_text()) if OUT.exists() else {}
    with mp.Pool(workers, initializer=_init, initargs=(path,)) as pool:
        for agent, grid in grids.items():
            def evaluate(threshold, seeds, regimes, n_ep, agent=agent):
                jobs = [(agent, threshold, g, s, n_ep) for s in seeds for g in regimes]
                rows = [r for run in pool.map(_job, jobs) for r in run]
                score = tuning_score(rows)
                print(f"{agent} threshold {threshold:9.4f} score {score:.3f}", flush=True)
                return score
            best, scores = tune(agent, grid, evaluate, BUDGET)
            chosen[agent] = {"threshold": best, "rate": ref.rates[grid.index(best)], "scores": scores, "grid": grid}
            print(f"{agent}: chosen threshold {best:.4f} (nominal false-alarm rate {chosen[agent]['rate']})", flush=True)
    OUT.write_text(json.dumps(chosen, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
