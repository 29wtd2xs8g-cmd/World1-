"""Detector-level replay at matched false-alarm rates (decision D9; toyworld/replay.py).

  PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/replay_detectors.py run results/baseline.pt [workers] [n_seeds]
  PYTHONPATH=. python scripts/replay_detectors.py check
  PYTHONPATH=. python scripts/replay_detectors.py summary

`run` plays the frozen agent once per regime and seed (8 regimes x 50 comparison seeds = 400 runs) with a recorder, computes
both statistics along every episode, and stores the first threshold crossings at the six grid levels of
`protocol/nominal_reference.json` in `results/replay.jsonl` (resumable). It needs neither the tuned thresholds nor any
closed-loop result, and never touches the tuning or test seeds.
`check` compares the replay with the closed-loop runs of `results/compare.jsonl`: the frozen rows of R0 and R1 must be
identical, and the first episode of every closed-loop run must alert at the same step as the replay at the tuned threshold.
`summary` prints the delay of both detectors at every false-alarm level and the verdict of prediction C. It refuses to run
until decision D9 is recorded in docs/protocol.md.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from toyworld.adaptive import run_agent
from toyworld.compare import AGENTS, COMPARISON_SEEDS, HARMFUL, load_runs, load_thresholds, require_d9_recorded
from toyworld.detectors import Recorder
from toyworld.model import WorldModel
from toyworld.nominal import FALSE_ALARM_RATES, NominalReference, model_hash
from toyworld.replay import (C_REGIMES, REPLAY_REGIMES, VERDICT_LEVEL, VERDICT_RATE, complete_seeds, grid, load_replay,
                             matched_contrast, replay_run, rows_at, verdict_c)
from toyworld.seeds import BOOTSTRAP_REPS

OUT = Path("results/replay.jsonl")
COMPARE = Path("results/compare.jsonl")
MODEL = REF = None


def _init(path):
    global MODEL, REF
    MODEL, REF = WorldModel.load(path), NominalReference.load()


def _job(args):
    regime, seed = args
    rec = Recorder()
    rows = run_agent(regime, seed, MODEL, rec)
    return {"regime": regime, "seed": seed, **replay_run(rows, rec.episodes, REF)}


def run(path, workers=2, n_seeds=50):
    if NominalReference.load().model_sha256 != model_hash(path):
        sys.exit("the nominal reference was computed on another model")
    seeds = COMPARISON_SEEDS[:n_seeds]
    done = {(r["regime"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    jobs = [(g, s) for s in seeds for g in REPLAY_REGIMES if (g, s) not in done]
    OUT.parent.mkdir(exist_ok=True)
    with mp.Pool(workers, initializer=_init, initargs=(path,)) as pool, open(OUT, "a") as f:
        for res in pool.imap_unordered(_job, jobs, chunksize=1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print("done", res["regime"], res["seed"], flush=True)


def check():
    """Consistency of the replay with the closed-loop runs. Reads no outcome of the comparison."""
    replay, runs = load_replay(OUT), load_runs(COMPARE)
    th = load_thresholds()
    ref = NominalReference.load()
    level = {a: grid(ref)[a].index(th[a]) for a in AGENTS}            # the tuned thresholds are grid values
    bad_frozen = bad_first = n_frozen = n_first = 0
    for (kind, regime, seed), rows in runs.items():
        if (regime, seed) not in replay:
            continue
        mine = replay[(regime, seed)]
        if kind == "frozen":
            n_frozen += 1
            keys = ("outcome", "steps", "ccum")
            bad_frozen += any(a[k] != b[k] for a, b in zip(rows, mine["rows"]) for k in keys) or len(rows) != len(mine["rows"])
        elif kind in AGENTS:
            n_first += 1
            bad_first += rows[0]["alert_step"] != rows_at(mine, kind, level[kind])[0]["alert_step"]
    print(f"frozen runs identical to the replay: {n_frozen - bad_frozen}/{n_frozen}")
    print(f"first-episode alerts identical at the tuned threshold: {n_first - bad_first}/{n_first}")
    if bad_frozen or bad_first:
        sys.exit("INCONSISTENT: the replay does not reproduce the closed-loop runs")


def _fmt(g):
    return f"{g['value']:+7.1f} [{g['lo']:+7.1f}, {g['hi']:+7.1f}]"


def summary():
    try:
        require_d9_recorded()
    except RuntimeError as e:
        sys.exit(str(e))
    replay = load_replay(OUT)
    seeds = complete_seeds(replay)
    if not seeds:
        sys.exit("no complete seed yet: run `replay_detectors.py run` first")
    print(f"{len(seeds)} paired comparison seeds; frozen trajectories; bootstrap {BOOTSTRAP_REPS} replications\n")
    print("Delay of the first alert after the onset (steps from the start of the onset episode, censored at the end of the run)")
    print("and false alarms per episode, at each nominal false-alarm level. Gap = Agent 2 median minus Agent 1 median.\n")
    print(f"{'regime':6s} {'level':>5s} | {'median A1':>9s} {'cens':>5s} | {'median A2':>9s} {'cens':>5s} | "
          f"{'gap A2 - A1 [95% CI]':>30s} | {'FA A1':>6s} {'FA A2':>6s}")
    at_verdict = []
    for g in REPLAY_REGIMES:
        for j, rate in enumerate(FALSE_ALARM_RATES):
            c = matched_contrast(replay, g, j, seeds, BOOTSTRAP_REPS)
            if j == VERDICT_LEVEL:
                at_verdict.append(c)
            mark = "*" if j == VERDICT_LEVEL else " "
            if g in HARMFUL:
                print(f"{g:6s} {rate:4.0%}{mark} | {c['delay_a']:9.0f} {c['censored_a']:5.2f} | {c['delay_b']:9.0f} "
                      f"{c['censored_b']:5.2f} | {_fmt(c['delay_gap']):>30s} | {c['false_alarm_a']:6.3f} {c['false_alarm_b']:6.3f}")
            else:
                print(f"{g:6s} {rate:4.0%}{mark} | {'':>9s} {'':>5s} | {'':>9s} {'':>5s} | {'':>30s} | "
                      f"{c['false_alarm_a']:6.3f} {c['false_alarm_b']:6.3f}")
        print()
    v = verdict_c(at_verdict)
    print(f"Prediction C at the {VERDICT_RATE:.0%} level (*), Holm over the five lethal regimes:")
    for g in C_REGIMES:
        print(f"  {g}: {'MET' if v['met'][g] else 'not met'}   (Holm p = {v['holm_p'][g]:.3f})")
    print(f"  regimes met: {v['n_met']} of {len(C_REGIMES)}; measured false-alarm rates matched in R0 and R1: "
          f"{'yes' if v['false_alarms_matched'] else 'NO'}")
    print(f"  PREDICTION C: {'SUPPORTED' if v['supported'] else 'NOT SUPPORTED'} on the validation seeds")
    print("\nR3d is reported, not judged. The closed-loop delays of `compare_agents.py summary` (tuned thresholds, unequal "
          "false-alarm rates) are descriptive only.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2, int(sys.argv[4]) if len(sys.argv) > 4 else 50)
    elif cmd == "check":
        check()
    elif cmd == "summary":
        summary()
    else:
        sys.exit(__doc__)
