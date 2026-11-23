"""Agent 1 against Agent 2 on the 50 comparison seeds (validation seeds minus the ten tuning seeds).

  PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/compare_agents.py run results/baseline.pt [workers] [n_seeds]
  PYTHONPATH=. python scripts/compare_agents.py summary [references.jsonl]

`run` needs the tuned thresholds (`protocol/trigger_thresholds.json`, written by `scripts/tune_triggers.py`) and refuses
to start without them. It plays Agent 1 and Agent 2 in the eight regimes, and the frozen and always-adapting references
in R0 and R1 (the harmful regimes take their references from the D1 file `results/headroom_rho0.7.jsonl`). Results are
appended to `results/compare.jsonl`, and finished jobs are skipped on a relaunch. Never touches the test seeds.
`summary` prints outcomes first (docs/protocol.md, section 3.1), then the paired contrasts and the verdict of
prediction B (decision D9). The detection verdict (prediction C) is read at matched false-alarm rates by
`scripts/replay_detectors.py`; the closed-loop delays printed here are descriptive.
`summary` refuses to run until decision D9 is recorded in docs/protocol.md (section 3.6).
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from toyworld.adaptive import run_agent
from toyworld.compare import (AGENTS, B_REGIMES, COMPARISON_SEEDS, HARMFUL, REGIMES, calm_false_alarm_gap, contrast,
                              holm_adjust, load_runs, load_thresholds, make_trigger, require_d9_recorded, seed_vectors,
                              verdict_b)
from toyworld.metrics import LETHAL, family
from toyworld.model import WorldModel
from toyworld.nominal import NominalReference, model_hash
from toyworld.seeds import BOOTSTRAP_REPS
from toyworld.triggers import AlwaysTrigger, NeverTrigger

OUT = Path("results/compare.jsonl")
REFS = Path("results/headroom_rho0.7.jsonl")
MODEL = REF = TH = None


def _init(path):
    global MODEL, REF, TH
    MODEL, REF, TH = WorldModel.load(path), NominalReference.load(), load_thresholds()


def _job(args):
    kind, regime, seed = args
    trig = {"frozen": NeverTrigger, "always": AlwaysTrigger}[kind]() if kind in ("frozen", "always") \
        else make_trigger(kind, TH, REF)
    return {"kind": kind, "regime": regime, "seed": seed, "rows": run_agent(regime, seed, MODEL, trig)}


def run(path, workers=2, n_seeds=50):
    ref = NominalReference.load()
    if ref.model_sha256 != model_hash(path):
        sys.exit("the nominal reference was computed on another model")
    load_thresholds()                                           # fails here, before any compute, if untuned
    seeds = COMPARISON_SEEDS[:n_seeds]
    done = {(r["kind"], r["regime"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    jobs = [(k, g, s) for s in seeds for g in REGIMES for k in AGENTS if (k, g, s) not in done]
    jobs += [(k, g, s) for s in seeds for g in ("R0", "R1") for k in ("frozen", "always") if (k, g, s) not in done]
    OUT.parent.mkdir(exist_ok=True)
    with mp.Pool(workers, initializer=_init, initargs=(path,)) as pool, open(OUT, "a") as f:
        for res in pool.imap_unordered(_job, jobs, chunksize=1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print("done", res["kind"], res["regime"], res["seed"], flush=True)


def _fmt(c, key):
    g = c[key]
    return f"{g['value']:+7.3f} [{g['lo']:+7.3f}, {g['hi']:+7.3f}]"


def summary(refs=REFS):
    try:
        require_d9_recorded()
    except RuntimeError as e:
        sys.exit(str(e))
    runs = load_runs(OUT)
    for k, v in load_runs(refs).items():
        runs.setdefault(k, v)
    seeds = [s for s in COMPARISON_SEEDS if all((a, g, s) in runs for a in AGENTS for g in REGIMES)]
    if not seeds:
        sys.exit("no complete seed yet: run `compare_agents.py run` first")
    print(f"{len(seeds)} paired comparison seeds; bootstrap {BOOTSTRAP_REPS} replications; post-onset episodes\n")
    print(f"{'regime':6s} {'family':11s} {'kind':13s} | {'failure':>7s} {'success':>7s} {'timeout':>7s} {'retreat':>7s} | "
          f"{'Ccum|succ':>9s} {'Ccum all':>8s} | {'revis/run':>9s} {'false al/ep':>11s}")
    for g in REGIMES:
        for k in ("frozen", "agent1_alert", "agent2_cusum", "always", "oracle"):
            if not all((k, g, s) in runs for s in seeds):
                continue
            v = seed_vectors(runs, k, g, seeds)
            fa = v["fa"].sum() / v["fa_n"].sum() if v["fa_n"].sum() else float("nan")
            print(f"{g:6s} {family(g):11s} {k:13s} | {v['failure'].mean():7.2f} {v['success'].mean():7.2f} "
                  f"{v['timeout'].mean():7.2f} {v['retreat'].mean():7.2f} | {np.nanmean(v['ccum_success']) if np.isfinite(v['ccum_success']).any() else float('nan'):9.2f} "
                  f"{v['ccum'].mean():8.2f} | {v['revisions'].mean():9.1f} {fa:11.3f}")
    a, b = AGENTS
    cs = [contrast(runs, a, b, g, seeds, BOOTSTRAP_REPS) for g in REGIMES]
    hf = holm_adjust(cs, "failure_gap", LETHAL)
    hc = holm_adjust(cs, "ccum_gap_rel", HARMFUL)
    cgap = calm_false_alarm_gap(runs, a, b, seeds)
    vb = verdict_b(cs)
    print(f"\nAgent 1 against Agent 2 (positive = Agent 1 better); Holm over the lethal regimes (failure) and the harmful ones (Ccum)")
    print(f"{'regime':6s} | {'failure gap (points)':>29s} {'Holm p':>7s} | {'Ccum gap (relative)':>29s} {'Holm p':>7s} | "
          f"{'success A1 - A2':>15s} | B")
    for c in cs:
        if c["regime"] in ("R0", "R1"):
            continue
        tag = ("MET" if vb["met"][c["regime"]] else "not met") if c["regime"] in B_REGIMES else "no verdict"
        print(f"{c['regime']:6s} | {_fmt(c, 'failure_gap'):>29s} {hf.get(c['regime'], float('nan')):7.3f} | "
              f"{_fmt(c, 'ccum_gap_rel'):>29s} {hc.get(c['regime'], float('nan')):7.3f} | "
              f"{c['success_a'] - c['success_b']:+15.3f} | {tag}")
    print(f"\nPREDICTION B (R2c and R3c, Holm over the two; Ccum on all post-onset episodes): "
          f"{'SUPPORTED' if vb['supported'] else 'NOT SUPPORTED'} on the validation seeds")
    print(f"  Holm p, failure gap: " + "  ".join(f"{g} {vb['holm_failure'][g]:.3f}" for g in B_REGIMES)
          + "   Ccum gap: " + "  ".join(f"{g} {vb['holm_ccum'][g]:.3f}" for g in B_REGIMES))
    print(f"\nClosed-loop detection, DESCRIPTIVE ONLY (tuned thresholds, unequal false-alarm rates; the verdict of C is in "
          f"replay_detectors.py): delay in steps from the onset episode, censored at the end of the run")
    print(f"{'regime':6s} | {'median A1':>9s} {'cens':>5s} | {'median A2':>9s} {'cens':>5s} | {'gap A2 - A1 [95% CI]':>30s}")
    for c in cs:
        if c["regime"] in HARMFUL:
            print(f"{c['regime']:6s} | {c['delay_a']:9.0f} {c['censored_a']:5.2f} | {c['delay_b']:9.0f} {c['censored_b']:5.2f} | "
                  f"{_fmt(c, 'delay_gap'):>30s}")
    fa = {c["regime"]: (c["false_alarm_a"], c["false_alarm_b"]) for c in cs}
    print(f"\nFalse alarms per episode (Agent 1 / Agent 2): " + "  ".join(f"{g} {fa[g][0]:.3f}/{fa[g][1]:.3f}" for g in REGIMES))
    print(f"Calm-regime gap (R0 and R1, Agent 2 minus Agent 1): {cgap:+.3f}")
    print("\nRegimes outside R2c and R3c are reported without a verdict. Predictions A, D and E are not computed here.")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2, int(sys.argv[4]) if len(sys.argv) > 4 else 50)
    elif cmd == "summary":
        summary(Path(sys.argv[2]) if len(sys.argv) > 2 else REFS)
    else:
        sys.exit(__doc__)
