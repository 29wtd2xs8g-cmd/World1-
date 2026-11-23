"""Checklist 2.3: headroom per regime = what the best possible reference gains over the frozen agent.

For each harmful regime and each validation seed it runs the ORACLE (true post-drift parameters, never revised) and
the ALWAYS-ADAPTING agent (fine-tuned at the end of every episode); the FROZEN agent is recomputed by
the same code (an earlier version read results/regimes_baseline.json, which was stale for R3d, see DEV-1). Results are appended to results/headroom.jsonl as they come, so a
partial run is usable. Criterion D1 (docs/research.md, section 6): headroom >= 15% in at least 3 harmful regimes,
at least one of them in {R3d, R3c}; headroom = (Ccum_frozen - Ccum_oracle) / Ccum_frozen, post-tau episodes.

Usage:
  PYTHONPATH=. python scripts/headroom.py run model.pt [n_seeds] [workers] [margin_frac] [regimes, comma separated | all]
  PYTHONPATH=. python scripts/headroom.py invariance margin_frac
  PYTHONPATH=. python scripts/headroom.py summary [margin_frac]
`margin_frac` (default 1.0 = the first protocol; the agents' default is 0.7 since DEV-1, docs/protocol_deviations.md) is
the safety buffer of the route rule (AgentParams.margin_frac); results with another value go to
results/headroom_rho<value>.jsonl. `summary` lists outcomes first (failure, success, timeout) and reports Ccum over the
successful episodes next to the all-episode mean (the all-episode mean is the pre-registered D1 definition).
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from toyworld import metrics
from toyworld.seeds import BOOTSTRAP_REPS, VALIDATION_SEEDS

HARMFUL = ["R2d", "R2c", "R2l", "R3d", "R3c", "R3l"]
OUT = Path("results/headroom.jsonl")


def out_path(rho):
    return OUT if rho == 1.0 else Path(f"results/headroom_rho{rho}.jsonl")



FROZEN = Path("results/regimes_baseline.json")
MODEL = None


def _init(path):
    global MODEL
    from toyworld.model import WorldModel
    MODEL = WorldModel.load(path)


def _job(args):
    kind, regime, seed, rho = args
    from toyworld.adaptive import run_agent
    from toyworld.agent import AgentParams
    from toyworld.triggers import AlwaysTrigger, NeverTrigger
    p = AgentParams(margin_frac=rho)
    if kind == "oracle":
        rows = run_agent(regime, seed, MODEL, NeverTrigger(), oracle=True, params=p)
    elif kind == "always":
        rows = run_agent(regime, seed, MODEL, AlwaysTrigger(), params=p)
    else:
        rows = run_agent(regime, seed, MODEL, NeverTrigger(), params=p)
    return {"kind": kind, "regime": regime, "seed": seed, "rows": rows}


def run(model_path, n_seeds=6, workers=2, rho=1.0, regimes=None):
    seeds = list(VALIDATION_SEEDS[:n_seeds])
    regimes = regimes or HARMFUL
    OUT = out_path(rho)
    done = set()
    if OUT.exists():
        done = {(r["kind"], r["regime"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}
    jobs = [(k, g, s, rho) for s in seeds for k in ("oracle", "always") for g in regimes if (k, g, s) not in done]
    jobs += [("frozen", g, s, rho) for s in seeds for g in regimes if ("frozen", g, s) not in done]
    OUT.parent.mkdir(exist_ok=True)
    with mp.Pool(workers, initializer=_init, initargs=(model_path,)) as pool, open(OUT, "a") as f:
        for res in pool.imap_unordered(_job, jobs, chunksize=1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print("done", res["kind"], res["regime"], res["seed"], flush=True)


def _load(OUT=OUT, legacy_frozen=False):
    """Runs of one output file. `legacy_frozen` (run 1 only) also reads the frozen runs of the old verification file,
    which is stale for R3d (written before its amplitude went from 0.25 to 0.35); never use it for new runs."""
    runs = {}
    if OUT.exists():
        for r in map(json.loads, OUT.read_text().splitlines()):
            runs[(r["kind"], r["regime"], r["seed"])] = r["rows"]
    if legacy_frozen and FROZEN.exists():
        by = {}
        for r in json.load(open(FROZEN)):
            by.setdefault((r["regime"], r["seed"]), []).append(r)
        for (g, s), rows in by.items():
            runs.setdefault(("frozen", g, s), sorted(rows, key=lambda r: r["episode"]))
    return runs


def frozen_invariance(rho):
    """The frozen agent must not depend on margin_frac (its predictions stay nominal, so its route stays A): compare the
    frozen runs computed under `rho` with those computed under 1.0, episode by episode."""
    def frozen(path):
        out = {}
        if path.exists():
            for r in map(json.loads, path.read_text().splitlines()):
                if r["kind"] == "frozen":
                    out[(r["regime"], r["seed"])] = [(e["ccum"], e["steps"], e["route"], e["outcome"]) for e in r["rows"]]
        return out
    a, b = frozen(out_path(rho)), frozen(out_path(1.0))
    common = sorted(set(a) & set(b))
    same = [k for k in common if a[k] == b[k]]
    print(f"frozen runs computed under both margin_frac {rho} and 1.0: {len(common)}; identical episode by episode: {len(same)}")
    for k in common:
        if k not in same:
            print("  differs:", k)


KINDS = ("frozen", "oracle", "always")


def _seeds(runs, g):
    return sorted({s for (k, gg, s) in runs if gg == g and k == "oracle"
                   and ("frozen", g, s) in runs and ("always", g, s) in runs})


def _boot_ratio(O, F, rng, reps):
    """Bootstrap interval of 1 - mean(O) / mean(F) over seeds (the headroom of criterion D1)."""
    i = rng.integers(0, len(F), (reps, len(F)))
    h = 1 - O[i].mean(axis=1) / F[i].mean(axis=1)
    return np.percentile(h, [2.5, 97.5])


def summary(rho=1.0, seed=0):
    """Outcomes first (rule 2 of toyworld/metrics.py), then the headroom. Post-tau episodes, paired seeds."""
    runs = _load(out_path(rho), legacy_frozen=(rho == 1.0))
    rng = np.random.default_rng(seed)
    print(f"margin_frac {rho}; post-tau episodes; bootstrap {BOOTSTRAP_REPS} replications over seeds\n")
    print(f"{'regime':6s} {'family':11s} {'kind':7s} {'seeds':>5s} | {'failure':>7s} {'success':>7s} {'timeout':>7s} {'route A':>7s} | "
          f"{'Ccum|success':>12s} {'Ccum all':>8s}")
    rows_h, ok = [], []
    for g in HARMFUL:
        seeds = _seeds(runs, g)
        if not seeds:
            continue
        per = {k: [metrics.outcome_table(metrics.post_tau(runs[(k, g, s)])) for s in seeds] for k in KINDS}
        for k in KINDS:                                    # means over seeds, so that every figure is paired and comparable
            m = {key: np.mean([t[key] for t in per[k]]) for key in ("failure", "success", "timeout", "ccum_all")}
            ok_seeds = [t["ccum_success"] for t in per[k] if not np.isnan(t["ccum_success"])]
            route_a = np.mean([np.mean([r["route"] == "A" for r in metrics.post_tau(runs[(k, g, s)])]) for s in seeds])
            print(f"{g:6s} {metrics.family(g):11s} {k:7s} {len(seeds):5d} | {m['failure']:7.2f} {m['success']:7.2f} {m['timeout']:7.2f} "
                  f"{route_a:7.2f} | {np.mean(ok_seeds) if ok_seeds else float('nan'):12.2f} {m['ccum_all']:8.2f}")
        F, O, A = (np.array([t["ccum_all"] for t in per[k]]) for k in KINDS)
        fF, fO, fA = (np.array([t["failure"] for t in per[k]]) for k in KINDS)
        lo, hi = _boot_ratio(O, F, rng, BOOTSTRAP_REPS)
        head = 1 - O.mean() / F.mean()
        fb = metrics.paired_bootstrap(fF, fO, BOOTSTRAP_REPS)
        rows_h.append((g, head, lo, hi, fb, 1 - A.mean() / F.mean()))
        ok.append((g, head))
    print(f"\n{'regime':6s} | {'failure frozen - oracle (points)':>34s} | {'D1 headroom, Ccum all episodes':>32s} | {'always-adapting gain':>20s}")
    for g, head, lo, hi, fb, gain in rows_h:
        print(f"{g:6s} | {fb['mean'] * 100:+9.1f}  [{fb['lo'] * 100:+6.1f}, {fb['hi'] * 100:+6.1f}]        | "
              f"{head * 100:+9.1f}%  [{lo * 100:+6.1f}, {hi * 100:+6.1f}] | {gain * 100:+19.1f}%")
    n_ok = sum(h >= 0.15 for _, h in ok)
    has_r3 = any(h >= 0.15 for g, h in ok if g in ("R3d", "R3c"))
    print(f"\ncriterion D1 as pre-registered (Ccum, all post-tau episodes): {n_ok} of {len(ok)} regimes at >= 15%; "
          f"one of R3d/R3c among them: {has_r3} -> {'MET' if (n_ok >= 3 and has_r3) else 'NOT MET'}")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "run":
        run(a[2], int(a[3]) if len(a) > 3 else 6, int(a[4]) if len(a) > 4 else 2,
            float(a[5]) if len(a) > 5 else 1.0, a[6].split(",") if len(a) > 6 and a[6] != "all" else None)
    elif a[1] == "invariance":
        frozen_invariance(float(a[2]))
    else:
        summary(float(a[2]) if len(a) > 2 else 1.0)
