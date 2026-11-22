"""Checklist 2.3: headroom per regime = what the best possible reference gains over the frozen agent.

For each harmful regime and each validation seed it runs the ORACLE (true post-drift parameters, never revised) and
the ALWAYS-ADAPTING agent (fine-tuned at the end of every episode); the FROZEN agent is read from
results/regimes_baseline.json (same code path: the Never-trigger runner reproduces it exactly, see tests) or
recomputed if that file does not cover the seeds. Results are appended to results/headroom.jsonl as they come, so a
partial run is usable. Criterion D1 (docs/research.md, section 6): headroom >= 15% in at least 3 harmful regimes,
at least one of them in {R3d, R3c}; headroom = (Ccum_frozen - Ccum_oracle) / Ccum_frozen, post-tau episodes.

Usage:
  PYTHONPATH=. python scripts/headroom.py run model.pt [n_seeds] [workers] [margin_frac] [regimes, comma separated]
  PYTHONPATH=. python scripts/headroom.py summary [margin_frac]
  PYTHONPATH=. python scripts/headroom.py detail [margin_frac]
`margin_frac` (default 1.0 = the pre-registered baseline) is the exploratory safety buffer of the route rule
(AgentParams.margin_frac); results with another value go to results/headroom_rho<value>.jsonl.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from toyworld.seeds import VALIDATION_SEEDS

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
    have_frozen = set()
    if FROZEN.exists():
        have_frozen = {(r["regime"], r["seed"]) for r in json.load(open(FROZEN))}
    jobs = [(k, g, s, rho) for s in seeds for k in ("oracle", "always") for g in regimes if (k, g, s) not in done]
    jobs += [("frozen", g, s, rho) for s in seeds for g in regimes if (g, s) not in have_frozen and ("frozen", g, s) not in done]
    OUT.parent.mkdir(exist_ok=True)
    with mp.Pool(workers, initializer=_init, initargs=(model_path,)) as pool, open(OUT, "a") as f:
        for res in pool.imap_unordered(_job, jobs, chunksize=1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print("done", res["kind"], res["regime"], res["seed"], flush=True)


def _load(OUT=OUT):
    runs = {}
    if OUT.exists():
        for r in map(json.loads, OUT.read_text().splitlines()):
            runs[(r["kind"], r["regime"], r["seed"])] = r["rows"]
    if FROZEN.exists():
        by = {}
        for r in json.load(open(FROZEN)):
            by.setdefault((r["regime"], r["seed"]), []).append(r)
        for (g, s), rows in by.items():
            runs.setdefault(("frozen", g, s), sorted(rows, key=lambda r: r["episode"]))
    return runs


def _post(rows):
    return [r for r in rows if r["episode"] >= r["tau_ep"]]


def summary(rho=1.0, boot=2000, seed=0):
    runs = _load(out_path(rho))
    rng = np.random.default_rng(seed)
    print(f"{'regime':6s} {'seeds':>5s} | {'Ccum frozen':>11s} {'oracle':>7s} {'always':>7s} | {'fail frozen':>11s} {'oracle':>7s} "
          f"{'always':>7s} | {'headroom oracle':>26s} | {'always-adapt gain':>17s}")
    ok = []
    for g in HARMFUL:
        seeds = sorted({s for (k, gg, s) in runs if gg == g and k == "oracle"
                        and ("frozen", g, s) in runs and ("always", g, s) in runs})
        if not seeds:
            continue
        def stats(k, s):
            post = _post(runs[(k, g, s)])
            return [np.mean([r["ccum"] for r in post]), np.mean([r["outcome"] == "failure" for r in post])]

        per = {k: np.array([stats(k, s) for s in seeds]) for k in ("frozen", "oracle", "always")}
        F, O, A = (per[k][:, 0] for k in ("frozen", "oracle", "always"))
        head = 1 - O.mean() / F.mean()
        bs = []
        for _ in range(boot):
            i = rng.integers(0, len(seeds), len(seeds))
            bs.append(1 - O[i].mean() / F[i].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{g:6s} {len(seeds):5d} | {F.mean():11.2f} {O.mean():7.2f} {A.mean():7.2f} | "
              f"{per['frozen'][:, 1].mean():11.2f} {per['oracle'][:, 1].mean():7.2f} {per['always'][:, 1].mean():7.2f} | "
              f"{head * 100:+9.1f}%  [{lo * 100:+6.1f}, {hi * 100:+6.1f}] | {(1 - A.mean() / F.mean()) * 100:+16.1f}%")
        ok.append((g, head))
    n_ok = sum(h >= 0.15 for _, h in ok)
    has_r3 = any(h >= 0.15 for g, h in ok if g in ("R3d", "R3c"))
    print(f"\nregimes with headroom >= 15%: {n_ok} of {len(ok)} measured; one of R3d/R3c among them: {has_r3}")
    print("criterion D1:", "MET" if (n_ok >= 3 and has_r3) else "NOT MET (on the seeds measured so far)")


def detail(rho=1.0):
    """Why the headroom is what it is: route chosen, outcomes and episode length, post-tau episodes."""
    runs = _load(out_path(rho))
    print(f"{'regime':6s} {'kind':7s} {'route A':>8s} {'success':>8s} {'failure':>8s} {'steps (success)':>16s} {'Ccum (success)':>15s}")
    for g in HARMFUL:
        for k in ("frozen", "oracle", "always"):
            rows = [r for (kk, gg, s), rs in runs.items() if kk == k and gg == g for r in _post(rs)]
            if not rows:
                continue
            ok = [r for r in rows if r["outcome"] == "success"]
            print(f"{g:6s} {k:7s} {np.mean([r['route'] == 'A' for r in rows]):8.2f} {np.mean([r['outcome'] == 'success' for r in rows]):8.2f} "
                  f"{np.mean([r['outcome'] == 'failure' for r in rows]):8.2f} {np.mean([r['steps'] for r in ok]) if ok else float('nan'):16.0f} "
                  f"{np.mean([r['ccum'] for r in ok]) if ok else float('nan'):15.2f}")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "detail":
        detail(float(a[2]) if len(a) > 2 else 1.0)
    elif a[1] == "run":
        run(a[2], int(a[3]) if len(a) > 3 else 6, int(a[4]) if len(a) > 4 else 2,
            float(a[5]) if len(a) > 5 else 1.0, a[6].split(",") if len(a) > 6 else None)
    else:
        summary(float(a[2]) if len(a) > 2 else 1.0)
