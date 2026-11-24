"""Records step-by-step traces of real closed-loop runs, for the animation (illustration only, not a result).

It replays `toyworld.adaptive.run_agent` with the same agent, trigger, retreat rule and revision recipe, and keeps what
happens at every step: position, nociception observed and predicted, the trigger statistic, damage, the hidden drift value,
and the events (alert, retreat, revision). Nothing in the package is changed.

It only reads VALIDATION seeds. It refuses test seeds (the test seeds were opened once, for the final study).

Usage:
    python scripts/record_traces.py --regime R3c --seed 2008 --out results/traces/R3c_2008.json
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyworld.adaptive import run_agent  # noqa: F401  (kept to show which loop this mirrors)
from toyworld.agent import AgentParams, BaselineAgent
from toyworld.config import WorldConfig
from toyworld.detectors import CusumTrigger, HarmExcessTrigger
from toyworld.model import WorldModel
from toyworld.policies import retreat_action
from toyworld.regimes import make_env, onset_episode
from toyworld.revision import RevisionParams, revise
from toyworld.seeds import TEST_SEEDS, VALIDATION_SEEDS
from toyworld.triggers import StepRecord

# Tuned thresholds of the final study (protocol/trigger_thresholds.json): the 10% grid point of Agent 1 and the 1% grid
# point of Agent 2 in protocol/nominal_reference.json.
THRESHOLD_1 = 0.11420344014886984
THRESHOLD_2 = 15.467351600690028


def make_triggers(ref):
    return {"agent1_alert": HarmExcessTrigger(THRESHOLD_1), "agent2_cusum": CusumTrigger(ref["sigma_v"], ref["sigma_n"], THRESHOLD_2)}


def trace_run(regime, seed, model, trigger, n_episodes=20):
    """Same loop as `adaptive.run_agent` (non-oracle), with a trace of every step."""
    cfg, p, revision = WorldConfig(), AgentParams(), RevisionParams()
    env = make_env(regime, seed=seed, n_episodes=n_episodes, config=cfg)
    world = env.unwrapped
    m = copy.deepcopy(model)
    agent = BaselineAgent(m, cfg, p)
    x0 = np.array(cfg.x0)
    tau = onset_episode(seed, n_episodes) if regime != "R0" else None
    trigger.reset_run()

    # record the statistic the trigger computes at each step
    last = {}
    orig = trigger.statistic

    def spy(rec):
        s = orig(rec)
        last["s"] = float(s)
        return s
    trigger.statistic = spy

    eps = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed if ep == 0 else None, options={"new_run": True} if ep == 0 else None)
        agent.reset()
        trigger.reset_episode()
        O, U = [obs], []
        hist = torch.zeros(p.K, 2)
        n_sum = float(obs[4])
        alert_step, retreating, aborted, done, retreat_from = None, False, False, False, None
        tr = {k: [] for k in ("x", "y", "n", "npred", "stat", "ccum", "theta", "d", "psi", "mode")}

        def log(o, truth, npred, stat, mode):
            tr["x"].append(float(truth["x"][0])); tr["y"].append(float(truth["x"][1]))          # true position
            tr["n"].append(float(o[4])); tr["npred"].append(npred); tr["stat"].append(stat)
            tr["ccum"].append(float(truth["ccum"])); tr["theta"].append(float(truth["theta"][0]))
            tr["d"].append(int(truth["d"])); tr["psi"].append(float(np.max(np.abs(truth["psi"])))); tr["mode"].append(mode)

        with torch.no_grad():
            np0 = float(m.predict_n(torch.tensor(obs[None, 0:2], dtype=torch.float32))[0])
        log(obs, info["truth"], np0, None, 0)
        while not done:
            u = retreat_action(obs[0:2], obs[2:4], x0) if retreating else agent.act(obs)
            hist = torch.cat([torch.as_tensor(u, dtype=torch.float32)[None], hist[:-1]])
            with torch.no_grad():
                dv_pred = m.predict_dv(torch.tensor(obs[None, 0:2], dtype=torch.float32),
                                       torch.tensor(obs[None, 2:4], dtype=torch.float32), hist[None])[0].numpy()
            new, _, term, trunc, info = env.step(u)
            n_sum += float(new[4])
            with torch.no_grad():
                n_pred = float(m.predict_n(torch.tensor(new[None, 0:2], dtype=torch.float32))[0])
            stat = None
            if alert_step is None:
                rec = StepRecord(t=info["truth"]["t"], obs=new.copy(), n_pred=n_pred,
                                 dv_resid=new[2:4] - obs[2:4] - dv_pred, b_hat=cfg.D - cfg.dt * n_sum)
                fired = trigger.step(rec)
                stat = last["s"]
                if fired:
                    alert_step = rec.t
                    retreating = bool(new[0] < cfg.wall_x)
                    retreat_from = rec.t if retreating else None
            log(new, info["truth"], n_pred, stat, 2 if retreating else 1)
            O.append(new)
            U.append(np.asarray(u, float))
            obs = new
            done = bool(term or trunc)
            if retreating and np.linalg.norm(obs[0:2] - x0) <= cfg.r0:
                done = aborted = True
        trigger.end_episode()
        truth = info["truth"]
        rev = None
        if trigger.revise_at_end(alert_step is not None):
            rev = revise(m, np.array(O), np.array(U), revision, seed=1000 * seed + ep)
        outcome = "retreat" if (aborted and truth["outcome"] is None) else truth["outcome"]
        eps.append({"ep": ep, "outcome": outcome, "route": agent.route, "steps": int(truth["t"]), "ccum": float(truth["ccum"]),
                    "alert_step": alert_step, "retreat_from": retreat_from, "revised": rev is not None,
                    "loss_before": None if rev is None else float(rev["before"]), "loss_after": None if rev is None else float(rev["after"]),
                    "tr": {k: ([round(v, 3) if isinstance(v, float) else v for v in vals]) for k, vals in tr.items()}})
    return {"tau": tau, "episodes": eps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent / "results" / "baseline.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    if a.seed in TEST_SEEDS or a.seed not in VALIDATION_SEEDS:
        sys.exit("only validation seeds are allowed here (the test seeds were opened once, for the final study)")
    ref = json.loads((Path(__file__).resolve().parent.parent / "protocol" / "nominal_reference.json").read_text())
    model = WorldModel.load(a.model)
    out = {"regime": a.regime, "seed": a.seed, "n_episodes": a.episodes, "agents": {}}
    for name, trig in make_triggers(ref).items():
        out["agents"][name] = trace_run(a.regime, a.seed, model, trig, a.episodes)
        print(name, [e["outcome"] for e in out["agents"][name]["episodes"]], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
