"""Run of one agent, with one trigger, over a full drift run (checklist 2.2 and 2.4).

One run = `n_episodes` chained episodes of one regime and one seed. The agent is `BaselineAgent`; what varies is

  * the trigger (`triggers.py`): when to alert and when to revise;
  * `oracle=True`: the planner uses `AnalyticModel` with the TRUE current hidden parameters (privileged read of the
    world, evaluation reference only) and is never revised.

Protocol (decisions D2 and D3, docs/research.md section 6):
  * the first alert of an episode makes the agent retreat towards the start `x0`, with the damage-free
    `retreat_action`, and the episode is closed ("retreat") once the agent is back inside the clear disc of radius
    `r0` around `x0`. The next episode is the retry, from `x0`, with a fresh margin estimate;
  * the retreat exists on the start side of the wall only: past the wall the agent keeps going to the goal (the
    nearest exit of the zone), the alert is only recorded;
  * at the end of an episode, if `trigger.revise_at_end(alerted)`, both networks are fine-tuned on the window of that
    episode (`revision.py`), with the same budget for every agent. Each run starts from its own copy of the
    pretrained model.
"""
import copy

import numpy as np
import torch

from .agent import AgentParams, BaselineAgent
from .config import WorldConfig
from .model import AnalyticModel
from .policies import retreat_action
from .regimes import make_env, onset_episode
from .revision import RevisionParams, revise
from .triggers import StepRecord


def _sync_oracle(model: AnalyticModel, world):
    """Privileged: the oracle reads the true hidden parameters of the world."""
    model.theta = torch.tensor(world.theta, dtype=torch.float32)
    model.psi = torch.tensor(world.psi, dtype=torch.float32)
    model.d = int(world.d)


def run_agent(regime, seed, model, trigger, oracle=False, n_episodes=20, revision: RevisionParams | None = None,
              params: AgentParams | None = None, config: WorldConfig | None = None):
    """Returns one record per episode."""
    cfg = config or WorldConfig()
    p = params or AgentParams()
    revision = revision or RevisionParams()
    env = make_env(regime, seed=seed, n_episodes=n_episodes, config=cfg)
    world = env.unwrapped
    m = AnalyticModel(cfg) if oracle else copy.deepcopy(model)
    agent = BaselineAgent(m, cfg, p)
    x0 = np.array(cfg.x0)
    tau = onset_episode(seed, n_episodes) if regime != "R0" else None
    trigger.reset_run()
    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed if ep == 0 else None, options={"new_run": True} if ep == 0 else None)
        if oracle:
            _sync_oracle(m, world)
        agent.reset()
        trigger.reset_episode()
        O, U = [obs], []
        hist = torch.zeros(p.K, 2)
        n_sum = float(obs[4])
        alert_step, alert_active, retreating, aborted, done = None, None, False, False, False
        while not done:
            if oracle:
                _sync_oracle(m, world)
            u = retreat_action(obs[0:2], obs[2:4], x0) if retreating else agent.act(obs)
            hist = torch.cat([torch.as_tensor(u, dtype=torch.float32)[None], hist[:-1]])
            with torch.no_grad():
                dv_pred = m.predict_dv(torch.tensor(obs[None, 0:2], dtype=torch.float32),
                                       torch.tensor(obs[None, 2:4], dtype=torch.float32), hist[None])[0].numpy()
            new, _, term, trunc, info = env.step(u)
            n_sum += float(new[4])
            if alert_step is None:
                with torch.no_grad():
                    n_pred = float(m.predict_n(torch.tensor(new[None, 0:2], dtype=torch.float32))[0])
                rec = StepRecord(t=info["truth"]["t"], obs=new.copy(), n_pred=n_pred,
                                 dv_resid=new[2:4] - obs[2:4] - dv_pred, b_hat=cfg.D - cfg.dt * n_sum)
                if trigger.step(rec):
                    alert_step = rec.t
                    alert_active = bool(info.get("drift", {}).get("active", False))      # ground truth, logging only
                    retreating = bool(new[0] < cfg.wall_x)
            O.append(new)
            U.append(np.asarray(u, float))
            obs = new
            done = bool(term or trunc)
            if retreating and np.linalg.norm(obs[0:2] - x0) <= cfg.r0:
                done = aborted = True
        trigger.end_episode()
        tr = info["truth"]
        rev = None
        if not oracle and trigger.revise_at_end(alert_step is not None):
            rev = revise(m, np.array(O), np.array(U), revision, seed=1000 * seed + ep)
        rows.append({
            "regime": regime, "seed": seed, "episode": ep, "tau_ep": tau, "ccum": float(tr["ccum"]),
            "outcome": "retreat" if (aborted and tr["outcome"] is None) else tr["outcome"],
            "steps": int(tr["t"]), "route": agent.route,
            "alert_step": alert_step, "alert_during_drift": alert_active,
            "revised": rev is not None,
            "loss_before": None if rev is None else rev["before"], "loss_after": None if rev is None else rev["after"],
        })
    return rows
