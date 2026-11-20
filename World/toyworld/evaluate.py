"""Evaluation of an agent over a full run, reading the ground truth."""
import numpy as np

from .config import WorldConfig
from .policies import ScriptedCrosser
from .regimes import make_env, onset_episode


def run_frozen(regime="R0", seeds=range(10), n_episodes=20, route="A", config: WorldConfig | None = None,
               policy_factory=None):
    """Run the frozen agent for `n_episodes` episodes for each seed.
    Returns one record per episode (seed, episode, ccum, outcome, residuals inside zones, ...)."""
    cfg = config or WorldConfig()
    rows = []
    for seed in seeds:
        env = make_env(regime, seed=seed, n_episodes=n_episodes, config=cfg)
        pol = policy_factory() if policy_factory else ScriptedCrosser(route)
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed if ep == 0 else None, options={"new_run": True} if ep == 0 else None)
            pol.reset()
            resid_zone = {0: [], 1: [], 2: []}
            resid_all = []
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(pol.act(obs))
                tr = info["truth"]
                resid_all.append(tr["frozen_resid"])
                if tr["zone"] >= 0:
                    resid_zone[tr["zone"]].append((tr["frozen_resid"], tr["phi"][tr["zone"]]))
                done = term or trunc
            rows.append({
                "seed": seed, "episode": ep, "ccum": tr["ccum"], "outcome": tr["outcome"], "steps": tr["t"],
                "resid_all": np.array(resid_all), "resid_zone": resid_zone,
                "tau_ep": onset_episode(seed, n_episodes) if regime != "R0" else None,
            })
    return rows
