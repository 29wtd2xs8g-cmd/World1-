"""Pretraining data for the world model (docs/agents.md, section 4).

Collected in the NOMINAL world, with a very large D and a goal outside the arena so that no episode
ends early. Each segment starts from a random state (privileged reset of the data-collection stage;
evaluation always starts at x0) and applies piecewise-constant random thrust. Everything recorded is
observed (noisy): the model learns from what an agent can see.
"""
from dataclasses import replace

import numpy as np

from .config import WorldConfig
from .model import K_DEFAULT
from .world import ToyWorld


def collect(seeds, segments_per_seed=20, seg_len=100, K=K_DEFAULT, config: WorldConfig | None = None):
    cfg = replace(config or WorldConfig(), D=1e9, goal=(100.0, 100.0))
    X_dyn, Y_dv, X_noc, Y_n = [], [], [], []
    for seed in seeds:
        env = ToyWorld(cfg, n_episodes=segments_per_seed + 1, run_seed=seed)
        rng = np.random.default_rng([seed, 4242])
        for s in range(segments_per_seed):
            env.reset(seed=seed if s == 0 else None, options={"new_run": True} if s == 0 else None)
            obs = env.teleport(rng.uniform(0.0, cfg.arena, 2), rng.normal(0.0, 0.7, 2))
            u = np.zeros((seg_len, 2))
            t = 0
            while t < seg_len:
                hold = int(rng.integers(5, 21))
                u[t:t + hold] = rng.uniform(-1.0, 1.0, 2)
                t += hold
            observed = [obs]
            for t in range(seg_len):
                obs, _, _, _, _ = env.step(u[t])
                observed.append(obs)
            O = np.array(observed)
            padded = np.vstack([np.zeros((K - 1, 2)), u])          # zeros before the segment start
            for t in range(seg_len):
                hist = padded[t:t + K][::-1]                        # newest first: u[t], u[t-1], ...
                X_dyn.append(np.concatenate([O[t, 0:2], O[t, 2:4], hist.ravel()]))
                Y_dv.append(O[t + 1, 2:4] - O[t, 2:4])
                X_noc.append(O[t, 0:2])
                Y_n.append(O[t, 4])
    return {"X_dyn": np.array(X_dyn), "Y_dv": np.array(Y_dv),
            "X_noc": np.array(X_noc), "Y_n": np.array(Y_n)}
