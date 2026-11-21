"""Ground-truth journal (checklist 1.6).

For every episode: onset and nature of the drift, cause k*, seed, costs, outcome.
Step-by-step trajectories go to an .npz file; per-episode summaries go to a .jsonl file.
This journal must never enter an agent's input.
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


class TruthLogger(gym.Wrapper):
    def __init__(self, env, regime="R0"):
        super().__init__(env)
        self.regime = regime
        self.records = []
        self._steps = None

    def _start(self, info):
        self._steps = {k: [] for k in ("x", "v", "c", "ccum", "theta", "psi", "d", "zone", "resid")}
        self._push(info)

    def _push(self, info):
        tr = info["truth"]
        self._steps["x"].append(tr["x"]); self._steps["v"].append(tr["v"])
        self._steps["c"].append(tr["c"]); self._steps["ccum"].append(tr["ccum"])
        self._steps["theta"].append(tr["theta"]); self._steps["psi"].append(tr["psi"])
        self._steps["d"].append(tr["d"]); self._steps["zone"].append(tr["zone"])
        self._steps["resid"].append(tr["frozen_resid"])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._start(info)
        self._last = info
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._push(info)
        self._last = info
        if term or trunc:
            tr, dr = info["truth"], info.get("drift", {})
            w = self.env.unwrapped
            self.records.append({
                "regime": self.regime, "seed": w.run_seed, "episode": tr["episode"],
                "nature": dr.get("nature"), "k_star": dr.get("k_star"),
                "tau_global": dr.get("tau_global"), "T": w.cfg.T,
                "outcome": tr["outcome"], "ccum": tr["ccum"], "steps": tr["t"],
                "_traj": {k: np.array(v) for k, v in self._steps.items()},
            })
        return obs, r, term, trunc, info

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "episodes.jsonl", "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps({k: v for k, v in r.items() if k != "_traj"}, ensure_ascii=False) + "\n")
        np.savez_compressed(
            directory / "trajectories.npz",
            **{f"ep{r['episode']}_{k}": v for r in self.records for k, v in r["_traj"].items()},
        )
