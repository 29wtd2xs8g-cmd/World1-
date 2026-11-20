"""The three drift wrappers (checklist 1.2): dynamics, cost, latency.

Each wrapper receives a `DriftSchedule` and a cause zone `k_star` (meaningless for latency).
It modifies a single hidden parameter of the world before every step.
Drift time is global to the run: t_global = episode * T + t.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class DriftSchedule:
    onset_episode: int = 5          # tau: episode in which the drift starts
    onset_step: int = 0             # tau: step inside that episode
    form: str = "jump"              # "jump" | "ramp" | "steps"
    target: float = 1.0             # final value of the parameter
    ramp_steps: int = 2500          # duration of a ramp, in global steps
    step_every: int = 1000          # "steps": +1 level every `step_every` global steps

    def onset(self, T: int) -> int:
        return self.onset_episode * T + self.onset_step

    def value(self, nominal: float, episode: int, t: int, T: int) -> float:
        s = episode * T + t - self.onset(T)
        if s < 0:
            return nominal
        if self.form == "jump":
            return self.target
        if self.form == "ramp":
            return nominal + (self.target - nominal) * min(1.0, s / self.ramp_steps)
        if self.form == "steps":
            return min(self.target, nominal + 1 + s // self.step_every)
        raise ValueError(f"unknown form: {self.form}")


class _Drift(gym.Wrapper):
    nature = None

    def __init__(self, env, schedule: DriftSchedule, k_star: int = 0):
        super().__init__(env)
        self.schedule = schedule
        self.k_star = k_star

    def _apply(self):
        w = self.env.unwrapped
        self._set(w, self.schedule.value(self._nominal(w), w.episode, w.t, w.cfg.T))

    def _info(self, info):
        w = self.env.unwrapped
        info["drift"] = {
            "nature": self.nature, "k_star": self.k_star if self.nature != "latency" else None,
            "tau_global": self.schedule.onset(w.cfg.T),
            "active": w.episode * w.cfg.T + w.t >= self.schedule.onset(w.cfg.T),
        }
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply()
        return obs, self._info(info)

    def step(self, action):
        self._apply()
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r, term, trunc, self._info(info)


class DynamicsDrift(_Drift):
    """Dynamics drift: the hidden force psi[k*] changes."""
    nature = "dynamics"

    def _nominal(self, w):
        return 0.0

    def _set(self, w, value):
        psi = np.zeros(len(w.cfg.z))
        psi[self.k_star] = value
        w.set_hidden(psi=psi)


class CostDrift(_Drift):
    """Cost drift: the damage rate theta[k*] changes."""
    nature = "cost"

    def _nominal(self, w):
        return w.cfg.theta0[self.k_star]

    def _set(self, w, value):
        theta = np.array(w.cfg.theta0, float)
        theta[self.k_star] = value
        w.set_hidden(theta=theta)


class LatencyDrift(_Drift):
    """Latency drift: the command delay d changes (global, k* is meaningless)."""
    nature = "latency"

    def _nominal(self, w):
        return 0

    def _set(self, w, value):
        w.set_hidden(d=int(value))
