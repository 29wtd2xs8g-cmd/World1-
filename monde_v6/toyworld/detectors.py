"""Agents 1 and 2: the two triggers under comparison (checklist 2.4; docs/agents.md, section 12).

Both are `StatisticTrigger`s: at every step they update one scalar statistic from the observable `StepRecord` and alert
when it exceeds a threshold. They differ only in what the statistic measures, and each has exactly one tuned
parameter, its threshold (equal tuning budget, `tuning.py`).

  HarmExcessTrigger  (Agent 1, perceptual alert). The statistic is the harm the agent has felt beyond what its own
                     model predicted, as a fraction of the margin it has left:
                         E[t] = dt * sum_{s<=t} (n_obs[s] - n_pred[s]),      stat[t] = E[t] / max(b_hat[t], eps).
                     The coupling to the margin is the division by b_hat: the same excess is tolerated when the
                     margin is large and alerts when it is nearly spent. It reads the nociceptive channel only.

  CusumTrigger       (Agent 2, prediction error). The statistic is a one-sided CUSUM of the standardised one-step
                     prediction error of the world model over both its heads (velocity change and nociception):
                         s[t] = (|dv_resid|^2 / sigma_v^2 + (n_obs - n_pred)^2 / sigma_n^2) / 3 - 1,
                         S[t] = max(0, S[t-1] + s[t] - k),
                     where sigma_v and sigma_n are the nominal residual scales measured once on the pretrained
                     model (`nominal.py`) and k is a fixed slack. It ignores the margin.

Neither reads anything but a `StepRecord` (observable quantities and the agent's own model), so the leak tests of
`triggers.py` cover them.
"""
import math

import numpy as np

from .config import WorldConfig
from .triggers import StepRecord, Trigger

CUSUM_SLACK = 0.5          # fixed a priori, not tuned
MARGIN_FLOOR = 0.01        # eps of Agent 1, as a fraction of D


class StatisticTrigger(Trigger):
    """Alert when `statistic(rec)` exceeds `threshold`. `peak` is the largest statistic of the current episode."""

    def __init__(self, threshold: float = math.inf):
        self.threshold = float(threshold)
        self.peak = -math.inf
        self.peaks = []                     # one value per finished episode (used by the nominal calibration)

    def reset_run(self):
        self.peaks = []

    def reset_episode(self):
        self.peak = -math.inf
        self._reset_statistic()

    def end_episode(self):
        self.peaks.append(self.peak)

    def _reset_statistic(self):
        pass

    def statistic(self, rec: StepRecord) -> float:
        raise NotImplementedError

    def step(self, rec: StepRecord) -> bool:
        s = self.statistic(rec)
        self.peak = max(self.peak, s)
        return s > self.threshold


class HarmExcessTrigger(StatisticTrigger):
    name = "agent1_alert"

    def __init__(self, threshold: float = math.inf, config: WorldConfig | None = None):
        super().__init__(threshold)
        cfg = config or WorldConfig()
        self.dt, self.floor = cfg.dt, MARGIN_FLOOR * cfg.D
        self.excess = 0.0

    def _reset_statistic(self):
        self.excess = 0.0

    def statistic(self, rec):
        self.excess += self.dt * (float(rec.obs[4]) - rec.n_pred)
        return self.excess / max(rec.b_hat, self.floor)


class CusumTrigger(StatisticTrigger):
    name = "agent2_cusum"

    def __init__(self, sigma_v: float, sigma_n: float, threshold: float = math.inf, slack: float = CUSUM_SLACK):
        super().__init__(threshold)
        self.sigma_v, self.sigma_n, self.slack = float(sigma_v), float(sigma_n), float(slack)
        self.cusum = 0.0

    def _reset_statistic(self):
        self.cusum = 0.0

    def statistic(self, rec):
        zv = float(np.dot(rec.dv_resid, rec.dv_resid)) / self.sigma_v ** 2
        zn = (float(rec.obs[4]) - rec.n_pred) ** 2 / self.sigma_n ** 2
        self.cusum = max(0.0, self.cusum + (zv + zn) / 3.0 - 1.0 - self.slack)
        return self.cusum


class Recorder(Trigger):
    """Never alerts, never revises; keeps every `StepRecord` of every episode (nominal calibration, replay)."""
    name = "recorder"

    def __init__(self):
        self.episodes = []

    def reset_episode(self):
        self.episodes.append([])

    def step(self, rec):
        self.episodes[-1].append(rec)
        return False

    def revise_at_end(self, alerted):
        return False
