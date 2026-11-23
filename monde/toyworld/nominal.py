"""Nominal reference of the triggers: residual scales and threshold grids (docs/agents.md, section 12).

Everything here is computed on the NOMINAL world (regime R0, no drift) with the pretrained model, on calibration
seeds only: never on a drifted run, never on validation or test seeds. It gives

  * sigma_v, sigma_n: root-mean-square one-step residuals of the model, which standardise Agent 2's statistic;
  * the threshold grid of each agent. A candidate is defined by the false-alarm rate per nominal episode it would
    have had: the threshold is the (1 - rate) quantile of the per-episode peak of the statistic. The same six rates
    are used for both agents, so the two grids are comparable and neither is hand-picked.

The result is written to `protocol/nominal_reference.json` (tracked by git) with the hash of the model it was computed
on; the tuning script refuses to run with another model.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .adaptive import run_agent
from .detectors import CusumTrigger, HarmExcessTrigger, Recorder
from .seeds import CALIBRATION_SEEDS

REFERENCE = Path(__file__).resolve().parent.parent / "protocol" / "nominal_reference.json"
FALSE_ALARM_RATES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)     # fixed a priori, one candidate each
NOMINAL_SEEDS = CALIBRATION_SEEDS[:10]
NOMINAL_EPISODES = 20


def model_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def record_nominal(model, seeds=NOMINAL_SEEDS, n_episodes=NOMINAL_EPISODES):
    """Runs the frozen agent in R0 and returns the list of episodes, each a list of `StepRecord`."""
    episodes = []
    for seed in seeds:
        rec = Recorder()
        run_agent("R0", seed, model, rec, n_episodes=n_episodes)
        episodes += rec.episodes
    return episodes


def residual_scales(episodes):
    steps = [r for ep in episodes for r in ep]
    dv2 = np.mean([float(np.dot(r.dv_resid, r.dv_resid)) for r in steps]) / 2.0
    n2 = np.mean([(float(r.obs[4]) - r.n_pred) ** 2 for r in steps])
    return float(np.sqrt(dv2)), float(np.sqrt(n2))


def peaks(trigger, episodes):
    """Per-episode peak of the trigger's statistic over recorded episodes (the trigger must not alert: replay only)."""
    trigger.reset_run()
    for ep in episodes:
        trigger.reset_episode()
        for r in ep:
            trigger.step(r)
        trigger.end_episode()
    return np.array(trigger.peaks)


def threshold_grid(peak_values, rates=FALSE_ALARM_RATES):
    return [float(np.quantile(peak_values, 1.0 - r)) for r in rates]


@dataclass
class NominalReference:
    model_sha256: str
    seeds: list
    n_episodes: int
    sigma_v: float
    sigma_n: float
    rates: list
    agent1_thresholds: list
    agent2_thresholds: list

    def save(self, path=REFERENCE):
        Path(path).parent.mkdir(exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @staticmethod
    def load(path=REFERENCE):
        return NominalReference(**json.loads(Path(path).read_text()))


def build_reference(model, episodes, model_sha, seeds=NOMINAL_SEEDS, n_episodes=NOMINAL_EPISODES):
    sv, sn = residual_scales(episodes)
    g1 = threshold_grid(peaks(HarmExcessTrigger(), episodes))
    g2 = threshold_grid(peaks(CusumTrigger(sv, sn), episodes))
    return NominalReference(model_sha, list(seeds), n_episodes, sv, sn, list(FALSE_ALARM_RATES), g1, g2)
