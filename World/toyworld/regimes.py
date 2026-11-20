"""The retained regime grid (checklist 0.4 x 0.5): 8 cells.

R0 no change; R1 harmless but surprising (dynamics only, phantom zone Z3);
R2 harmful abrupt and R3 harmful slow, each in dynamics (d), cost (c) and latency (l).
Zone indices: Z1 = 0 (opening A, direct route), Z2 = 1, Z3 = 2 (phantom).
"""
import numpy as np

from .config import WorldConfig
from .drift import CostDrift, DriftSchedule, DynamicsDrift, LatencyDrift
from .world import ToyWorld

# Durations are proportional to n_episodes (at 20 episodes: tau in {5..10}, ramps of 5 and 10 episodes).
ONSET_FRAC = (0.25, 0.5)     # tau (episode) is drawn between 25% and 50% of the episodes of the run

# id -> (wrapper class, k*, form, final value, drift duration as a fraction of n_episodes)
#
# Values ADJUSTED relative to the checklist grid, after measuring with the frozen agent
# (scripts/explore_regimes.py). Reason: the original values did not produce the extra damage
# X = 10% required by the grid.
#   R2l  d: 0 -> 3 steps   =>  0 -> 10 steps   (d = 3 gives only +4% damage)
#   R3d  final psi1 0.1    =>  0.25            (0.1 gives +7%; 0.25 about +17%, SNR < 1 inside the zone)
#   R3c  theta1 1 -> 3 over 5 episodes  =>  1 -> 2 over 10 episodes (+0.1 damage units/s per episode
#        = sigma_n, at 20 episodes)
#   R3l  d: +1 step every 2 episodes up to 3  =>  +1 step per episode up to 10 (at 20 episodes)
REGIMES = {
    "R0": None,
    "R1": (DynamicsDrift, 2, "jump", 1.0, 0),      # phantom zone: psi3 0 -> 1.0 m/s^2, theta3 = 0
    "R2d": (DynamicsDrift, 0, "jump", 1.0, 0),     # psi1 0 -> 1.0 m/s^2
    "R2c": (CostDrift, 0, "jump", 3.0, 0),         # theta1 1 -> 3 damage units/s
    "R2l": (LatencyDrift, None, "jump", 10, 0),    # d 0 -> 10 steps
    "R3d": (DynamicsDrift, 0, "ramp", 0.25, 0.25),  # psi1 0 -> 0.25 m/s^2 over 25% of the run (5 episodes at 20)
    "R3c": (CostDrift, 0, "ramp", 2.0, 0.5),        # theta1 1 -> 2 damage units/s over 50% of the run (10 at 20)
    "R3l": (LatencyDrift, None, "steps", 10, 0.5),  # d 0 -> 10 over 50% of the run (+1 step per episode at 20)
}


def onset_window(n_episodes: int) -> tuple:
    """Window of tau in episodes: [25%, 50%] of n_episodes (20 episodes => {5,...,10})."""
    lo = max(1, round(ONSET_FRAC[0] * n_episodes))
    hi = max(lo, round(ONSET_FRAC[1] * n_episodes))
    return lo, hi


def onset_episode(seed: int, n_episodes: int = 20) -> int:
    """tau (episode) drawn in the proportional window, determined by the run seed."""
    lo, hi = onset_window(n_episodes)
    return int(np.random.default_rng([seed, 999]).integers(lo, hi + 1))


def make_env(regime="R0", seed=0, n_episodes=20, config: WorldConfig | None = None,
             onset_ep: int | None = None, onset_step: int = 0):
    cfg = config or WorldConfig()
    env = ToyWorld(cfg, n_episodes=n_episodes, run_seed=seed)
    spec = REGIMES[regime]
    if spec is None:
        return env
    cls, k_star, form, target, frac = spec
    duration = max(1, int(round(frac * n_episodes * cfg.T)))        # length of a slow drift, in steps
    sched = DriftSchedule(
        onset_episode=onset_episode(seed, n_episodes) if onset_ep is None else onset_ep,
        onset_step=onset_step, form=form, target=target,
        ramp_steps=duration,
        step_every=max(1, duration // max(int(target), 1)),         # "steps": reaches `target` after `duration`
    )
    return cls(env, sched, k_star=0 if k_star is None else k_star)
