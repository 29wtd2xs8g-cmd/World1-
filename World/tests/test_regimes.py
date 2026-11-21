"""Verification of every regime (checklist 1.3) with the frozen agent (no adaptation).

Criteria of the retained grid (0.4 x 0.5): X = 10%; masking = one-step residual below the noise.
Standardised residual r = (true v - v predicted by the frozen nominal model) / (sqrt(dt) * sigma).
Residual excess = sqrt(max(0, mean |r|^2 - 2)): 0 without drift, grows with the signal.
Dynamics SNR = psi * phi * sqrt(dt) / sigma (~ excess inside the zone); cost SNR = dtheta * phi / sigma_n.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from toyworld import WorldConfig
from toyworld.evaluate import run_frozen
from toyworld.regimes import REGIMES, make_env

CFG = WorldConfig()
FREE = replace(CFG, D=1e9)       # damage measured without failure cutoff, so that Ccum is not capped at D
SEEDS = range(12)
X = 0.10
C1 = 2.5027                  # damage of one crossing of the SCRIPTED agent (these tests use it)
C1_BASELINE = 1.4908         # damage of one crossing of the baseline agent; D = 2 * C1_BASELINE


@lru_cache(maxsize=None)
def rows(reg):
    return run_frozen(reg, seeds=SEEDS, n_episodes=20, config=FREE)


def mean_ccum(rs):
    return float(np.mean([r["ccum"] for r in rs]))


def fail_rate(rs):
    """Failure = damage reaching D. The scripted agent ignores D, so this equals the failure rate with a real D."""
    return float(np.mean([r["ccum"] >= CFG.D for r in rs]))


def excess(rs, zone=None):
    r2 = []
    for r in rs:
        if zone is None:
            r2 += [float(x @ x) for x in r["resid_all"]]
        else:
            r2 += [float(x @ x) for x, _ in r["resid_zone"][zone]]
    return float(np.sqrt(max(0.0, np.mean(r2) - 2.0))) if r2 else 0.0


def split(reg):
    rs = rows(reg)
    pre = [r for r in rs if r["tau_ep"] is None or r["episode"] < r["tau_ep"]]
    post = [r for r in rs if r["tau_ep"] is not None and r["episode"] >= r["tau_ep"]]
    return pre, post


def by_rel(reg):
    """Episodes aligned on the drift episode: rel = episode - tau_ep >= 0."""
    out = {}
    for r in rows(reg):
        if r["tau_ep"] is not None and r["episode"] >= r["tau_ep"]:
            out.setdefault(r["episode"] - r["tau_ep"], []).append(r)
    return dict(sorted(out.items()))


def first_rel_above(reg, ref):
    for rel, rs in by_rel(reg).items():
        if mean_ccum(rs) >= (1 + X) * ref:
            return rel, rs
    return None, None


R0_CCUM = None


def r0_mean():
    return mean_ccum(rows("R0"))


# ------------------------------------------------------------------ R0
def test_R0_stable_without_drift():
    rs = rows("R0")
    assert fail_rate(rs) == 0.0
    assert abs(r0_mean() - C1) < 0.1
    a = [r["ccum"] for r in rs if r["episode"] < 10]
    b = [r["ccum"] for r in rs if r["episode"] >= 10]
    se = np.sqrt(np.var(a) / len(a) + np.var(b) / len(b))
    assert abs(np.mean(a) - np.mean(b)) < 3 * se          # Ccum stable (3 standard errors)
    assert excess(rs) < 0.3                                # the nominal model predicts well


# ------------------------------------------------------------------ R1: harmless but surprising
def test_R1_surprising_but_harmless():
    pre, post = split("R1")
    assert fail_rate(post) == 0.0 and fail_rate(pre) == 0.0
    se = np.sqrt(np.var([r["ccum"] for r in pre]) / len(pre) + np.var([r["ccum"] for r in post]) / len(post))
    assert abs(mean_ccum(post) - mean_ccum(pre)) < 3 * se       # damage not degraded
    assert excess(pre, zone=2) < 0.3
    assert excess(post, zone=2) > 1.0                           # but the dynamics surprise inside Z3


# ------------------------------------------------------------------ R2: harmful abrupt
def test_R2d_dynamics():
    pre, post = split("R2d")
    assert mean_ccum(post) >= (1 + X) * r0_mean()
    assert fail_rate(post) >= 0.9 > fail_rate(pre)
    assert excess(post, zone=0) > 1.0


def test_R2c_cost():
    pre, post = split("R2c")
    assert 3 * C1_BASELINE > CFG.D                              # a crossing of the baseline through A costs 3*C1' > D
    assert mean_ccum(post) >= (1 + X) * r0_mean()
    assert fail_rate(post) >= 0.9 > fail_rate(pre)
    assert excess(post, zone=0) < 0.3                           # the dynamics did not move


def test_R2l_latency():
    pre, post = split("R2l")
    assert mean_ccum(post) >= (1 + X) * r0_mean()
    assert fail_rate(post) > fail_rate(pre)
    assert excess(post) > 1.0


# ------------------------------------------------------------------ R3: harmful slow / masked
def test_R3d_masked_when_damage_exceeds_X():
    rel, rs = first_rel_above("R3d", r0_mean())
    assert rel is not None, "damage never exceeds R0 by X%: raise the final psi"
    assert excess(rs, zone=0) < 1.0                             # SNR < 1 inside the zone at that moment


def test_R3c_masked_when_damage_exceeds_X():
    rel, rs = first_rel_above("R3c", r0_mean())
    assert rel is not None, "damage never exceeds R0 by X%: raise the final theta"
    sched = make_env("R3c", seed=0).schedule
    mid_step = int(np.mean([r["steps"] for r in rs]) * 0.7)     # average moment of the crossing of Z1
    dtheta = sched.value(1.0, sched.onset_episode + rel, mid_step, CFG.T) - 1.0
    phis = [p for r in rs for _, p in r["resid_zone"][0]]
    snr_cout = dtheta * float(np.mean(phis)) / CFG.sigma_n
    assert snr_cout <= 1.0


def test_R3l_not_maskable_with_this_agent():
    """Characterised result: latency only becomes harmful (+X%) when the one-step error is already
    very visible. Cell R3l exists but is not masked; we write that down rather than force it."""
    rel, rs = first_rel_above("R3l", r0_mean())
    assert rel is not None
    assert excess(rs) > 1.0


# ------------------------------------------------------------------ proportionality to n_episodes
def test_tau_window_and_durations_are_proportional():
    from toyworld.regimes import onset_episode, onset_window
    assert onset_window(20) == (5, 10)                         # prototype values unchanged
    assert onset_window(100) == (25, 50)
    assert all(5 <= onset_episode(s, 20) <= 10 for s in range(50))
    assert all(25 <= onset_episode(s, 100) <= 50 for s in range(50))
    for n in (20, 100):
        T = CFG.T
        assert make_env("R3d", seed=0, n_episodes=n).schedule.ramp_steps == round(0.25 * n * T)
        assert make_env("R3c", seed=0, n_episodes=n).schedule.ramp_steps == round(0.5 * n * T)
        l = make_env("R3l", seed=0, n_episodes=n).schedule                # d reaches 10 after 50% of the run
        assert l.step_every * 10 == round(0.5 * n * T)
    l20 = make_env("R3l", seed=0, n_episodes=20).schedule
    assert l20.step_every == CFG.T                                         # +1 step per episode at 20 episodes


def test_long_run_contains_the_whole_drift():
    """At 100 episodes the slow drift unfolds inside the run (last episode = final value)."""
    env = make_env("R3c", seed=1, n_episodes=100)
    s = env.schedule
    last = s.value(1.0, 99, CFG.T - 1, CFG.T)
    assert last == pytest.approx(2.0)
