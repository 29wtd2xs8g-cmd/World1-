"""Agents 1 and 2 (triggers), nominal reference and threshold grids. Fast versions."""
import math
from collections import namedtuple

import numpy as np
import pytest
import torch

from toyworld import WorldConfig
from toyworld.adaptive import run_agent
from toyworld.data import collect
from toyworld.detectors import CusumTrigger, HarmExcessTrigger, Recorder
from toyworld.model import WorldModel
from toyworld.nominal import (FALSE_ALARM_RATES, NominalReference, model_hash, peaks, residual_scales,
                              threshold_grid)
from toyworld.triggers import STEP_RECORD_FIELDS, NeverTrigger, StepRecord, Trigger

CFG = WorldConfig()


def rec(n_obs=0.0, n_pred=0.0, dv=(0.0, 0.0), b_hat=CFG.D, t=1):
    return StepRecord(t=t, obs=np.array([0, 0, 0, 0, n_obs]), n_pred=n_pred, dv_resid=np.array(dv, float), b_hat=b_hat)


@pytest.fixture(scope="module")
def small_model():
    d = collect(range(4000, 4008), segments_per_seed=10, seg_len=100)
    torch.manual_seed(0)
    m = WorldModel()
    m.fit(d, epochs=6, seed=3)
    return m.eval()


# ---------- Agent 1: harm excess coupled to the margin ----------

def test_agent1_statistic_is_excess_over_margin():
    a = HarmExcessTrigger()
    a.reset_episode()
    s = [a.statistic(rec(n_obs=0.5, n_pred=0.3)) for _ in range(5)]
    assert s[-1] == pytest.approx(5 * CFG.dt * 0.2 / CFG.D)
    assert all(b > a_ for a_, b in zip(s, s[1:]))                      # excess accumulates


def test_agent1_no_excess_no_alert():
    a = HarmExcessTrigger(threshold=0.0)
    a.reset_episode()
    assert not any(a.step(rec(n_obs=0.3, n_pred=0.3)) for _ in range(200))


def test_agent1_is_coupled_to_the_margin():
    big, small = HarmExcessTrigger(), HarmExcessTrigger()
    big.reset_episode(), small.reset_episode()
    assert small.statistic(rec(0.5, 0.3, b_hat=0.3)) > 5 * big.statistic(rec(0.5, 0.3, b_hat=CFG.D))


def test_agent1_margin_floor_handles_spent_margin():
    a = HarmExcessTrigger(threshold=0.5)
    a.reset_episode()
    assert a.step(rec(0.5, 0.3, b_hat=-1.0)) is True                   # no division by zero or by a negative margin


def test_agent1_ignores_dynamics_residual():
    a = HarmExcessTrigger()
    a.reset_episode()
    assert a.statistic(rec(0.3, 0.3, dv=(5.0, 5.0))) == 0.0


# ---------- Agent 2: CUSUM of the standardised prediction error ----------

def test_agent2_stays_at_zero_without_error():
    c = CusumTrigger(sigma_v=0.1, sigma_n=0.1)
    c.reset_episode()
    assert [c.statistic(rec()) for _ in range(50)] == [0.0] * 50


def test_agent2_accumulates_a_persistent_error_and_resets():
    c = CusumTrigger(sigma_v=0.1, sigma_n=0.1, threshold=5.0)
    c.reset_episode()
    fired = [c.step(rec(dv=(0.3, 0.0))) for _ in range(20)]            # 3 sigma on the velocity channel, every step
    assert any(fired) and not fired[0]
    c.reset_episode()
    assert c.cusum == 0.0 and c.peak == -math.inf


def test_agent2_sees_cost_drift_through_n_and_ignores_the_margin():
    def first(b_hat):
        c = CusumTrigger(sigma_v=0.1, sigma_n=0.1)
        c.reset_episode()
        return c.statistic(rec(n_obs=0.9, n_pred=0.1, b_hat=b_hat))
    assert first(CFG.D) > 0.0
    assert first(0.2) == first(CFG.D)


# ---------- interface, leaks ----------

def test_triggers_read_only_the_step_record_fields():
    Only = namedtuple("Only", STEP_RECORD_FIELDS)
    r = rec(0.5, 0.3, dv=(0.1, 0.1))
    only = Only(*[getattr(r, f) for f in STEP_RECORD_FIELDS])
    for trig in (HarmExcessTrigger(0.1), CusumTrigger(0.1, 0.1, 0.1)):
        trig.reset_episode()
        trig.step(only)                                                # raises AttributeError if anything else is read


def test_triggers_do_not_revise_without_alert():
    for trig in (HarmExcessTrigger(), CusumTrigger(0.1, 0.1)):
        assert trig.revise_at_end(False) is False and trig.revise_at_end(True) is True


# ---------- runner integration ----------

def test_end_episode_is_called_once_per_episode_also_for_the_oracle(small_model):
    class Count(Trigger):
        def __init__(self):
            self.n = 0

        def end_episode(self):
            self.n += 1
    for oracle in (False, True):
        c = Count()
        run_agent("R0", 1000, small_model, c, oracle=oracle, n_episodes=2)
        assert c.n == 2


def test_replay_gives_the_live_peaks(small_model):
    seed = 1001
    live = HarmExcessTrigger()                                         # threshold inf: never alerts, runs like frozen
    run_agent("R0", seed, small_model, live, n_episodes=2)
    recorder = Recorder()
    run_agent("R0", seed, small_model, recorder, n_episodes=2)
    assert np.allclose(peaks(HarmExcessTrigger(), recorder.episodes), live.peaks)
    sv, sn = residual_scales(recorder.episodes)
    live2 = CusumTrigger(sv, sn)
    run_agent("R0", seed, small_model, live2, n_episodes=2)
    assert np.allclose(peaks(CusumTrigger(sv, sn), recorder.episodes), live2.peaks)


def test_a_low_threshold_makes_the_agent_retreat_every_episode(small_model):
    rows = run_agent("R0", 1002, small_model, HarmExcessTrigger(threshold=-1e9), n_episodes=2)
    assert [r["outcome"] for r in rows] == ["retreat", "retreat"]
    assert all(r["alert_step"] == 1 for r in rows)


def test_an_infinite_threshold_reproduces_the_frozen_run(small_model):
    a = run_agent("R3c", 1003, small_model, HarmExcessTrigger(), n_episodes=3)
    b = run_agent("R3c", 1003, small_model, NeverTrigger(), n_episodes=3)
    assert [(r["ccum"], r["outcome"]) for r in a] == [(r["ccum"], r["outcome"]) for r in b]


# ---------- nominal reference and grids ----------

def test_threshold_grid_matches_the_false_alarm_rates():
    p = np.random.default_rng(0).normal(size=1000)
    g = threshold_grid(p)
    assert len(g) == len(FALSE_ALARM_RATES) == 6 and g == sorted(g, reverse=True)
    for t, r in zip(g, FALSE_ALARM_RATES):
        assert np.mean(p > t) == pytest.approx(r, abs=0.005)


def test_reference_roundtrip_and_model_hash(tmp_path):
    ref = NominalReference("abc", [1, 2], 20, 0.1, 0.2, list(FALSE_ALARM_RATES), [3.0] * 6, [4.0] * 6)
    ref.save(tmp_path / "r.json")
    assert NominalReference.load(tmp_path / "r.json") == ref
    f = tmp_path / "m.pt"
    f.write_bytes(b"model")
    assert model_hash(f) == model_hash(f) and len(model_hash(f)) == 16
