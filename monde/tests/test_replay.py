"""Matched-false-alarm replay (decision D9, prediction C): crossings, delays at each level, verdict. Synthetic, fast."""
import numpy as np
import pytest

from toyworld.config import WorldConfig
from toyworld.detectors import CusumTrigger, HarmExcessTrigger
from toyworld.nominal import FALSE_ALARM_RATES, NominalReference
from toyworld.replay import (C_REGIMES, VERDICT_LEVEL, VERDICT_RATE, first_crossings, grid, level_vectors, matched_contrast,
                             replay_run, rows_at, verdict_c)
from toyworld.triggers import StepRecord

CFG = WorldConfig()
N_LEVELS = len(FALSE_ALARM_RATES)


def rec(t, n_obs, n_pred=0.0, dv=(0.0, 0.0), b_hat=None):
    obs = np.array([0.0, 0.0, 0.0, 0.0, n_obs])
    return StepRecord(t=t, obs=obs, n_pred=n_pred, dv_resid=np.array(dv, float), b_hat=CFG.D if b_hat is None else b_hat)


def episode(n_obs_seq, **kw):
    return [rec(t + 1, n, **kw) for t, n in enumerate(n_obs_seq)]


def reference():
    return NominalReference("x", [1], 20, sigma_v=0.05, sigma_n=0.08, rates=list(FALSE_ALARM_RATES),
                            agent1_thresholds=[0.5, 0.4, 0.3, 0.2, 0.1, 0.05], agent2_thresholds=[30.0, 20.0, 10.0, 6.0, 3.0, 1.0])


# ---------- crossings reproduce the closed-loop trigger ----------

def test_first_crossings_agree_with_the_trigger_step_by_step():
    rng = np.random.default_rng(0)
    ep = [rec(t + 1, float(rng.uniform(0, 3)), n_pred=float(rng.uniform(0, 1)), dv=rng.normal(scale=0.08, size=2),
              b_hat=CFG.D - 0.01 * t) for t in range(120)]
    ths = [0.02, 0.1, 0.3, 1.0, 50.0]
    for make in (lambda th: HarmExcessTrigger(th), lambda th: CusumTrigger(0.05, 0.08, th)):
        crossings = first_crossings(make(float("inf")), ep, ths)
        for th, c in zip(ths, crossings):
            trig = make(th)
            trig.reset_episode()
            alert = next((r.t for r in ep if trig.step(r)), None)         # what the closed loop would have recorded
            assert c == alert


def test_higher_thresholds_cross_later_or_never():
    ep = episode([2.0] * 200)
    cr = first_crossings(HarmExcessTrigger(), ep, [0.01, 0.05, 0.2, 0.8, 100.0])
    seen = [c for c in cr if c is not None]
    assert seen == sorted(seen) and cr[-1] is None and cr[0] is not None


def test_crossings_restart_in_every_episode():
    t = HarmExcessTrigger()
    a = first_crossings(t, episode([2.0] * 100), [0.1])
    b = first_crossings(t, episode([2.0] * 100), [0.1])
    assert a == b and a[0] is not None


def test_agent_two_ignores_the_margin_and_agent_one_does_not():
    low_margin = episode([1.0] * 60, b_hat=0.05)
    big_margin = episode([1.0] * 60, b_hat=CFG.D)
    assert first_crossings(HarmExcessTrigger(), low_margin, [0.2])[0] is not None
    assert first_crossings(HarmExcessTrigger(), big_margin, [0.2])[0] is None or \
        first_crossings(HarmExcessTrigger(), big_margin, [0.2])[0] > first_crossings(HarmExcessTrigger(), low_margin, [0.2])[0]
    a = first_crossings(CusumTrigger(0.05, 0.08), low_margin, [5.0])
    b = first_crossings(CusumTrigger(0.05, 0.08), big_margin, [5.0])
    assert a == b


def test_replay_run_stores_one_crossing_list_per_episode_and_agent():
    rows = [{"regime": "R3c", "seed": 0, "episode": e, "tau_ep": 1, "steps": 50, "alert_step": None} for e in range(2)]
    eps = [episode([0.0] * 50), episode([3.0] * 50)]
    out = replay_run(rows, eps, reference())
    assert set(out["cross"]) == {"agent1_alert", "agent2_cusum"}
    assert all(len(v) == 2 and all(len(c) == N_LEVELS for c in v) for v in out["cross"].values())
    assert all(c == [None] * N_LEVELS for c in (out["cross"]["agent1_alert"][0], out["cross"]["agent2_cusum"][0]))
    assert out["cross"]["agent1_alert"][1][-1] is not None                 # loosest level fires on a strong signal
    with pytest.raises(ValueError):
        replay_run(rows, eps[:1], reference())


def test_grid_has_one_threshold_per_level_and_agent():
    g = grid(reference())
    assert len(g["agent1_alert"]) == len(g["agent2_cusum"]) == N_LEVELS


# ---------- delays and false alarms read with the closed-loop functions ----------

def fake_entry(regime, tau, steps=100, n=10, a1=None, a2=None):
    """One frozen run; `a1`, `a2`: {episode: step} of the first alert of each agent at every level."""
    a1, a2 = a1 or {}, a2 or {}
    rows = [{"regime": regime, "seed": 0, "episode": e, "tau_ep": tau, "steps": steps, "alert_step": None} for e in range(n)]
    cross = {"agent1_alert": [[a1.get(e)] * N_LEVELS for e in range(n)],
             "agent2_cusum": [[a2.get(e)] * N_LEVELS for e in range(n)]}
    return {"rows": rows, "cross": cross}


def fake_replay(seeds, per_regime):
    """`per_regime`: regime -> (tau, a1, a2)."""
    return {(g, s): fake_entry(g, tau, a1=a1, a2=a2) for g, (tau, a1, a2) in per_regime.items() for s in seeds}


def test_rows_at_overwrites_only_the_alert_step():
    e = fake_entry("R3c", 4, a1={5: 12})
    r = rows_at(e, "agent1_alert", 2)
    assert r[5]["alert_step"] == 12 and r[4]["alert_step"] is None and r[5]["steps"] == 100
    assert e["rows"][5]["alert_step"] is None                              # the stored run is not modified


def test_level_vectors_delay_false_alarms_and_censoring():
    seeds = [0, 1]
    replay = fake_replay(seeds, {"R3c": (4, {2: 7, 6: 30}, {})})             # alert before the onset (false alarm) and after
    v = level_vectors(replay, "agent1_alert", "R3c", 0, seeds)
    assert v["delay"].tolist() == [200 + 30, 200 + 30] and v["censored"].tolist() == [0.0, 0.0]
    assert (v["fa"], v["fa_n"]) == (2, 8)                                    # 4 pre-onset episodes per seed, one alert each
    w = level_vectors(replay, "agent2_cusum", "R3c", 0, seeds)
    assert w["censored"].tolist() == [1.0, 1.0] and w["delay"].tolist() == [600, 600]


def test_calm_regimes_have_false_alarms_and_no_delay():
    replay = fake_replay([0], {"R0": (None, {3: 5}, {3: 5, 4: 5})})
    c = matched_contrast(replay, "R0", 2, [0], reps=50)
    assert "delay_gap" not in c and c["false_alarm_a"] == pytest.approx(0.1) and c["false_alarm_b"] == pytest.approx(0.2)


# ---------- verdict of prediction C ----------

def all_regimes(seeds, a1, a2, fa_a=None, fa_b=None, overrides=None):
    per = {g: (4, {4: a1}, {4: a2}) if a1 is not None else (4, {}, {4: a2}) for g in C_REGIMES}
    per.update(overrides or {})
    per["R0"] = (None, fa_a or {}, fa_b or {})
    per["R1"] = (4, fa_a or {}, fa_b or {})
    return fake_replay(seeds, per)


def verdict(replay, seeds):
    cs = [matched_contrast(replay, g, VERDICT_LEVEL, seeds, reps=300) for g in (*C_REGIMES, "R0", "R1")]
    return verdict_c(cs)


def test_verdict_level_is_the_five_percent_level():
    assert FALSE_ALARM_RATES[VERDICT_LEVEL] == VERDICT_RATE == 0.05


def test_c_supported_when_agent_one_is_faster_everywhere_at_matched_false_alarms():
    seeds = list(range(20))
    v = verdict(all_regimes(seeds, a1=5, a2=40), seeds)
    assert v["supported"] and v["n_met"] == 5 and v["false_alarms_matched"]


def test_c_not_supported_when_the_gap_is_under_ten_steps():
    seeds = list(range(20))
    v = verdict(all_regimes(seeds, a1=5, a2=12), seeds)
    assert not v["supported"] and v["n_met"] == 0


def test_c_needs_three_of_five_regimes():
    seeds = list(range(20))
    slow = {g: (4, {4: 40}, {4: 40}) for g in C_REGIMES[:3]}                 # no gap in three regimes
    assert verdict(all_regimes(seeds, a1=5, a2=40, overrides=slow), seeds)["n_met"] == 2
    assert not verdict(all_regimes(seeds, a1=5, a2=40, overrides=slow), seeds)["supported"]
    slow = {g: (4, {4: 40}, {4: 40}) for g in C_REGIMES[:2]}
    v = verdict(all_regimes(seeds, a1=5, a2=40, overrides=slow), seeds)
    assert v["n_met"] == 3 and v["supported"]


def test_censored_median_of_agent_one_is_not_a_win():
    seeds = list(range(20))
    v = verdict(all_regimes(seeds, a1=None, a2=40), seeds)                   # Agent 1 never alerts after the onset
    assert v["n_met"] == 0 and not v["supported"]


def test_censored_median_of_agent_two_alone_is_a_valid_lower_bound():
    seeds = list(range(20))
    per = {g: (4, {4: 5}, {}) for g in C_REGIMES}                            # Agent 2 never alerts
    v = verdict(all_regimes(seeds, a1=5, a2=40, overrides=per), seeds)
    assert v["n_met"] == 5 and v["supported"]


def test_c_needs_matched_false_alarms():
    seeds = list(range(20))
    # Agent 2 alerts in 30% of the nominal episodes, Agent 1 in none: the rates are not matched.
    fa_b = {0: 5, 1: 5, 2: 5}
    v = verdict(all_regimes(seeds, a1=5, a2=40, fa_b=fa_b), seeds)
    assert v["n_met"] == 5 and not v["false_alarms_matched"] and not v["supported"]


def test_verdict_c_asks_for_the_five_lethal_regimes_and_the_calm_ones():
    seeds = list(range(5))
    replay = all_regimes(seeds, a1=5, a2=40)
    cs = [matched_contrast(replay, g, VERDICT_LEVEL, seeds, reps=50) for g in C_REGIMES]
    with pytest.raises(ValueError):
        verdict_c(cs)                                                        # R0 and R1 missing
