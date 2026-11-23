"""Comparison of Agents 1 and 2: detection, false alarms, paired statistics, guards. Synthetic data, fast."""
import json

import numpy as np
import pytest

from toyworld.compare import (COMPARISON_SEEDS, calm_false_alarm_gap, contrast, holm_adjust, load_thresholds, seed_vectors,
                              verdict_b, verdict_c)
from toyworld.metrics import detection, false_alarms, paired_bootstrap_stat
from toyworld.seeds import TEST_SEEDS, VALIDATION_SEEDS
from toyworld.tuning import BUDGET


def row(regime, ep, tau, outcome="success", ccum=1.0, steps=100, alert=None, revised=False):
    return {"regime": regime, "seed": 0, "episode": ep, "tau_ep": tau, "ccum": ccum, "outcome": outcome,
            "steps": steps, "alert_step": alert, "revised": revised}


def run_rows(regime, tau, alerts=None, n=10, **kw):
    alerts = alerts or {}
    return [row(regime, e, tau, alert=alerts.get(e), **kw) for e in range(n)]


# ---------- detection delay (decision D4) ----------

def test_delay_counts_steps_from_the_start_of_the_onset_episode():
    rows = run_rows("R3c", 4, alerts={6: 30})                         # two full episodes after the onset, then step 30
    assert detection(rows) == {"delay": 100 + 100 + 30, "censored": False}


def test_alert_in_the_onset_episode_gives_its_step():
    assert detection(run_rows("R3c", 4, alerts={4: 12})) == {"delay": 12, "censored": False}


def test_no_alert_after_the_onset_is_censored_at_the_end_of_the_run():
    assert detection(run_rows("R3c", 4, alerts={1: 50})) == {"delay": 6 * 100, "censored": True}


def test_alert_before_the_onset_is_not_a_negative_delay():
    d = detection(run_rows("R3c", 4, alerts={2: 10, 7: 5}))
    assert d == {"delay": 3 * 100 + 5, "censored": False}


def test_no_onset_no_delay():
    assert detection(run_rows("R0", None)) is None


# ---------- false alarms ----------

def test_false_alarms_in_calm_regimes_are_every_alert():
    assert false_alarms(run_rows("R0", None, alerts={2: 5, 9: 7})) == (2, 10)
    assert false_alarms(run_rows("R1", 4, alerts={6: 5})) == (1, 10)


def test_false_alarms_in_harmful_regimes_are_the_pre_onset_alerts():
    assert false_alarms(run_rows("R3c", 4, alerts={1: 5, 6: 5})) == (1, 4)


# ---------- paired statistic ----------

def test_paired_bootstrap_stat_detects_a_constant_gap():
    a, b = np.full(40, 1.0), np.full(40, 3.0)
    r = paired_bootstrap_stat(a, b, lambda x, y: float(y.mean() - x.mean()), reps=500)
    assert r["value"] == 2.0 and r["lo"] == r["hi"] == 2.0 and r["p"] == 0.0


def test_paired_bootstrap_stat_no_gap_is_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(size=60)
    r = paired_bootstrap_stat(a, a + rng.normal(scale=0.5, size=60), lambda x, y: float(y.mean() - x.mean()), reps=1000)
    assert r["lo"] < 0 < r["hi"] and r["p"] > 0.05


# ---------- contrasts and verdicts ----------

def fake_runs(seeds, a_kw, b_kw, regime="R3c", alerts_a=None, alerts_b=None):
    runs = {}
    for s in seeds:
        runs[("agent1_alert", regime, s)] = run_rows(regime, 4, alerts=alerts_a, **a_kw)
        runs[("agent2_cusum", regime, s)] = run_rows(regime, 4, alerts=alerts_b, **b_kw)
    return runs


def test_contrast_signs_and_verdict_b():
    seeds = list(range(20))
    runs = fake_runs(seeds, {"ccum": 1.0}, {"ccum": 2.0})
    c = contrast(runs, "agent1_alert", "agent2_cusum", "R3c", seeds, reps=300)
    assert c["ccum_gap_rel"]["value"] == pytest.approx(0.5)             # Agent 1 halves the damage
    assert c["failure_gap"]["value"] == 0.0
    assert verdict_b(c) is False                                         # lethal regime: failure gap 0 < 10 points


def test_verdict_b_met_when_both_criteria_hold():
    seeds = list(range(20))
    runs = {}
    for s in seeds:
        runs[("agent1_alert", "R3c", s)] = run_rows("R3c", 4, ccum=1.0)
        runs[("agent2_cusum", "R3c", s)] = run_rows("R3c", 4, ccum=2.0, outcome="failure")
    c = contrast(runs, "agent1_alert", "agent2_cusum", "R3c", seeds, reps=300)
    assert c["failure_gap"]["value"] == 1.0 and verdict_b(c) is True


def test_delay_gap_and_verdict_c():
    seeds = list(range(20))
    runs = fake_runs(seeds, {}, {}, alerts_a={4: 5}, alerts_b={4: 40})
    c = contrast(runs, "agent1_alert", "agent2_cusum", "R3c", seeds, reps=300)
    assert c["delay_gap"]["value"] == 35.0 and c["censored_a"] == 0.0
    assert verdict_c(c, calm_false_alarm_gap=0.06) is True
    assert verdict_c(c, calm_false_alarm_gap=0.0) is False               # faster only by firing as often: not enough


def test_calm_false_alarm_gap():
    seeds = list(range(10))
    runs = {}
    for s in seeds:
        for g in ("R0", "R1"):
            runs[("agent1_alert", g, s)] = run_rows(g, None, alerts={0: 3}, n=10)       # 10% of episodes
            runs[("agent2_cusum", g, s)] = run_rows(g, None, alerts={0: 3, 1: 3, 2: 3}, n=10)   # 30%
    assert calm_false_alarm_gap(runs, "agent1_alert", "agent2_cusum", seeds) == pytest.approx(0.2)


def test_holm_adjust_orders_and_scales():
    cs = [{"regime": "R2d", "x": {"p": 0.01}}, {"regime": "R2c", "x": {"p": 0.04}}, {"regime": "R3d", "x": {"p": 0.001}}]
    adj = holm_adjust(cs, "x", ("R2d", "R2c"))
    assert set(adj) == {"R2d", "R2c"} and adj["R2d"] == pytest.approx(0.02) and adj["R2c"] == pytest.approx(0.04)


# ---------- seeds and guards ----------

def test_comparison_seeds_exclude_tuning_and_test_seeds():
    assert len(COMPARISON_SEEDS) == 50
    assert not set(COMPARISON_SEEDS) & set(BUDGET.seeds)
    assert not set(COMPARISON_SEEDS) & set(TEST_SEEDS)
    assert set(COMPARISON_SEEDS) <= set(VALIDATION_SEEDS)


def test_seed_vectors_read_post_onset_episodes():
    runs = {("agent1_alert", "R3c", 0): [row("R3c", e, 4, outcome="failure" if e >= 4 else "success") for e in range(10)]}
    v = seed_vectors(runs, "agent1_alert", "R3c", [0])
    assert v["failure"][0] == 1.0 and v["success"][0] == 0.0


def test_load_thresholds_refuses_an_untuned_comparison(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_thresholds(tmp_path / "none.json", tmp_path / "ledger.jsonl")
    p, led = tmp_path / "t.json", tmp_path / "ledger.jsonl"
    p.write_text(json.dumps({"agent1_alert": {"threshold": 0.1}}))
    with pytest.raises(ValueError):
        load_thresholds(p, led)                                          # agent 2 missing
    p.write_text(json.dumps({"agent1_alert": {"threshold": 0.1}, "agent2_cusum": {"threshold": 9.0}}))
    led.write_text("")
    with pytest.raises(ValueError):
        load_thresholds(p, led)                                          # empty ledger: nothing was tuned
    entry = {"budget": {"candidates": 6}, "candidates": [1], "scores": [0], "chosen": 0}
    led.write_text(json.dumps({"agent": "agent1_alert", **entry}) + "\n")
    with pytest.raises(ValueError):
        load_thresholds(p, led)                                          # only agent 1 in the ledger
    led.write_text("".join(json.dumps({"agent": a, **entry}) + "\n" for a in ("agent1_alert", "agent2_cusum")))
    assert load_thresholds(p, led) == {"agent1_alert": 0.1, "agent2_cusum": 9.0}
