"""Comparison of Agents 1 and 2: detection, false alarms, paired statistics, guards. Synthetic data, fast."""
import json

import numpy as np
import pytest

from toyworld.compare import (COMPARISON_SEEDS, calm_false_alarm_gap, contrast, holm_adjust, load_thresholds,
                              require_d9_recorded, seed_vectors, verdict_b)
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


def b_runs(seeds, regimes=("R2c", "R3c"), a_kw=None, b_kw=None, a_outcomes=None, b_outcomes=None):
    """Agent 1 and Agent 2 over the regimes of prediction B. Outcomes: one value for all post-onset episodes."""
    runs = {}
    for g in regimes:
        for s in seeds:
            runs[("agent1_alert", g, s)] = [row(g, e, 4, outcome=(a_outcomes or {}).get(g, "success") if e >= 4 else "success",
                                                **(a_kw or {})) for e in range(10)]
            runs[("agent2_cusum", g, s)] = [row(g, e, 4, outcome=(b_outcomes or {}).get(g, "success") if e >= 4 else "success",
                                                **(b_kw or {})) for e in range(10)]
    return runs


def b_contrasts(runs, seeds, regimes=("R2c", "R3c")):
    return [contrast(runs, "agent1_alert", "agent2_cusum", g, seeds, reps=300) for g in regimes]


def test_contrast_signs():
    seeds = list(range(20))
    runs = b_runs(seeds, a_kw={"ccum": 1.0}, b_kw={"ccum": 2.0})
    c = contrast(runs, "agent1_alert", "agent2_cusum", "R3c", seeds, reps=300)
    assert c["ccum_gap_rel"]["value"] == pytest.approx(0.5)             # Agent 1 halves the damage
    assert c["failure_gap"]["value"] == 0.0
    assert c["success_a"] == c["success_b"] == 1.0


def test_verdict_b_needs_failure_gap_as_well_as_damage():
    seeds = list(range(20))
    runs = b_runs(seeds, a_kw={"ccum": 1.0}, b_kw={"ccum": 2.0})        # damage halved, failure gap 0 < 10 points
    v = verdict_b(b_contrasts(runs, seeds))
    assert v["supported"] is False and v["met"] == {"R2c": False, "R3c": False}


def test_verdict_b_supported_when_both_regimes_meet_every_criterion():
    seeds = list(range(20))
    runs = b_runs(seeds, a_kw={"ccum": 1.0}, b_kw={"ccum": 2.0}, b_outcomes={"R2c": "failure", "R3c": "failure"})
    v = verdict_b(b_contrasts(runs, seeds))
    assert v["met"] == {"R2c": True, "R3c": True} and v["supported"] is True


def test_verdict_b_needs_both_regimes():
    seeds = list(range(20))
    runs = b_runs(seeds, a_kw={"ccum": 1.0}, b_kw={"ccum": 2.0}, b_outcomes={"R2c": "failure"})   # R3c: no failure gap
    v = verdict_b(b_contrasts(runs, seeds))
    assert v["met"] == {"R2c": True, "R3c": False} and v["supported"] is False


def test_verdict_b_success_guard_blocks_a_gain_obtained_by_retreating():
    seeds = list(range(20))
    runs = {}
    for g in ("R2c", "R3c"):
        for s in seeds:
            # Agent 1 retreats in every post-onset episode: no failure, little damage, no success either.
            runs[("agent1_alert", g, s)] = [row(g, e, 4, outcome="retreat" if e >= 4 else "success", ccum=0.5) for e in range(10)]
            # Agent 2 fails in a third of the post-onset episodes and succeeds in the others.
            runs[("agent2_cusum", g, s)] = [row(g, e, 4, outcome=("failure" if e % 3 == 0 else "success") if e >= 4 else "success",
                                                ccum=2.0) for e in range(10)]
    cs = b_contrasts(runs, seeds)
    assert all(c["failure_gap"]["value"] >= 0.10 and c["ccum_gap_rel"]["value"] >= 0.15 for c in cs)
    assert all(c["success_a"] - c["success_b"] < -0.05 for c in cs)
    assert verdict_b(cs)["supported"] is False


def test_verdict_b_requires_significance_after_holm():
    rng = np.random.default_rng(3)
    seeds = list(range(8))
    runs = {}
    for g in ("R2c", "R3c"):
        for s in seeds:                                                 # noisy gap with a large mean but a wide interval
            runs[("agent1_alert", g, s)] = run_rows(g, 4, ccum=float(rng.uniform(0.0, 2.0)))
            runs[("agent2_cusum", g, s)] = run_rows(g, 4, ccum=float(rng.uniform(0.0, 2.0)), outcome="failure")
    cs = b_contrasts(runs, seeds)
    v = verdict_b(cs)
    assert all(c["failure_gap"]["value"] == 1.0 for c in cs)
    assert v["supported"] == (all(p < 0.05 for p in v["holm_ccum"].values())
                              and all(c["ccum_gap_rel"]["value"] >= 0.15 for c in cs))


def test_verdict_b_only_judges_r2c_and_r3c():
    seeds = list(range(5))
    runs = b_runs(seeds, regimes=("R2c",))
    with pytest.raises(ValueError):
        verdict_b(b_contrasts(runs, seeds, regimes=("R2c",)))


def test_closed_loop_delay_gap_is_descriptive():
    seeds = list(range(20))
    runs = fake_runs(seeds, {}, {}, alerts_a={4: 5}, alerts_b={4: 40})
    c = contrast(runs, "agent1_alert", "agent2_cusum", "R3c", seeds, reps=300)
    assert c["delay_gap"]["value"] == 35.0 and c["censored_a"] == 0.0


def test_summaries_wait_for_d9(tmp_path):
    p = tmp_path / "protocol.md"
    with pytest.raises(RuntimeError):
        require_d9_recorded(p)                                          # no file
    p.write_text("## 3.6\nD9-STATUS: PROPOSED\n")
    with pytest.raises(RuntimeError):
        require_d9_recorded(p)
    p.write_text("## 3.6\nD9-STATUS: RECORDED\n")
    require_d9_recorded(p)


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
