"""Evaluation protocol (checklist 3.1 to 3.4): seeds, test-seed guard, metrics, statistics, equal tuning budget."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from toyworld import metrics, seeds, tuning

ROOT = Path(__file__).resolve().parent.parent


def row(outcome, ccum, steps=100, regime="R2c", episode=6, tau=5, seed=1, **kw):
    return {"regime": regime, "seed": seed, "episode": episode, "tau_ep": tau, "ccum": ccum, "outcome": outcome,
            "steps": steps, **kw}


# ---------------------------------------------------------------------- seeds
def test_seed_lists_have_final_size_and_keep_their_first_thirty():
    assert len(seeds.VALIDATION_SEEDS) == len(seeds.TEST_SEEDS) == seeds.N_FINAL_SEEDS == 60 and seeds.N_FINAL_SEEDS > 50
    assert seeds.VALIDATION_SEEDS[:30] == tuple(range(2000, 2030)) and seeds.TEST_SEEDS[:30] == tuple(range(3000, 3030))
    assert seeds.BOOTSTRAP_REPS == 5000
    lists = [set(seeds.CALIBRATION_SEEDS), set(seeds.VALIDATION_SEEDS), set(seeds.TEST_SEEDS), set(seeds.TRAIN_SEEDS)]
    assert sum(len(s) for s in lists) == len(set().union(*lists))


def test_test_seeds_are_untouched_and_only_the_final_study_may_open_them():
    ledger = (ROOT / "protocol" / "test_seed_ledger.jsonl").read_text().strip().splitlines()
    assert len(ledger) <= 1                                                           # used at most once
    for path in list((ROOT / "toyworld").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if path.name in ("seeds.py",) or path.name.startswith("final_"):
            continue
        assert not re.search(r"TEST_SEEDS|open_test_seeds", path.read_text()), f"{path.name} touches the test seeds"


def test_test_seeds_open_once_and_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(seeds, "TEST_SEED_LEDGER", tmp_path / "ledger.jsonl")
    assert seeds.open_test_seeds("final study") == seeds.TEST_SEEDS
    assert json.loads((tmp_path / "ledger.jsonl").read_text())["purpose"] == "final study"
    with pytest.raises(PermissionError):
        seeds.open_test_seeds("a second look")


# ---------------------------------------------------------------------- metrics
def test_ccum_alone_would_reward_dying_early_so_outcomes_come_with_it():
    dies = [row("failure", 2.98, 79) for _ in range(10)]
    survives = [row("success", 2.90, 120) for _ in range(9)] + [row("failure", 2.98, 79)]
    a, b = metrics.outcome_table(dies), metrics.outcome_table(survives)
    assert a["failure"] == 1.0 and np.isnan(a["ccum_success"])                          # no success to report a cost on
    assert b["success"] == 0.9 and b["failure"] == 0.1 and b["ccum_success"] == pytest.approx(2.90)
    assert list(b)[:5] == ["n", "success", "failure", "timeout", "retreat"]            # outcome rates before any ccum


def test_family_reading_of_the_regimes():
    assert [metrics.family(g) for g in ("R2d", "R2c", "R2l", "R3c", "R3l")] == ["lethal"] * 5
    assert [metrics.family(g) for g in ("R0", "R1", "R3d")] == ["damage_only"] * 3
    with pytest.raises(ValueError):
        metrics.family("R9")


def test_run_totals_recovery_and_repeated_failures():
    rows = [row("success", 1.5, episode=e) for e in range(5)] + [
        row("failure", 2.98, 80, episode=5), row("retreat", 0.8, 40, episode=6, revised=True),
        row("failure", 2.98, 80, episode=7), row("success", 1.0, 100, episode=8)]
    t = metrics.run_totals(rows)
    assert t["failures"] == 2 and t["retreats"] == 1 and t["revisions"] == 1
    assert t["ccum_run"] == pytest.approx(5 * 1.5 + 2.98 + 0.8 + 2.98 + 1.0) and t["steps_run"] == 5 * 100 + 80 + 40 + 80 + 100
    r = metrics.recovery(rows)
    assert r == {"episodes": 4, "steps": 300, "ccum": pytest.approx(2.98 + 0.8 + 2.98 + 1.0), "censored": False}
    assert metrics.recovery(rows[:7])["censored"] is True
    assert metrics.repeated_failures(rows) == 1                                         # D5: failures after the first
    assert metrics.repeated_failures(rows[:5]) == 0


def test_paired_bootstrap_and_holm():
    rng = np.random.default_rng(0)
    base = rng.normal(2.0, 0.3, 60)
    better = base - 0.4 + rng.normal(0, 0.1, 60)
    r = metrics.paired_bootstrap(base, better)
    assert r["reps"] == 5000 and r["n"] == 60 and r["lo"] < r["mean"] < r["hi"] and r["lo"] > 0 and r["p"] < 0.01
    assert metrics.paired_bootstrap(base, better) == r                                  # deterministic
    null = metrics.paired_bootstrap(base, base + rng.normal(0, 0.1, 60))
    assert null["lo"] < 0 < null["hi"] and null["p"] > 0.05
    assert metrics.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_tuning_score_counts_failures_after_drift_and_false_alarms_without_harm():
    rows = [row("failure", 3.0), row("success", 1.5), row("failure", 3.0, episode=2),          # pre-drift: ignored
            row("retreat", 0.5, regime="R0", tau=None, episode=3), row("success", 1.5, regime="R0", tau=None, episode=4)]
    assert metrics.tuning_score(rows) == pytest.approx(0.5 + 0.5)


# ---------------------------------------------------------------------- equal tuning budget
def test_tuning_budget_is_enforced_and_logged(tmp_path):
    led = tmp_path / "t.jsonl"
    calls = []
    cands = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    best, scores = tuning.tune("agent1", cands, lambda c, s, g, n: calls.append((s, g, n)) or abs(c - 0.3), ledger=led)
    assert best == 0.3 and len(scores) == 6 and len(calls) == tuning.BUDGET.candidates
    assert all(c == (tuning.BUDGET.seeds, tuning.BUDGET.regimes, tuning.BUDGET.n_episodes) for c in calls)
    tuning.tune("agent2", cands, lambda c, s, g, n: c, ledger=led)
    assert tuning.check_equal_budgets(led)["candidates"] == 6
    with pytest.raises(ValueError):
        tuning.tune("agent3", cands[:4], lambda c, s, g, n: c, ledger=led)                # a smaller grid is also refused
    with pytest.raises(ValueError):
        tuning.tune("agent3", cands, lambda c, s, g, n: c, tuning.TuningBudget(seeds=seeds.TEST_SEEDS[:10]), ledger=led)
    tuning.tune("agent2", cands, lambda c, s, g, n: c, ledger=led)                       # tuned a second time
    with pytest.raises(AssertionError):
        tuning.check_equal_budgets(led)


def test_unequal_budgets_in_the_ledger_are_detected(tmp_path):
    led = tmp_path / "t.jsonl"
    b1 = {"candidates": 6, "seeds": [1], "regimes": ["R0"], "n_episodes": 20}
    b2 = dict(b1, candidates=12)
    led.write_text("\n".join(json.dumps({"agent": a, "budget": b}) for a, b in (("a1", b1), ("a2", b2))) + "\n")
    with pytest.raises(AssertionError):
        tuning.check_equal_budgets(led)
    assert tuning.check_equal_budgets(tmp_path / "missing.jsonl") is None


def test_the_committed_tuning_ledger_gives_every_agent_the_same_budget():
    tuning.check_equal_budgets()
