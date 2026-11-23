"""Secondary analysis (decision D11): non-completion = failure + retreat + timeout. Synthetic data, fast."""
import numpy as np
import pytest

from toyworld.compare import AGENTS, REGIMES
from toyworld.secondary import (HOLM_FAMILY, SECONDARY_REGIMES, complete_run_seeds, non_completion, reading,
                                secondary_contrast, secondary_table)


def rows(regime, outcomes, tau=4):
    return [{"regime": regime, "seed": 0, "episode": e, "tau_ep": None if regime == "R0" else tau, "ccum": 1.0,
             "outcome": o, "steps": 100, "alert_step": None, "revised": False} for e, o in enumerate(outcomes)]


def make_runs(n_seeds=10, a_post=("success",) * 5 + ("failure",), b_post=("retreat",) * 5 + ("failure",)):
    """Agent 1 completes the task, Agent 2 retreats; the first four episodes are before the onset."""
    runs = {}
    for s in range(n_seeds):
        for g in REGIMES:
            for kind, post in zip(AGENTS, (a_post, b_post)):
                runs[(kind, g, s)] = rows(g, ("success",) * 4 + post)
    return runs


def test_non_completion_is_failure_plus_retreat_plus_timeout():
    runs = make_runs(b_post=("retreat", "retreat", "timeout", "failure", "success", "success"))
    c = secondary_contrast(runs, "R2l", list(range(10)), reps=200)
    assert c["noncompletion_b"] == pytest.approx(c["failure_b"] + c["retreat_b"] + c["timeout_b"])
    assert c["noncompletion_b"] == pytest.approx(4 / 6)
    assert c["noncompletion_b"] == pytest.approx(1.0 - c["success_b"])
    assert np.allclose(non_completion(runs, AGENTS[1], "R2l", list(range(10))), 4 / 6)


def test_the_registered_failure_rate_is_untouched():
    c = secondary_contrast(make_runs(), "R2l", list(range(10)), reps=200)
    assert c["failure_a"] == pytest.approx(1 / 6) and c["failure_b"] == pytest.approx(1 / 6)


def test_agent_that_retreats_completes_less_often():
    c = secondary_contrast(make_runs(), "R2l", list(range(10)), reps=200)
    assert c["noncompletion_gap"]["mean"] > 0.5            # Agent 2 minus Agent 1: positive = Agent 1 completes more often


def test_holm_covers_the_five_lethal_regimes_and_not_r3d():
    table = secondary_table(make_runs(), list(range(10)), reps=200)
    assert [c["regime"] for c in table] == list(SECONDARY_REGIMES)
    assert {c["regime"] for c in table if c["holm_p"] is not None} == set(HOLM_FAMILY)
    assert next(c for c in table if c["regime"] == "R3d")["holm_p"] is None
    assert reading(next(c for c in table if c["regime"] == "R3d")) == "not judged"


def test_reading_follows_the_sign_and_the_adjusted_p():
    table = secondary_table(make_runs(), list(range(10)), reps=200)
    assert reading(next(c for c in table if c["regime"] == "R2l")) == "Agent 1 completes more"
    same = secondary_table(make_runs(b_post=("success",) * 5 + ("failure",)), list(range(10)), reps=200)
    assert reading(next(c for c in same if c["regime"] == "R2l")) == "no difference"


def test_nothing_is_read_before_the_required_seeds_are_complete():
    runs = make_runs(n_seeds=10)
    assert complete_run_seeds(runs, 10) == list(range(10))
    with pytest.raises(ValueError, match="before the study is complete"):
        complete_run_seeds(runs, 60)
    del runs[(AGENTS[0], "R3l", 3)]                         # one missing run: the seed is incomplete
    with pytest.raises(ValueError):
        complete_run_seeds(runs, 10)
