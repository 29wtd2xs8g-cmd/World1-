"""Trigger interface, revision and adaptive runner (checklist 2.2 and 2.4). Fast versions."""
import numpy as np
import pytest
import torch

from toyworld import WorldConfig, make_env
from toyworld.adaptive import _sync_oracle, run_agent
from toyworld.agent import BaselineAgent
from toyworld.data import collect
from toyworld.evaluate import run_frozen
from toyworld.model import K_DEFAULT, AnalyticModel, WorldModel
from toyworld.revision import RevisionParams, window_data
from toyworld.triggers import STEP_RECORD_FIELDS, AlwaysTrigger, NeverTrigger, StepRecord, Trigger


@pytest.fixture(scope="module")
def small_model():
    d = collect(range(4000, 4008), segments_per_seed=10, seg_len=100)
    torch.manual_seed(0)
    m = WorldModel()
    m.fit(d, epochs=6, seed=3)
    return m.eval()


class Alert(Trigger):
    """Test trigger: alerts when the observed position satisfies a condition, in the listed episodes only."""
    name = "test"

    def __init__(self, cond, episodes=(0,), revise=False):
        self.cond, self.episodes, self.revise, self.ep = cond, set(episodes), revise, -1

    def reset_episode(self):
        self.ep += 1

    def step(self, rec):
        return self.ep in self.episodes and self.cond(rec.obs)

    def revise_at_end(self, alerted):
        return self.revise and alerted


# ---------------------------------------------------------------------- interface
def test_step_record_exposes_only_observable_quantities():
    assert set(STEP_RECORD_FIELDS) == {"t", "obs", "n_pred", "dv_resid", "b_hat"}
    hidden = {"theta", "psi", "d", "tau", "k_star", "regime", "seed", "truth", "ccum", "zone"}
    assert not hidden & set(STEP_RECORD_FIELDS)


def test_limit_triggers():
    rec = StepRecord(t=3, obs=np.zeros(5), n_pred=0.0, dv_resid=np.zeros(2), b_hat=1.0)
    for trig, revises in ((NeverTrigger(), False), (AlwaysTrigger(), True)):
        trig.reset_run()
        trig.reset_episode()
        assert trig.step(rec) is False                                   # neither raises an in-episode alert
        assert trig.revise_at_end(False) is revises and trig.revise_at_end(True) is revises
    assert Trigger().revise_at_end(True) is True and Trigger().revise_at_end(False) is False   # default: if alerted


# ---------------------------------------------------------------------- revision
def test_window_data_follows_the_collect_format():
    rng = np.random.default_rng(0)
    T = 30
    O, U = rng.normal(size=(T + 1, 5)), rng.uniform(-1, 1, size=(T, 2))
    d = window_data(O, U)
    assert d["X_dyn"].shape == (T, 4 + 2 * K_DEFAULT) and d["Y_dv"].shape == (T, 2)
    assert np.array_equal(d["X_dyn"][5, :4], O[5, :4])
    hist = d["X_dyn"][5, 4:].reshape(K_DEFAULT, 2)
    assert np.array_equal(hist[0], U[5]) and np.array_equal(hist[1], U[4])           # newest first
    assert np.all(d["X_dyn"][0, 4:].reshape(K_DEFAULT, 2)[1:] == 0.0)                # zero padding before the start
    assert np.allclose(d["Y_dv"], O[1:, 2:4] - O[:-1, 2:4])
    assert np.array_equal(d["X_noc"], O[:-1, 0:2]) and np.array_equal(d["Y_n"], O[:-1, 4])


def test_finetune_lowers_the_window_loss_keeps_normalisation_and_is_deterministic(small_model):
    d = collect([4500], segments_per_seed=3, seg_len=100)
    shifted = dict(d, Y_dv=d["Y_dv"] + np.array([0.05, -0.05]), Y_n=d["Y_n"] + 0.3)    # a changed world
    import copy
    a, b = copy.deepcopy(small_model), copy.deepcopy(small_model)
    buffers = {k: v.clone() for k, v in small_model.named_buffers()}
    ra = a.finetune(shifted, steps=40, seed=7)
    rb = b.finetune(shifted, steps=40, seed=7)
    assert ra["after"] < ra["before"]
    assert ra == rb
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb)
    for k, v in a.named_buffers():
        assert torch.equal(v, buffers[k])                                              # standardisation unchanged
    assert not all(torch.equal(p, q) for p, q in zip(a.parameters(), small_model.parameters()))
    assert len(list(small_model.parameters())) > 0 and small_model is not a


# ---------------------------------------------------------------------- runner
def test_never_trigger_runner_reproduces_the_frozen_evaluation():
    cfg = WorldConfig()
    ref = run_frozen("R0", seeds=[3], n_episodes=2, policy_factory=lambda: BaselineAgent(AnalyticModel(cfg)))
    out = run_agent("R0", 3, AnalyticModel(cfg), NeverTrigger(), n_episodes=2)
    assert [r["ccum"] for r in out] == [r["ccum"] for r in ref]
    assert [r["steps"] for r in out] == [r["steps"] for r in ref]
    assert all(not r["revised"] and r["alert_step"] is None for r in out)


def test_alert_on_the_start_side_triggers_a_retreat_that_closes_the_episode():
    cfg = WorldConfig()
    trig = Alert(lambda o: o[0] > 3.5, episodes=(0,))
    out = run_agent("R0", 4, AnalyticModel(cfg), trig, n_episodes=2)
    first, second = out
    assert first["outcome"] == "retreat" and first["alert_step"] is not None
    assert first["ccum"] < cfg.D and first["steps"] > first["alert_step"]            # it kept moving back to the start
    assert second["outcome"] == "success" and second["alert_step"] is None           # the retry is a normal episode


def test_alert_past_the_wall_is_recorded_without_retreat():
    cfg = WorldConfig()
    trig = Alert(lambda o: o[0] > 6.0, episodes=(0,))
    out = run_agent("R0", 4, AnalyticModel(cfg), trig, n_episodes=1)
    assert out[0]["alert_step"] is not None and out[0]["outcome"] == "success"


def test_revision_runs_at_the_end_of_an_alerted_episode(small_model):
    before = {k: v.clone() for k, v in small_model.state_dict().items()}
    trig = Alert(lambda o: o[0] > 3.5, episodes=(0,), revise=True)
    out = run_agent("R0", 4, small_model, trig, n_episodes=2, revision=RevisionParams(steps=20))
    assert out[0]["revised"] and not out[1]["revised"]
    assert out[0]["loss_after"] < out[0]["loss_before"]
    for k, v in small_model.state_dict().items():
        assert torch.equal(v, before[k])                                              # each run revises its own copy


def test_always_trigger_revises_every_episode(small_model):
    out = run_agent("R0", 5, small_model, AlwaysTrigger(), n_episodes=2, revision=RevisionParams(steps=5))
    assert all(r["revised"] and r["alert_step"] is None for r in out)


# ---------------------------------------------------------------------- oracle
def test_oracle_reads_the_true_parameters_and_avoids_a_route_that_became_too_costly():
    cfg = WorldConfig()
    env = make_env("R2c", seed=1, onset_ep=0)                                         # theta_1: 1 -> 3 from episode 0
    env.reset(seed=1, options={"new_run": True})
    m = AnalyticModel(cfg)
    agent = BaselineAgent(m, cfg)
    assert agent.route == "A"                                                         # nominal parameters
    _sync_oracle(m, env.unwrapped)
    assert float(m.theta[0]) == pytest.approx(3.0)
    agent.reset()
    assert agent.route == "B"                                                         # A would predict 3 x 1.5 > D


def test_margin_frac_tightens_the_route_constraint_and_defaults_to_0_7():
    from toyworld.agent import AgentParams
    cfg = WorldConfig()
    assert AgentParams().margin_frac == 0.7                                             # DEV-1: the default for all agents
    whole = BaselineAgent(AnalyticModel(cfg), cfg, AgentParams(margin_frac=1.0))        # the first protocol
    buffered = BaselineAgent(AnalyticModel(cfg), cfg, AgentParams(margin_frac=0.6))
    assert whole.route_costs(2.0)["A"][3]                                               # A predicts about 1.5 < 2.0
    assert not buffered.route_costs(2.0)["A"][3]                                        # 1.5 >= 0.6 * 2.0
    assert buffered.choose_route(2.0) == "B"
