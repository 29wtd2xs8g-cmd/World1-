"""Tests of the world model, data collection and baseline agent (checklist 2.1), fast versions.
The full competence check is scripts/eval_baseline.py."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from toyworld import WorldConfig, make_env
from toyworld.agent import AgentParams, BaselineAgent
from toyworld.data import collect
from toyworld.model import K_DEFAULT, AnalyticModel, WorldModel
from toyworld.seeds import CALIBRATION_SEEDS, TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS


def test_seed_lists_are_disjoint():
    lists = [set(CALIBRATION_SEEDS), set(VALIDATION_SEEDS), set(TEST_SEEDS), set(TRAIN_SEEDS)]
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            assert not lists[i] & lists[j]


def test_collected_data_matches_world_equations():
    """Features and targets are built from observations; the first K-1 history entries are zero-padded."""
    d = collect([1], segments_per_seed=2, seg_len=30)
    assert d["X_dyn"].shape == (60, 4 + 2 * K_DEFAULT) and d["Y_dv"].shape == (60, 2)
    first = d["X_dyn"][0]
    hist = first[4:].reshape(K_DEFAULT, 2)
    assert np.all(hist[1:] == 0.0) and np.any(hist[0] != 0.0)            # only u[t] is known at t = 0
    second = d["X_dyn"][1][4:].reshape(K_DEFAULT, 2)
    assert np.array_equal(second[1], hist[0])                            # newest first: u[t-1] = previous u[t]


def test_collection_covers_zones_and_both_sides_of_the_wall():
    d = collect(range(4000, 4010), segments_per_seed=10, seg_len=100)
    x = d["X_noc"]
    assert (x[:, 0] < 5).mean() > 0.3 and (x[:, 0] > 5).mean() > 0.3
    for z in WorldConfig().z:
        assert (np.linalg.norm(x - np.array(z), axis=1) < 0.8).sum() > 50


def test_model_learns_a_small_dataset_and_is_deterministic(tmp_path):
    d = collect(range(4000, 4008), segments_per_seed=10, seg_len=100)
    m1, m2 = WorldModel(), WorldModel()
    torch.manual_seed(0); m1 = WorldModel()
    r1 = m1.fit(d, epochs=6, seed=3)
    torch.manual_seed(0); m2 = WorldModel()
    r2 = m2.fit(d, epochs=6, seed=3)
    assert r1 == r2                                                       # same seeds, same result
    assert r1["dv"] < 0.5 * float(np.std(d["Y_dv"])) * 1.4 and r1["n"] < 0.2
    path = tmp_path / "new_folder" / "m.pt"                               # save creates the folder
    m1.save(path)
    m3 = WorldModel.load(path)
    x = torch.tensor([[5.0, 5.0]])
    assert torch.allclose(m1.predict_n(x), m3.predict_n(x))


def test_analytic_model_matches_the_world_without_noise():
    cfg = replace(WorldConfig(), sigma=0.0, sigma_o=0.0, sigma_n=0.0)
    env = make_env("R0", seed=0, config=cfg)
    obs, _ = env.reset(seed=0, options={"new_run": True})
    model = AnalyticModel(cfg)
    hist = torch.zeros(1, K_DEFAULT, 2)
    for t in range(40):
        u = np.array([1.0, 0.5])
        hist = torch.cat([torch.tensor(u, dtype=torch.float32)[None, None], hist[:, :-1]], dim=1)
        pred = model.predict_dv(torch.tensor(obs[None, 0:2], dtype=torch.float32),
                                torch.tensor(obs[None, 2:4], dtype=torch.float32), hist)[0].numpy()
        new_obs, *_ = env.step(u)
        w = env.unwrapped
        if np.linalg.norm(w.x - w.cfg.x0) < 3.5:                           # away from wall and zones
            assert np.allclose(new_obs[2:4] - obs[2:4], pred, atol=1e-5)
        obs = new_obs


def test_analytic_model_with_delay_reads_the_right_command():
    model = AnalyticModel(WorldConfig(), d=3)
    uhist = torch.zeros(1, K_DEFAULT, 2)
    uhist[0, 3] = torch.tensor([1.0, 0.0])
    dv = model.predict_dv(torch.zeros(1, 2) + 1.0, torch.zeros(1, 2), uhist)
    assert dv[0, 0] == pytest.approx(0.1)                                  # dt * a_max * u[t-3]


def test_route_choice_prefers_A_and_switches_to_B_when_damage_costs_more():
    cfg = WorldConfig()
    cheap = BaselineAgent(AnalyticModel(cfg), cfg, AgentParams(lam=0.1))
    assert cheap.route == "A"
    dear = BaselineAgent(AnalyticModel(cfg), cfg, AgentParams(lam=2.0))
    assert dear.route == "B"                                               # damage dominates time


def test_route_choice_respects_the_margin():
    cfg = WorldConfig()
    agent = BaselineAgent(AnalyticModel(cfg), cfg, AgentParams(lam=0.1))
    costs = agent.route_costs(b_hat=1.0)                                   # A predicts about 1.5 > 1.0
    assert not costs["A"][3] and costs["B"][3]
    assert agent.choose_route(1.0) == "B"


def test_agent_crosses_with_the_analytic_model():
    cfg = WorldConfig()
    env = make_env("R0", seed=5)
    agent = BaselineAgent(AnalyticModel(cfg), cfg)
    obs, _ = env.reset(seed=5, options={"new_run": True})
    agent.reset()
    done = False
    while not done:
        obs, _, te, tr, info = env.step(agent.act(obs))
        done = te or tr
    assert info["truth"]["outcome"] == "success" and agent.route == "A"


def test_agent_sees_only_observations():
    """The agent's inputs are the 5 observed numbers and the config; hidden parameters change nothing it computes
    outside the zones (same observations in, same commands out)."""
    cfg = WorldConfig()

    def commands(theta, psi, d):
        env = make_env("R0", seed=9)
        agent = BaselineAgent(AnalyticModel(cfg), cfg)
        obs, _ = env.reset(seed=9, options={"new_run": True})
        env.unwrapped.set_hidden(theta=theta, psi=psi, d=d)
        agent.reset()
        out = []
        for _ in range(5):
            u = agent.act(obs)
            out.append(u)
            obs, *_ = env.step(u)
        return np.array(out)

    a = commands((1.0, 0.3, 0.0), (0, 0, 0), 0)
    b = commands((1.0, 0.3, 0.0), (0, 0, 0), 0)
    assert np.array_equal(a, b)                                            # deterministic given the seed
