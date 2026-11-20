"""Basic world tests (checklist 1.1, 1.4, 1.5, 1.6): dynamics, geometry, damage, retreat, leaks."""
import numpy as np
import pytest

from monde import WorldConfig, make_env
from monde.policies import ScriptedCrosser, retreat_action
from monde.world import ToyWorld


def quiet(**kw):
    return WorldConfig(sigma=0.0, sigma_o=0.0, sigma_n=0.0, **kw)


# ---------------------------------------------------------------- 1.1 dynamics
def test_terminal_velocity_without_noise():
    """Constant thrust: terminal speed sqrt(a_max / gamma) ~ 1.41 m/s (away from zones and wall)."""
    cfg = quiet()
    env = ToyWorld(cfg)
    env.reset()
    env.x = np.array([1.0, 0.2])
    for _ in range(150):
        env.step([0.0, 1.0])
        if env.x[1] > 9.5:
            break
    assert abs(env.v[1] - np.sqrt(cfg.a_max / cfg.gamma)) < 0.03


def test_determinism_and_paired_scenarios():
    """Same seed => same noise draws, whatever the agent's actions."""
    def noise_trace(actions):
        env = make_env("R0", seed=7)
        env.reset(seed=7, options={"new_run": True})
        return [env.unwrapped._eps[i].copy() for i in range(5)], [env.step(a)[0] for a in actions]

    e1, o1 = noise_trace([[1, 0]] * 5)
    e2, o2 = noise_trace([[0, 1]] * 5)
    assert all(np.array_equal(a, b) for a, b in zip(e1, e2))      # same noise draws
    e3, o3 = noise_trace([[1, 0]] * 5)
    assert all(np.array_equal(a, b) for a, b in zip(o1, o3))      # same agent => same observations


def test_episodes_differ_within_a_run():
    env = make_env("R0", seed=3)
    env.reset(seed=3, options={"new_run": True})
    a = env.unwrapped._eps.copy()
    env.reset()
    assert env.unwrapped.episode == 1 and not np.array_equal(a, env.unwrapped._eps)


# ---------------------------------------------------------------- 1.1 geometry
def test_wall_blocks_outside_openings_and_lets_through_opening_A():
    cfg = quiet()
    env = ToyWorld(cfg)
    env.reset()
    env.x, env.v = np.array([4.9, 3.0]), np.array([1.0, 0.0])    # y = 3: solid wall
    for _ in range(20):
        env.step([1.0, 0.0])
    assert env.x[0] < cfg.wall_x
    env.reset()
    env.x, env.v = np.array([4.9, 5.0]), np.array([1.0, 0.0])    # y = 5: opening A
    for _ in range(20):
        env.step([1.0, 0.0])
    assert env.x[0] > cfg.wall_x


def test_arena_borders():
    env = ToyWorld(quiet())
    env.reset()
    env.x, env.v = np.array([0.2, 0.2]), np.array([-1.0, -1.0])
    for _ in range(30):
        env.step([-1.0, -1.0])
    assert (env.x >= 0).all() and (env.v <= 1e-9).all()


def test_start_is_outside_every_zone():
    cfg = WorldConfig()
    env = ToyWorld(cfg)
    env.reset()
    for zk, rk in zip(cfg.z, cfg.rho):
        assert np.linalg.norm(np.array(zk) - np.array(cfg.x0)) >= rk + cfg.r0


# ---------------------------------------------------------------- 1.4 graded damage and failure at D
def test_profile_and_damage_rate():
    cfg = quiet()
    env = ToyWorld(cfg)
    env.reset()
    assert env.damage_rate([5.0, 5.0]) == pytest.approx(cfg.theta0[0])               # centre of Z1: theta1 * 1
    assert env.damage_rate([5.0, 5.5]) == pytest.approx(cfg.theta0[0] * 0.75)        # phi = 1 - 0.25
    assert env.damage_rate([5.0, 6.0]) == 0.0 and env.damage_rate([5.0, 7.0]) == 0.0  # edge and outside
    assert env.damage_rate([5.0, 1.5]) == pytest.approx(cfg.theta0[1])
    assert env.damage_rate([3.0, 3.0]) == 0.0                                        # phantom zone: theta3 = 0


def test_cumulation_margin_and_irreversible_failure():
    cfg = quiet(D=0.5)
    env = ToyWorld(cfg)
    env.reset()
    env.x, env.v = np.array([4.5, 5.0]), np.zeros(2)
    last = None
    for _ in range(300):
        _, _, te, tr, info = env.step([0.0, 0.0])        # stays inside Z1 (phi = 0.75 at x = 4.5)
        last = info["truth"]
        if te or tr:
            break
    assert last["outcome"] == "failure" and last["ccum"] >= cfg.D
    assert last["b"] == pytest.approx(cfg.D - last["ccum"])


def test_success_requires_ccum_below_D():
    env = ToyWorld(quiet())
    env.reset()
    env.x = np.array([8.8, 8.8])
    _, _, te, _, info = env.step([0.0, 0.0])
    assert te and info["truth"]["outcome"] == "success"


def test_timeout_at_T_minus_1():
    env = ToyWorld(quiet())
    env.reset()
    for _ in range(env.cfg.T):
        _, _, te, tr, info = env.step([0.0, 0.0])
        if tr:
            break
    assert info["truth"]["outcome"] == "timeout" and info["truth"]["t"] == env.cfg.T - 1


# ---------------------------------------------------------------- 1.5 damage-free retreat
@pytest.mark.parametrize("start", [(1.8, 1.0), (1.0, 2.2), (2.2, 1.6), (3.7, 1.2)])
def test_retreat_from_outside_a_zone_causes_no_damage(start):
    env = make_env("R0", seed=0)
    env.reset(seed=0, options={"new_run": True})
    w = env.unwrapped
    w.x, w.v = np.array(start, float), np.array([0.5, 0.3])
    c = w.cfg
    for _ in range(200):
        _, _, te, tr, info = env.step(retreat_action(w.x, w.v, c.x0))
        assert info["truth"]["c"] == 0.0
        if te or tr:
            break
    assert np.linalg.norm(w.x - np.array(c.x0)) < 0.2 and np.linalg.norm(w.v) < 0.1


def test_retreat_from_inside_a_zone_counts_the_damage():
    env = make_env("R0", seed=0)
    env.reset(seed=0, options={"new_run": True})
    w = env.unwrapped
    w.x, w.v = np.array([4.6, 5.0]), np.zeros(2)
    ccum0 = w.ccum
    for _ in range(40):
        env.step(retreat_action(w.x, w.v, w.cfg.x0))
    assert w.ccum > ccum0


# ---------------------------------------------------------------- 1.6 / ground-truth leaks
def test_observation_has_five_numbers_and_no_ground_truth():
    env = make_env("R2c", seed=0)
    obs, info = env.reset(seed=0, options={"new_run": True})
    assert obs.shape == (5,) and env.observation_space.shape == (5,)
    assert "truth" in info and "drift" in info        # ground truth lives in info, never in obs


def test_hidden_parameters_do_not_leak_into_the_observation():
    """Outside every zone and motionless: changing theta, psi, d does not change the observation."""
    def obs_seq(theta, psi, d):
        env = ToyWorld(WorldConfig(), run_seed=5)
        env.reset()
        env.set_hidden(theta=theta, psi=psi, d=d)
        return np.array([env.step([0.0, 0.0])[0] for _ in range(10)])

    a = obs_seq((1.0, 0.3, 0.0), (0, 0, 0), 0)
    b = obs_seq((9.0, 7.0, 5.0), (3, 3, 3), 4)
    assert np.array_equal(a, b)


def test_ground_truth_journal(tmp_path):
    from monde.journal import TruthLogger
    env = TruthLogger(make_env("R2c", seed=2, n_episodes=2), regime="R2c")
    pol = ScriptedCrosser("A")
    for ep in range(2):
        obs, _ = env.reset(seed=2 if ep == 0 else None, options={"new_run": True} if ep == 0 else None)
        pol.reset()
        done = False
        while not done:
            obs, _, te, tr, _ = env.step(pol.act(obs))
            done = te or tr
    env.save(tmp_path)
    lines = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and '"nature": "cost"' in lines[0] and '"k_star": 0' in lines[0]
    assert (tmp_path / "trajectories.npz").exists()
