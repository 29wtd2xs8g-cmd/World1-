"""Toy world: a point mass with inertia in the plane, hidden drift, graded irreversible damage.

Equations (world specification, section 2):
    x[t+1] = x[t] + dt * v[t+1]
    v[t+1] = v[t] + dt * ( a_max * u[t-d] - gamma * |v[t]| * v[t] + w[t](x[t]) ) + sqrt(dt) * sigma * eps[t]
    phi_k(x) = max(0, 1 - |x - z_k|^2 / rho_k^2)
    w[t](x)  = sum_k psi_k(t) * phi_k(x) * e_k
    c[t]     = sum_k theta_k(t) * phi_k(x[t])           (damage units / s)
    Ccum[t]  = sum_{u<=t} c[u] * dt ;  b[t] = D - Ccum[t]
    o[t]     = ( x[t] + eta, v[t] + eta', n[t] ),  n[t] = max(0, c[t] + eps_n)

The agent only sees `obs` (5 numbers). All ground truth is in `info["truth"]`.
The hidden parameters (theta, psi, d) are modified only by the wrappers in `drift.py`.

A "run" chains `n_episodes` episodes: each reset moves to the next episode, and
`reset(options={"new_run": True})` restarts at episode 0. Ccum restarts at 0 in every episode.
Noise draws depend only on (run seed, episode index): two agents facing the same run suffer exactly
the same noise (paired scenarios).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import WorldConfig


class ToyWorld(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: WorldConfig | None = None, n_episodes: int = 20, run_seed: int = 0):
        super().__init__()
        self.cfg = config or WorldConfig()
        c = self.cfg
        self.n_episodes = n_episodes
        self.run_seed = run_seed
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)

        self._Z = np.array(c.z, float)
        self._rho2 = np.array(c.rho, float) ** 2
        self._E = np.array(c.e, float)
        self._E = self._E / np.linalg.norm(self._E, axis=1, keepdims=True)
        self._x0 = np.array(c.x0, float)
        self._g = np.array(c.goal, float)

        self.episode = -1
        self.t = 0
        self.theta = np.array(c.theta0, float)
        self.psi = np.zeros(len(c.z))
        self.d = 0
        self.x = self._x0.copy()
        self.v = np.zeros(2)
        self.ccum = 0.0

    # ------------------------------------------------------------------ hidden parameters
    def set_hidden(self, theta=None, psi=None, d=None):
        """Reserved for the drift wrappers."""
        if theta is not None:
            self.theta = np.array(theta, float)
        if psi is not None:
            self.psi = np.array(psi, float)
        if d is not None:
            self.d = int(d)

    def _reset_hidden(self):
        self.theta = np.array(self.cfg.theta0, float)
        self.psi = np.zeros(len(self.cfg.z))
        self.d = 0

    # ------------------------------------------------------------------ geometry
    def phi(self, x):
        d2 = np.sum((self._Z - np.asarray(x, float)) ** 2, axis=1)
        return np.maximum(0.0, 1.0 - d2 / self._rho2)

    def damage_rate(self, x):
        return float(self.theta @ self.phi(x))

    def _collide(self, x_old, x_new, v_new):
        c = self.cfg
        x_new = x_new.copy()
        v_new = v_new.copy()
        # wall: it can only be crossed through an opening
        wx = c.wall_x
        if (x_old[0] - wx) * (x_new[0] - wx) < 0:
            s = (wx - x_old[0]) / (x_new[0] - x_old[0])
            y_cross = x_old[1] + s * (x_new[1] - x_old[1])
            if not any(abs(y_cross - yc) <= hw for yc, hw in c.openings):
                side = np.sign(x_old[0] - wx)
                x_new[0] = wx + side * 1e-6
                v_new[0] = 0.0
        # arena borders: position is clipped, normal velocity is cancelled
        for i in range(2):
            if x_new[i] < 0.0:
                x_new[i] = 0.0
                v_new[i] = max(v_new[i], 0.0)
            elif x_new[i] > c.arena:
                x_new[i] = c.arena
                v_new[i] = min(v_new[i], 0.0)
        return x_new, v_new

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.run_seed = int(seed)
        if options and options.get("new_run"):
            self.episode = -1
        self.episode += 1
        c = self.cfg
        ss = np.random.SeedSequence([self.run_seed, self.episode])
        rng = np.random.default_rng(ss)
        self._eps = rng.standard_normal((c.T, 2))            # process noise
        self._eta_x = rng.standard_normal((c.T + 1, 2))      # observation noise (position)
        self._eta_v = rng.standard_normal((c.T + 1, 2))      # observation noise (velocity)
        self._eps_n = rng.standard_normal(c.T + 1)           # nociceptive noise

        self._reset_hidden()
        self.x = self._x0.copy()
        self.v = np.zeros(2)
        self.t = 0
        self._u_hist = []
        self.ccum = 0.0
        self._c = self.damage_rate(self.x)
        self.ccum += self._c * c.dt
        obs = self._observe()
        return obs, {"truth": self._truth(outcome=None, resid=np.zeros(2))}

    def _observe(self):
        c = self.cfg
        n = max(0.0, self._c + c.sigma_n * self._eps_n[self.t])
        return np.concatenate([
            self.x + c.sigma_o * self._eta_x[self.t],
            self.v + c.sigma_o * self._eta_v[self.t],
            [n],
        ])

    def step(self, action):
        c = self.cfg
        t = self.t
        u = np.clip(np.asarray(action, float).reshape(2), -1.0, 1.0)
        self._u_hist.append(u)
        k = t - self.d
        u_eff = self._u_hist[k] if k >= 0 else np.zeros(2)

        w = (self.psi * self.phi(self.x)) @ self._E
        drag = c.gamma * np.linalg.norm(self.v) * self.v
        v_hat = self.v + c.dt * (c.a_max * u - drag)         # prediction of a frozen nominal model
        v_new = (self.v + c.dt * (c.a_max * u_eff - drag + w)
                 + np.sqrt(c.dt) * c.sigma * self._eps[t])
        scale = np.sqrt(c.dt) * c.sigma if c.sigma > 0 else 1.0
        resid = (v_new - v_hat) / scale                      # standardised residual (ground truth)
        x_new = self.x + c.dt * v_new
        x_new, v_new = self._collide(self.x, x_new, v_new)

        self.x, self.v = x_new, v_new
        self.t = t + 1
        self._c = self.damage_rate(self.x)
        self.ccum += self._c * c.dt

        failed = self.ccum >= c.D
        success = (not failed) and np.linalg.norm(self.x - self._g) <= c.r_goal
        terminated = bool(failed or success)
        truncated = bool((not terminated) and self.t >= c.T - 1)
        outcome = "failure" if failed else "success" if success else "timeout" if truncated else None

        obs = self._observe()
        # The specification defines no reward: the goal is g under a damage constraint.
        return obs, 0.0, terminated, truncated, {"truth": self._truth(outcome, resid)}

    def _truth(self, outcome, resid):
        c = self.cfg
        phi = self.phi(self.x)
        return {
            "episode": self.episode, "t": self.t,
            "x": self.x.copy(), "v": self.v.copy(),
            "c": self._c, "ccum": self.ccum, "b": c.D - self.ccum,
            "theta": self.theta.copy(), "psi": self.psi.copy(), "d": self.d,
            "phi": phi, "zone": int(np.argmax(phi)) if phi.max() > 0 else -1,
            "frozen_resid": np.asarray(resid).copy(),
            "outcome": outcome,
        }
