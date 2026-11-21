"""Baseline agent (checklist 2.1): route choice + CEM model-predictive control. See docs/agents.md.

It plans with a world model (`WorldModel` or `AnalyticModel`) that it never updates. It only reads
observations (x, y, vx, vy, n); it knows D, g, the openings and its own commands.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .config import WorldConfig
from .model import K_DEFAULT

torch.set_num_threads(1)


@dataclass(frozen=True)
class AgentParams:
    lam: float = 0.1            # weight of predicted damage against time
    v_cruise: float = 0.9       # m/s, speed used to turn lengths into times
    horizon: int = 20           # planning steps
    samples: int = 256
    elites: int = 25
    iters: int = 4
    reach: float = 0.3          # m, waypoint switching radius
    w_pos: float = 1.0          # squared distance to the current waypoint
    w_cross: float = 3.0        # squared cross-track error while crossing an opening
    w_vel: float = 0.3          # squared speed, only close to the gate waypoints
    w_u: float = 0.01           # command effort
    K: int = K_DEFAULT
    seed: int = 0


class BaselineAgent:
    def __init__(self, model, config: WorldConfig | None = None, params: AgentParams | None = None):
        self.m = model
        self.cfg = config or WorldConfig()
        self.p = params or AgentParams()
        self.gen = torch.Generator().manual_seed(self.p.seed)
        self.routes = self._build_routes()
        self.route = None
        self.reset()

    # ------------------------------------------------------------------ routes
    def _build_routes(self):
        c = self.cfg
        g = np.array(c.goal)
        out = {}
        for name, (yo, hw) in zip("AB", c.openings):
            out[name] = np.array([(c.wall_x - 0.8, yo), (c.wall_x + 0.8, yo), g])
        return out

    def _polyline(self, name):
        return np.vstack([np.array(self.cfg.x0), self.routes[name]])

    def route_costs(self, b_hat):
        """Returns {route: (J, length, predicted damage, allowed)}; J = L / v + lam * damage."""
        p = self.p
        with torch.no_grad():
            floor = float(self.m.predict_n(torch.tensor([self.cfg.x0], dtype=torch.float32))[0])
            out = {}
            for name in self.routes:
                pts = self._polyline(name)
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                L = float(seg.sum())
                dense = np.vstack([pts[i] + (pts[i + 1] - pts[i]) * np.linspace(0, 1, max(2, int(s / 0.05)), endpoint=False)[:, None]
                                   for i, s in enumerate(seg)])
                n_hat = self.m.predict_n(torch.tensor(dense, dtype=torch.float32)).numpy()
                dmg = float(np.maximum(0.0, n_hat - floor).sum() * 0.05 / p.v_cruise)
                out[name] = (L / p.v_cruise + p.lam * dmg, L, dmg, dmg < b_hat)
        return out

    def choose_route(self, b_hat):
        """Cheapest route whose predicted damage fits in the estimated margin; if none fits, the least damaging."""
        costs = self.route_costs(b_hat)
        allowed = {k: v for k, v in costs.items() if v[3]}
        if allowed:
            return min(allowed, key=lambda k: allowed[k][0])
        return min(costs, key=lambda k: costs[k][2])

    # ------------------------------------------------------------------ episode
    def reset(self):
        self.hist = torch.zeros(self.p.K - 1, 2)         # last applied commands, newest first
        self.n_sum = 0.0
        self.wp_i = 0
        self._mean = None
        self.route = self.choose_route(self.cfg.D)
        self.wps = self.routes[self.route]
        self.gen.manual_seed(self.p.seed + 1000003 * getattr(self, "_episodes", 0))
        self._episodes = getattr(self, "_episodes", 0) + 1

    @property
    def b_hat(self):
        return self.cfg.D - self.n_sum * self.cfg.dt

    # ------------------------------------------------------------------ control
    def act(self, obs):
        p, c = self.p, self.cfg
        x = np.asarray(obs[0:2], float)
        v = np.asarray(obs[2:4], float)
        self.n_sum += float(obs[4])
        while self.wp_i < len(self.wps) - 1 and np.linalg.norm(self.wps[self.wp_i] - x) < p.reach:
            self.wp_i += 1
        u = self._plan(x, v)
        self.hist = torch.cat([torch.as_tensor(u, dtype=torch.float32)[None], self.hist[:-1]])
        return u

    @torch.no_grad()
    def _plan(self, x, v):
        p, c = self.p, self.cfg
        H, N = p.horizon, p.samples
        wp = torch.tensor(self.wps[self.wp_i], dtype=torch.float32)
        prev = torch.tensor(self.wps[self.wp_i - 1] if self.wp_i > 0 else np.array(c.x0), dtype=torch.float32)
        gate_leg = self.wp_i == 1                          # crossing the opening: stay on the line
        near_gate = self.wp_i == 0
        x0 = torch.tensor(x, dtype=torch.float32).expand(N, 2)
        v0 = torch.tensor(v, dtype=torch.float32).expand(N, 2)
        mean = self._mean if self._mean is not None else torch.zeros(H, 2)
        std = torch.full((H, 2), 0.6)
        line = wp - prev
        line_n = line / line.norm().clamp_min(1e-6)
        normal = torch.stack([-line_n[1], line_n[0]])
        for _ in range(p.iters):
            U = (mean + std * torch.randn(N, H, 2, generator=self.gen)).clamp(-1.0, 1.0)
            cost = self._rollout_cost(x0, v0, U, wp, prev, normal, gate_leg, near_gate)
            elite = U[cost.topk(p.elites, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0).clamp_min(0.05)
        self._mean = torch.cat([mean[1:], torch.zeros(1, 2)])
        return mean[0].clamp(-1.0, 1.0).numpy().astype(float)

    def _rollout_cost(self, X, V, U, wp, prev, normal, gate_leg, near_gate):
        p, c = self.p, self.cfg
        N, H, _ = U.shape
        hist = self.hist[None].expand(N, -1, -1)
        cost = torch.zeros(N)
        for h in range(H):
            uh = U[:, h]
            uhist = torch.cat([uh[:, None], hist], dim=1)
            V = V + self.m.predict_dv(X, V, uhist)
            X = X + c.dt * V
            hist = uhist[:, :-1]
            d = X - wp
            cost = cost + p.w_pos * (d ** 2).sum(1) + p.lam * self.m.predict_n(X) * c.dt + p.w_u * (uh ** 2).sum(1)
            if gate_leg:
                cost = cost + p.w_cross * (((X - prev) @ normal) ** 2)
            if near_gate:
                close = (d.norm(dim=1) < 1.5).float()
                cost = cost + p.w_vel * close * (V ** 2).sum(1)
        return cost

