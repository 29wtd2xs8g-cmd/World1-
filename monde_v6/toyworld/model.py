"""World model: one model, two MLP heads (PyTorch). See docs/agents.md, section 3.

dynamics head:    (x, v, u[t], ..., u[t-K+1])  ->  v[t+1] - v[t]
nociception head: x                            ->  n[t]

Everything is expressed on observed (noisy) quantities. Command history is ordered newest first.
`AnalyticModel` has the same interface and uses the true equations: it is the oracle of checklist 2.2
and a reference for testing the planner without learning.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import WorldConfig

K_DEFAULT = 12


def _mlp(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(),
                         nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                         nn.Linear(n_hidden, n_out))


class WorldModel(nn.Module):
    def __init__(self, K=K_DEFAULT, h_dyn=128, h_noc=64):
        super().__init__()
        self.K, self.h_dyn, self.h_noc = K, h_dyn, h_noc
        self.dyn = _mlp(4 + 2 * K, h_dyn, 2)
        self.noc = _mlp(2, h_noc, 1)
        for name, n in (("dyn_in", 4 + 2 * K), ("noc_in", 2)):
            self.register_buffer(f"{name}_m", torch.zeros(n))
            self.register_buffer(f"{name}_s", torch.ones(n))
        for name, n in (("dyn_out", 2), ("noc_out", 1)):
            self.register_buffer(f"{name}_m", torch.zeros(n))
            self.register_buffer(f"{name}_s", torch.ones(n))

    # ------------------------------------------------------------------ prediction
    def predict_dv(self, x, v, uhist):
        """x, v: (N, 2); uhist: (N, K, 2), newest first. Returns v[t+1] - v[t], shape (N, 2)."""
        z = torch.cat([x, v, uhist.reshape(uhist.shape[0], -1)], dim=1)
        return self.dyn((z - self.dyn_in_m) / self.dyn_in_s) * self.dyn_out_s + self.dyn_out_m

    def predict_n(self, x):
        """x: (N, 2). Returns the expected nociceptive signal, shape (N,)."""
        out = self.noc((x - self.noc_in_m) / self.noc_in_s) * self.noc_out_s + self.noc_out_m
        return out[:, 0]

    # ------------------------------------------------------------------ training
    def fit(self, data, epochs=30, batch=512, lr=1e-3, seed=0, holdout=0.1, verbose=False):
        """data: dict of numpy arrays X_dyn (M, 4+2K), Y_dv (M, 2), X_noc (M, 2), Y_n (M,).
        Returns the holdout RMSE of each head."""
        g = torch.Generator().manual_seed(seed)
        t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        M = t["X_dyn"].shape[0]
        perm = torch.randperm(M, generator=g)
        n_val = int(holdout * M)
        va, tr = perm[:n_val], perm[n_val:]
        with torch.no_grad():
            for key, X, Y in (("dyn", t["X_dyn"][tr], t["Y_dv"][tr]), ("noc", t["X_noc"][tr], t["Y_n"][tr][:, None])):
                getattr(self, f"{key}_in_m").copy_(X.mean(0))
                getattr(self, f"{key}_in_s").copy_(X.std(0).clamp_min(1e-3))
                getattr(self, f"{key}_out_m").copy_(Y.mean(0))
                getattr(self, f"{key}_out_s").copy_(Y.std(0).clamp_min(1e-3))
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for ep in range(epochs):
            order = tr[torch.randperm(len(tr), generator=g)]
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                zd = (self._dyn_std_out(t["Y_dv"][idx]) - self._raw_dyn(t["X_dyn"][idx])) ** 2
                zn = (self._noc_std_out(t["Y_n"][idx][:, None]) - self._raw_noc(t["X_noc"][idx])) ** 2
                loss = zd.mean() + zn.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            if verbose and (ep % 5 == 4 or ep == epochs - 1):
                print(f"epoch {ep + 1}: holdout RMSE {self.holdout_rmse(t, va)}")
        return self.holdout_rmse(t, va)

    # ------------------------------------------------------------------ revision
    def finetune(self, data, steps=50, batch=128, lr=1e-3, seed=0):
        """Revision of a pretrained model (docs/agents.md, section 9): `steps` Adam steps on minibatches drawn with
        replacement from `data` (the window of the last episode, no replay). The standardisation buffers are NOT
        re-estimated: they stay those of the pretraining. Returns the mean standardised loss before and after,
        measured on the whole window."""
        g = torch.Generator().manual_seed(seed)
        t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        M = t["X_dyn"].shape[0]
        everything = torch.arange(M)
        before = self._loss(t, everything).item()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            idx = torch.randint(0, M, (min(batch, M),), generator=g)
            loss = self._loss(t, idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return {"before": before, "after": self._loss(t, everything).item()}

    def _loss(self, t, idx):
        zd = (self._dyn_std_out(t["Y_dv"][idx]) - self._raw_dyn(t["X_dyn"][idx])) ** 2
        zn = (self._noc_std_out(t["Y_n"][idx][:, None]) - self._raw_noc(t["X_noc"][idx])) ** 2
        return zd.mean() + zn.mean()

    # standardised-space helpers for training
    def _raw_dyn(self, X):
        return self.dyn((X - self.dyn_in_m) / self.dyn_in_s)

    def _raw_noc(self, X):
        return self.noc((X - self.noc_in_m) / self.noc_in_s)

    def _dyn_std_out(self, Y):
        return (Y - self.dyn_out_m) / self.dyn_out_s

    def _noc_std_out(self, Y):
        return (Y - self.noc_out_m) / self.noc_out_s

    @torch.no_grad()
    def holdout_rmse(self, t, idx):
        d = self._raw_dyn(t["X_dyn"][idx]) * self.dyn_out_s + self.dyn_out_m - t["Y_dv"][idx]
        n = self._raw_noc(t["X_noc"][idx]) * self.noc_out_s + self.noc_out_m - t["Y_n"][idx][:, None]
        return {"dv": float(d.pow(2).mean().sqrt()), "n": float(n.pow(2).mean().sqrt())}

    # ------------------------------------------------------------------ files
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hparams": {"K": self.K, "h_dyn": self.h_dyn, "h_noc": self.h_noc},
                    "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        m = cls(**blob["hparams"])
        m.load_state_dict(blob["state"])
        return m.eval()


class AnalyticModel:
    """The true equations, same interface as WorldModel (no noise). With the true hidden parameters
    after a drift it is the oracle of checklist 2.2; with the nominal ones it is the nominal reference."""

    def __init__(self, config: WorldConfig | None = None, theta=None, psi=None, d=0, floor=0.0):
        c = config or WorldConfig()
        self.cfg, self.d, self.floor = c, int(d), floor
        self.Z = torch.tensor(c.z, dtype=torch.float32)
        self.rho2 = torch.tensor(c.rho, dtype=torch.float32) ** 2
        e = torch.tensor(c.e, dtype=torch.float32)
        self.E = e / e.norm(dim=1, keepdim=True)
        self.theta = torch.tensor(c.theta0 if theta is None else theta, dtype=torch.float32)
        self.psi = torch.zeros(len(c.z)) if psi is None else torch.tensor(psi, dtype=torch.float32)
        self.K = max(K_DEFAULT, self.d + 1)

    def _phi(self, x):
        d2 = ((x[:, None, :] - self.Z[None]) ** 2).sum(-1)
        return torch.clamp(1.0 - d2 / self.rho2, min=0.0)

    def predict_dv(self, x, v, uhist):
        c = self.cfg
        w = (self.psi * self._phi(x)) @ self.E
        drag = c.gamma * v.norm(dim=1, keepdim=True) * v
        return c.dt * (c.a_max * uhist[:, self.d] - drag + w)

    def predict_n(self, x):
        return self._phi(x) @ self.theta + self.floor
