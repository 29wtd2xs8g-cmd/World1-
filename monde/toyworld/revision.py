"""Model revision (decision D2, docs/research.md section 6): fine-tune both networks on the window of the last
episode, at the end of the episode, without replay. Every adaptive agent uses the same budget and hyperparameters.

The window holds only what the agent observed and commanded: (o[t], u[t], o[t+1]). Features and targets follow
`data.collect` exactly (command history newest first, zero-padded before the episode start).
"""
from dataclasses import dataclass

import numpy as np

from .model import K_DEFAULT


@dataclass(frozen=True)
class RevisionParams:
    steps: int = 50          # gradient steps per revision (the gradient budget)
    batch: int = 128
    lr: float = 1e-3


def window_data(O, U, K=K_DEFAULT):
    """O: observations o[0..T'] (T'+1, 5); U: applied commands u[0..T'-1] (T', 2). Returns the training dict."""
    O, U = np.asarray(O, float), np.asarray(U, float)
    T = len(U)
    padded = np.vstack([np.zeros((K - 1, 2)), U])
    X_dyn = np.array([np.concatenate([O[t, 0:2], O[t, 2:4], padded[t:t + K][::-1].ravel()]) for t in range(T)])
    return {"X_dyn": X_dyn, "Y_dv": O[1:T + 1, 2:4] - O[:T, 2:4], "X_noc": O[:T, 0:2], "Y_n": O[:T, 4]}


def revise(model, O, U, params: RevisionParams, seed=0):
    """Fine-tune `model` in place on the window (O, U). Returns the loss before/after."""
    return model.finetune(window_data(O, U, model.K), steps=params.steps, batch=params.batch, lr=params.lr, seed=seed)
