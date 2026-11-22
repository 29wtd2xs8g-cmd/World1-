"""Common trigger interface (checklist 2.4).

Agents 1 and 2 are the same agent: same model, planner, revision and response to an alert. Only the trigger
differs. A trigger reads a `StepRecord` (agent-observable quantities only, never the ground truth) and decides

  * `step(rec)`: raise an alert now? The first alert of an episode makes the agent retreat (decision D3,
    see `adaptive.py`);
  * `revise_at_end(alerted)`: fine-tune the model at the end of this episode (decision D2)?

The two limit triggers frame every comparison:
  NeverTrigger   never alerts, never revises  -> the frozen reference;
  AlwaysTrigger  never alerts, revises at the end of every episode -> the always-adapting reference.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepRecord:
    """What a trigger may read about the transition t-1 -> t. Everything comes from observations, the agent's own
    commands and its own model."""
    t: int                    # step index inside the episode
    obs: np.ndarray           # o[t] = (x, y, vx, vy, n), noisy
    n_pred: float             # nociceptive signal predicted by the agent's model at the observed position
    dv_resid: np.ndarray      # observed minus predicted velocity change over the last step, shape (2,)
    b_hat: float              # running margin estimate D - dt * sum(n); the floor of n is NOT removed


STEP_RECORD_FIELDS = tuple(StepRecord.__dataclass_fields__)


class Trigger:
    name = "trigger"

    def reset_run(self):
        """Called once at the start of a run (a sequence of episodes)."""

    def reset_episode(self):
        """Called at the start of every episode."""

    def step(self, rec: StepRecord) -> bool:
        """True = alert now. Not called any more in an episode after its first alert."""
        return False

    def end_episode(self):
        """Called once when an episode ends, before `revise_at_end` (and also for the oracle)."""

    def revise_at_end(self, alerted: bool) -> bool:
        """Default: revise when the episode contained an alert."""
        return alerted


class NeverTrigger(Trigger):
    name = "never"

    def revise_at_end(self, alerted):
        return False


class AlwaysTrigger(Trigger):
    name = "always"

    def revise_at_end(self, alerted):
        return True
