"""Equal tuning budget for every tunable agent (checklist 3.3; docs/protocol.md, section 3).

Every agent that has parameters to tune (the thresholds of Agents 1 and 2) goes through `tune`, which enforces the same
budget: the same number of candidate configurations, the same tuning seeds, the same regimes, the same episodes,
the same score (`metrics.tuning_score`). Tuning seeds must belong to the validation list, never to the test list.
Each tuning is appended to `protocol/tuning_ledger.jsonl` (tracked by git), and `check_equal_budgets` lets a
reviewer verify that no agent received more.

The revision recipe (`revision.RevisionParams`) is common to all adaptive agents (decision D2); it is not part of the
per-agent budget. Frozen and oracle have nothing to tune.
"""
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .seeds import VALIDATION_SEEDS

LEDGER = Path(__file__).resolve().parent.parent / "protocol" / "tuning_ledger.jsonl"


@dataclass(frozen=True)
class TuningBudget:
    candidates: int = 6                                      # configurations tried per agent
    seeds: tuple = VALIDATION_SEEDS[30:40]                   # 10 validation seeds, apart from the headroom seeds
    regimes: tuple = ("R0", "R1", "R2c", "R3c")              # two calm regimes (false alarms), two lethal ones
    n_episodes: int = 20

    @property
    def runs(self):
        return self.candidates * len(self.seeds) * len(self.regimes)

    @property
    def episodes(self):
        return self.runs * self.n_episodes


BUDGET = TuningBudget()


def tune(agent, candidates, evaluate, budget: TuningBudget = BUDGET, ledger: Path = LEDGER):
    """`evaluate(candidate, seeds, regimes, n_episodes)` returns the score of one candidate (lower is better), normally
    `metrics.tuning_score` of the episodes it ran. Returns (best candidate, scores). Ties go to the first candidate."""
    if len(candidates) != budget.candidates:
        raise ValueError(f"{agent}: {len(candidates)} candidates, the budget is {budget.candidates} for every agent")
    if not set(budget.seeds) <= set(VALIDATION_SEEDS):
        raise ValueError("tuning is allowed on validation seeds only")
    scores = [float(evaluate(c, budget.seeds, budget.regimes, budget.n_episodes)) for c in candidates]
    best = min(range(len(scores)), key=lambda i: (scores[i], i))
    ledger = Path(ledger)
    ledger.parent.mkdir(exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps({"date": date.today().isoformat(), "agent": agent, "budget": asdict(budget),
                            "candidates": list(candidates), "scores": scores, "chosen": best}) + "\n")
    return candidates[best], scores


def check_equal_budgets(ledger: Path = LEDGER):
    """Raises if two agents did not receive exactly the same budget, or if one was tuned twice. Returns the common
    budget, or None when nothing has been tuned yet."""
    ledger = Path(ledger)
    entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    agents = [e["agent"] for e in entries]
    if len(set(agents)) != len(agents):
        raise AssertionError(f"an agent was tuned more than once: {sorted(a for a in set(agents) if agents.count(a) > 1)}")
    budgets = {json.dumps(e["budget"], sort_keys=True) for e in entries}
    if len(budgets) > 1:
        raise AssertionError("agents received different tuning budgets")
    return json.loads(budgets.pop()) if budgets else None
