"""Frozen seed lists (checklist 3.2 and 3.3): tuning and evaluation never share seeds.

- CALIBRATION: measures C1 and D (scripts/calibrate.py), and the revision budget (scripts/tune_revision.py).
- TRAIN: pretraining data of the world model.
- VALIDATION: tuning of hyperparameters and thresholds, and the headroom (checklist 2.3). 60 paired seeds.
- TEST: conclusions only, used once by the final study, through `open_test_seeds`. 60 paired seeds.

History: VALIDATION and TEST had 30 seeds each; on 2026-11-22 they were extended to 60 (decision D8, N > 50) before
any test seed had been used. The first 30 of each list are unchanged.
"""
import json
from datetime import date
from pathlib import Path

CALIBRATION_SEEDS = tuple(range(1000, 1050))
VALIDATION_SEEDS = tuple(range(2000, 2060))
TEST_SEEDS = tuple(range(3000, 3060))
TRAIN_SEEDS = tuple(range(4000, 4100))

N_FINAL_SEEDS = 60            # paired seeds of the final study (decision D8: more than 50)
BOOTSTRAP_REPS = 5000         # bootstrap replications over seeds (decision D8)

TEST_SEED_LEDGER = Path(__file__).resolve().parent.parent / "protocol" / "test_seed_ledger.jsonl"


def open_test_seeds(purpose: str):
    """The only door to the test seeds. It records the access in `protocol/test_seed_ledger.jsonl` (tracked by git)
    and refuses a second access: the test seeds serve once, for the final result."""
    if TEST_SEED_LEDGER.exists() and TEST_SEED_LEDGER.read_text().strip():
        raise PermissionError("the test seeds have already been used once (protocol/test_seed_ledger.jsonl)")
    TEST_SEED_LEDGER.parent.mkdir(exist_ok=True)
    with open(TEST_SEED_LEDGER, "a") as f:
        f.write(json.dumps({"date": date.today().isoformat(), "purpose": purpose, "seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]]}) + "\n")
    return TEST_SEEDS
