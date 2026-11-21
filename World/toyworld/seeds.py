"""Frozen seed lists (checklist 3.3): tuning and evaluation never share seeds.

- CALIBRATION: used once to measure C1 and D (scripts/calibrate.py uses 1000..1049).
- TRAIN: pretraining data of the world model.
- VALIDATION: tuning of hyperparameters and thresholds.
- TEST: conclusions only; not used before the final comparison.
"""
CALIBRATION_SEEDS = tuple(range(1000, 1050))
VALIDATION_SEEDS = tuple(range(2000, 2030))
TEST_SEEDS = tuple(range(3000, 3030))
TRAIN_SEEDS = tuple(range(4000, 4100))
