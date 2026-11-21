"""Pretrain the world model of the baseline agent in the nominal world (docs/agents.md, sections 3-4).
Usage: PYTHONPATH=. python scripts/train_baseline.py [out.pt]"""
import sys
import time

from toyworld.data import collect
from toyworld.model import WorldModel
from toyworld.seeds import TRAIN_SEEDS

out = sys.argv[1] if len(sys.argv) > 1 else "results/baseline.pt"
t0 = time.time()
data = collect(TRAIN_SEEDS, segments_per_seed=20, seg_len=100)
print(f"{len(data['Y_n'])} samples collected in {time.time() - t0:.0f} s")
model = WorldModel()
rmse = model.fit(data, epochs=30, verbose=True)
print(f"holdout RMSE: dv {rmse['dv']:.4f} (process noise floor about 0.032, observation noise adds about 0.028), n {rmse['n']:.4f} (sigma_n = 0.1)")
model.save(out)
print(f"saved {out} ({time.time() - t0:.0f} s)")
