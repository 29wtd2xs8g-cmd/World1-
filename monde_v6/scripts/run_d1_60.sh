#!/usr/bin/env bash
# Confirm criterion D1 on the 60 validation seeds (margin_frac 0.7, DEV-1).
# Resumable: finished (agent, regime, seed) jobs are skipped when the script is relaunched.
# Usage: bash scripts/run_d1_60.sh [workers]      (default 8; one single-threaded process per worker)
# Set PYTHON=/path/to/python to choose the interpreter (default: python on the PATH).
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${1:-8}"
PY="${PYTHON:-python}"
"$PY" -c "import torch, numpy, gymnasium" 2>/dev/null || {
  echo "Missing packages for $($PY -c 'import sys;print(sys.executable)'). Run: $PY -m pip install -r requirements.txt" >&2
  exit 1
}
mkdir -p results
[ -f results/baseline.pt ] || cp archive/2026-11-22_run1_margin1.0/baseline.pt results/baseline.pt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
"$PY" scripts/headroom.py run results/baseline.pt 60 "$WORKERS" 0.7 all
"$PY" scripts/headroom.py summary 0.7 | tee results/summary_rho0.7_60seeds.txt
