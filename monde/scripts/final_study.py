"""The final study: Agent 1 against Agent 2 on the test seeds (decision D10; rules D9, docs/protocol.md sections 3.6 and 3.7).

  PYTHONPATH=. python scripts/final_study.py plan results/baseline.pt
  PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/final_study.py run results/baseline.pt [workers] [--open]
  PYTHONPATH=. python scripts/final_study.py check
  PYTHONPATH=. python scripts/final_study.py summary

This is the only script that reaches the test seeds, through `seeds.test_seeds_for_study`. The first `run` opens them: it writes
one line in `protocol/test_seed_ledger.jsonl` (the purpose names the hashes of the model, the tuned thresholds and the nominal
reference) and needs `--open` to do so, because it cannot be undone. A relaunch after an interruption resumes the same study
without a second line; a changed model, threshold or reference cannot resume. The agents, thresholds, revision recipe and
rules are the ones of the validation comparison; nothing is tuned here.

`run` plays the closed-loop comparison (60 seeds x (2 agents x 8 regimes + frozen and always-adapting in R0 and R1) = 1,200
jobs, `results/test_compare.jsonl`) then the frozen-trajectory replay (60 seeds x 8 regimes = 480 jobs,
`results/test_replay.jsonl`). Both are resumable. `check` prints counts only. `summary` prints nothing until all 60 seeds are
complete (no interim look at partial results), then the verdicts of predictions B and C.
"""
import sys
from pathlib import Path

import compare_agents
import replay_detectors
from toyworld.compare import (AGENTS, REGIMES, load_thresholds, require_d9_recorded, require_d10_recorded, study_purpose)
from toyworld.nominal import NominalReference, model_hash
from toyworld.replay import REPLAY_REGIMES
from toyworld.seeds import N_FINAL_SEEDS, TEST_SEED_LEDGER, test_seeds_for_study

COMPARE = Path("results/test_compare.jsonl")
REPLAY = Path("results/test_replay.jsonl")


def guards(model):
    """Everything that must hold before the test seeds are opened. Raises or exits on the first failure."""
    require_d9_recorded()
    require_d10_recorded()
    if NominalReference.load().model_sha256 != model_hash(model):
        sys.exit("the nominal reference was computed on another model")
    load_thresholds()                                  # both agents tuned once, under one budget
    return study_purpose(model)


def ledger_state(purpose):
    import json
    lines = [json.loads(l) for l in TEST_SEED_LEDGER.read_text().splitlines() if l.strip()] if TEST_SEED_LEDGER.exists() else []
    if not lines:
        return "closed (the ledger is empty)"
    return "open for this study (a run resumes)" if len(lines) == 1 and lines[0]["purpose"] == purpose \
        else "OPENED FOR ANOTHER PURPOSE OR MORE THAN ONCE: this study cannot run"


def plan(model):
    purpose = guards(model)
    print("guards passed: D9 recorded, D10 recorded, nominal reference matches the model, both agents tuned")
    print(f"purpose written in the ledger:\n  {purpose}")
    print(f"test seeds: {ledger_state(purpose)}")
    n_cl = N_FINAL_SEEDS * (len(AGENTS) * len(REGIMES) + 4)
    print(f"jobs: {n_cl} closed-loop + {N_FINAL_SEEDS * len(REPLAY_REGIMES)} replay, {N_FINAL_SEEDS} seeds")


def run(model, workers, confirmed):
    purpose = guards(model)
    state = ledger_state(purpose)
    if state.startswith("OPENED"):
        sys.exit(state)
    if state.startswith("closed") and not confirmed:
        sys.exit("This opens the test seeds, once and for good. Check `plan`, then rerun with --open.")
    seeds = test_seeds_for_study(purpose)
    print(f"test seeds: {len(seeds)}; ledger: {ledger_state(purpose)}", flush=True)
    compare_agents.run(model, workers, seeds, COMPARE)
    replay_detectors.run(model, workers, seeds, REPLAY)


def summary():
    replay_detectors.summary(REPLAY, required=N_FINAL_SEEDS)
    print("\n" + "=" * 100 + "\n")
    compare_agents.summary(COMPARE, refs=None, replay=REPLAY, required=N_FINAL_SEEDS)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--open"]
    cmd = args[0] if args else ""
    if cmd == "plan":
        plan(args[1])
    elif cmd == "run":
        run(args[1], int(args[2]) if len(args) > 2 else 2, "--open" in sys.argv)
    elif cmd == "check":
        replay_detectors.check(REPLAY, COMPARE)
    elif cmd == "summary":
        summary()
    else:
        sys.exit(__doc__)
