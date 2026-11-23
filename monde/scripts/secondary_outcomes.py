"""Secondary analysis of decision D11: non-completion of the task (failure + retreat + timeout), per regime.

  PYTHONPATH=. python scripts/secondary_outcomes.py archive/2026-11-23_validation_comparison/compare.jsonl
  PYTHONPATH=. python scripts/secondary_outcomes.py results/test_compare.jsonl --label test --required 60

Reads the closed-loop file of `compare_agents.py`. Changes no verdict and no registered definition: `failure` keeps its meaning
(irreversible failure). Refuses to print until every required paired seed is complete, like the summaries of the verdicts.
"""
import argparse
import sys

from toyworld.secondary import complete_run_seeds, reading, secondary_table
from toyworld.compare import load_runs

HEADINGS = {
    "validation": "POST HOC (computed after the verdicts were read), on the validation seeds",
    "test": "SECONDARY ANALYSIS registered as D11 before the test summary was read",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compare", help="jsonl of scripts/compare_agents.py")
    ap.add_argument("--label", default="validation", choices=sorted(HEADINGS))
    ap.add_argument("--required", type=int, default=50, help="complete paired seeds required (50 validation, 60 test)")
    args = ap.parse_args()
    try:
        seeds = complete_run_seeds(load_runs(args.compare), args.required)
    except ValueError as e:
        sys.exit(str(e))
    table = secondary_table(load_runs(args.compare), seeds)

    print(f"Non-completion of the task = 1 - success = failure + retreat + timeout. {HEADINGS[args.label]}.")
    print("Not part of the verdicts of B and C; `failure` keeps its registered definition (irreversible failure).")
    print(f"{len(seeds)} paired seeds, post-onset episodes; gap = Agent 2 minus Agent 1 (positive = Agent 1 completes more often);")
    print("95% paired bootstrap interval, Holm over the five lethal regimes (R3d reported, not judged).\n")
    print("regime | Agent 1: fail retreat timeout | non-compl | Agent 2: fail retreat timeout | non-compl |"
          "     gap (points) [95% CI]  Holm p | reading")
    for c in table:
        g = c["noncompletion_gap"]
        p = "     -" if c["holm_p"] is None else f"{c['holm_p']:6.3f}"
        print(f"{c['regime']:6s} |          {c['failure_a']:4.2f}  {c['retreat_a']:5.2f}  {c['timeout_a']:6.2f} |"
              f"      {c['noncompletion_a']:4.2f} |          {c['failure_b']:4.2f}  {c['retreat_b']:5.2f}  {c['timeout_b']:6.2f} |"
              f"      {c['noncompletion_b']:4.2f} | {100 * g['mean']:+7.1f} [{100 * g['lo']:+6.1f}, {100 * g['hi']:+6.1f}] {p} |"
              f" {reading(c)}")


if __name__ == "__main__":
    main()
