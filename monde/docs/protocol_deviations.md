# Protocol deviations

Any change to a frozen element of the protocol made after a result was seen is logged here: date, trigger, cause,
change, what stays valid, and the data that motivated it. The data of the run that triggered a deviation is archived
(`archive/`) and never overwritten.

## DEV-1 (2026-11-22): safety buffer in the route constraint, `margin_frac` 1.0 -> 0.7

**What was frozen.** The baseline's route rule: a route is allowed if its predicted damage is below the whole
estimated margin `b_hat` ([agents.md](agents.md), section 2). Criterion D1 ([research.md](research.md), section 6).

**Trigger.** First headroom measurement (run 1: 6 validation seeds 2000-2005, 20 episodes, post-`tau` episodes). D1 was
not met. Exact figures of the regime that carries the question:

| R3c (run 1, `margin_frac` 1.0) | `Ccum` (all post-`tau` episodes) | Failure | Route A |
|---|---|---|---|
| frozen | 2.35 | 0.11 | 1.00 |
| oracle | 2.37 | 0.15 | 1.00 |
| always-adapting | 1.91 | 0.05 | 0.83 |

Headroom of the oracle on R3c: **-0.8%** (95% interval -1.3 to -0.3). On R3d: -1.6%. Only R2c (+87.4%), R2l (+28.7%)
and R3l (+21.5%) reached 15%, so D1 failed on its second condition (one of R3d, R3c). Data, model and summaries:
`archive/2026-11-22_run1_margin1.0/` (SHA-256 of the files below).

| File | SHA-256 (first 16 characters) |
|---|---|
| `headroom.jsonl` | `59bbc9e8b4ad4148` |
| `regimes_baseline_frozen.json` | `af806212ae395878` |
| `baseline.pt` | `fd1278e20d501d15` |
| `summary.txt` | `a31315e469f46a91` |

**Mechanical cause.** The constraint had no safety buffer. Route A stayed allowed while its predicted damage was below
the whole remaining margin, so even an oracle moved to route B only when `theta_1` reached `D / C1'` = 2.0, at the end of
the ramp, where the realised damage (noise about 0.08) already fails often. The oracle behaved like the frozen agent
for most of the drift. The headroom was nil because of the rule, not because knowing the cost is useless. In R3d the
drift is harmful (+15% damage) but never fatal (`Ccum` 1.6 against `D` = 2.98), which is a property of the regime.

**Change.** `AgentParams.margin_frac` = 0.7: a route is allowed if its predicted damage is below 0.7 `b_hat`. At the
start of an episode the threshold is 0.7 x 2.981 = 2.087 (a product; it is not `D - 0.7`).

- **One rule for every agent.** It is the default of `AgentParams`, so frozen, oracle, always-adapting, Agent 1 and
  Agent 2 all use it; no agent has a value of its own.
- **0.7 was fixed a priori and never tuned.** It was the only value ever run: first in an exploratory R3c run, then in the full
  re-measurement. It was chosen after seeing run 1 (hence this entry), not after trying several values.
- **No contact with the test seeds.** `protocol/test_seed_ledger.jsonl` is empty and a test checks that no code
  outside `toyworld/seeds.py` refers to the test seeds. The amendment was decided and measured on validation seeds only.

**Finding during the re-measurement: a stale frozen reference on R3d in run 1.** Run 1 read its frozen runs from
`results/regimes_baseline.json`, written before the final amplitude of R3d was raised from 0.25 to 0.35. For R3d the frozen agent
was thus compared with an oracle and an always-adapting agent that ran at 0.35. Recomputing the frozen agent with the current code on
all 36 regime-seed pairs shows that the other five regimes match the file episode by episode, and that R3d does not.
Corrected figures for run 1: frozen R3d `Ccum` 1.67 (not 1.62), oracle headroom on R3d **+1.5%** (not -1.6%). No verdict
changes: R3d never fails with any agent. The archive keeps the file as it was and says so
(`archive/2026-11-22_run1_margin1.0/README.md`); `scripts/headroom.py` no longer reads it.

**Nominal check (R0, before any drift).** With the learned model the predicted damage of route A at the start of an
episode is 1.48 and that of route B 0.38; the threshold is 2.087, so route A is still chosen. Competence criteria
re-run with 0.7: see [agents.md](agents.md), section 6.

**What stays valid.** `D` = 2.981 (calibrated on route A, which is unchanged before drift); the regimes; the frozen
agent: its predictions do not change, so its route does not. Checked by recomputation: 12 frozen runs (2 seeds x 6 regimes)
computed under both 0.7 and 1.0 are identical episode by episode. The revision budget was chosen with `margin_frac` 1.0 and
was re-checked under 0.7 ([agents.md](agents.md), section 11).

**Cost and trade-off.** Route B is longer, so successful crossings take a few more steps. A buffer also takes away
freedom to explore and correct: the agent has 2.087 of predicted damage to spend on route A instead of 2.981.

**How D1 is judged afterwards.** Literal definition unchanged (`Ccum` over all post-`tau` episodes, 15%, 3 regimes, one
of R3d and R3c), computed on the amended baseline. The failure-first reading of [protocol.md](protocol.md) is reported
beside it. Run 1 stays in the record as the first protocol.

**Result under the amended baseline** (6 validation seeds, [agents.md](agents.md), section 11): D1 is met, with four regimes
above 15% (R2c, R2l, R3c, R3l), R3c among them at +58.1% (95% interval +53.6 to +62.0; failure 0.11 to 0.00). 

**Confirmed on the 60 validation seeds (2026-11-23, [agents.md](agents.md), section 11.1):** D1 is met, with the same four
regimes above 15% (R2c +87.3%, R3c +58.8% [57.6, 60.0], R2l +29.5%, R3l +22.9%); R2d +5.1% and R3d +1.6% stay below.
Failure rate in R3c: 0.14 (frozen) to 0.00 (oracle). No test seed was used.

## Incident log

### INC-1 (2026-11-23): tuning files lost and restored

`protocol/tuning_ledger.jsonl` and `protocol/trigger_thresholds.json`, written by `scripts/tune_triggers.py` at 02:15 (Paris),
disappeared from the working folder during a folder update at about 12:40, before they had been committed (`results/` was lost
in the same update, as it had been on 2026-11-23 at 00:27). They were restored the same day, to the byte, from a copy of their
text read from the folder at 12:30, before the loss: the restored files have the original sizes (922 and 819 bytes) and the
values printed by the tuning run (Agent 1: 0.1142; Agent 2: 15.4674). Tuning is deterministic, so a rerun would reproduce them;
the once-per-agent guard blocks a rerun unless the ledger is removed first. No test seed was involved.

The tag `thresholds-fixed` was first created on a commit that did not yet contain these two files; it was deleted and
recreated on the commit that does.

Rule from now on: update the folder by unzipping over it (`unzip -o`), never by replacing it, and commit files of `protocol/`
as soon as they are written.

### INC-2 (2026-11-23): D9 commit and tag created before the content was applied

At 16:58 the commit `95b8f83` ("Record D9 ...") and the tag `d9-recorded` existed, but the working folder still held the earlier
files: section 3.6 of `protocol.md` was not in them (the overlay of `monde_v6.zip` had not been applied; the old folder had been
moved to the Trash and put back in between, with no data lost). The tag pointed to an earlier commit. The overlay was applied at
about 17:00 and committed with the text of D9; the tag was moved to that commit. No result of the comparison had been read at any
of these moments (`compare_agents.py summary` and `replay_detectors.py summary` refuse to run without the line `D9-STATUS:
RECORDED`, which was absent from the folder until then). The D9 rules themselves are unchanged from the text shown to the author
and approved at 16:31.
