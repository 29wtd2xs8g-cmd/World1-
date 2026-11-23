# Run 1: first headroom measurement (first protocol, `margin_frac` 1.0)

Exact record of the run that motivated deviation DEV-1 (see `docs/protocol_deviations.md`). Not to be overwritten.

| File | Content |
|---|---|
| `headroom.jsonl` | oracle and always-adapting runs, 6 harmful regimes x 6 validation seeds (2000-2005), `margin_frac` 1.0 |
| `regimes_baseline_frozen.json` | the frozen agent's runs that run 1 read as its frozen reference |
| `baseline.pt` | the pretrained world model used |
| `summary.txt` | the summary printed by the first version of `scripts/headroom.py` (reported figures) |
| `summary_outcome_first.txt` | the same data read by the outcome-first summary of `scripts/headroom.py` |

**Known defect of this archive.** `regimes_baseline_frozen.json` was written before the final amplitude of R3d was raised
from 0.25 to 0.35, so its R3d rows are stale (the other five regimes match the current code episode by episode). Run 1
therefore compared a frozen R3d at 0.25 with an oracle and an always-adapting agent at 0.35. Corrected with the current
code: frozen R3d `Ccum` 1.67 instead of 1.62, headroom of the oracle +1.5% instead of -1.6%. No verdict changes.
