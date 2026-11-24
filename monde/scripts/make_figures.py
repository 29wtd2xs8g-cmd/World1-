"""Figures for the article and the slides, drawn from the data files of the two comparison scripts.

    PYTHONPATH=. python scripts/make_figures.py \
        --compare archive/2026-11-23_validation_comparison/compare.jsonl \
        --replay archive/2026-11-23_validation_comparison/replay.jsonl \
        --out figures/validation --label validation

    PYTHONPATH=. python scripts/make_figures.py --compare results/test_compare.jsonl \
        --replay results/test_replay.jsonl --out figures/test --label test --required 60

Every number is computed with the helpers of the verdicts (`toyworld.compare`, `toyworld.replay`, `toyworld.metrics`),
so a figure and the printed summary cannot disagree. No interim look: the script draws nothing unless the files hold
`--required` complete paired seeds (50 on the validation seeds, 60 on the test seeds). It never reads the seed lists.

Figures (PNG at 200 dpi and vector PDF):
  fig1_outcomes        outcome rates per regime and agent (closed loop, tuned thresholds)
  fig2_delay           delay to the first alert against the nominal false-alarm rate (replay, equal rates)
  fig3_false_alarms    measured false-alarm rate against the nominal rate, in R0 and R1 (replay)
  fig4_loop            success and retreat rates by episode since the drift onset (closed loop)
  fig5_run_cost        exploratory: cost of a whole run and runs without any success after the onset
  fig6_headroom        where the two agents sit between "never revise" and the oracle (failure rate)
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from toyworld.compare import AGENTS, REGIMES, load_runs, seed_vectors
from toyworld.metrics import post_tau, recovery, run_totals
from toyworld.nominal import FALSE_ALARM_RATES
from toyworld.replay import REPLAY_REGIMES, VERDICT_LEVEL, level_vectors, load_replay, matched_contrast

PANELS = ("R2d", "R2c", "R2l", "R3c", "R3l", "R3d")          # the five lethal regimes, then the damage-only one
DESCRIPTION = {"R2d": "abrupt, dynamics", "R2c": "abrupt, cost", "R2l": "abrupt, latency",
               "R3c": "gradual, cost", "R3l": "gradual, latency", "R3d": "gradual, dynamics"}
EPISODES_SINCE_ONSET = 10         # every seed has at least this many post-onset episodes (onset episode 5 to 10 of 20)

# Failure rates of the reference agents on the 60 validation seeds (docs/agents.md, section 11.1): frozen, oracle, always-adapting.
# Typed from that table, not recomputed: the reference rows are not part of the comparison files.
HEADROOM_FAILURE = {"R2d": (0.65, 0.36, 0.60), "R2c": (1.00, 0.00, 0.17), "R2l": (0.80, 0.01, 0.48),
                    "R3c": (0.14, 0.00, 0.01), "R3l": (0.49, 0.00, 0.16)}

# Colours: the two agents are a dark red (Agent 1) and a dark blue (Agent 2) pair, separated by hue (colour-blind delta E 18.8 on protan);
# outcomes use the fixed status colours and are also named by a legend, printed values and a hatch on failures.
AGENT_COLOR = {"agent1_alert": "#9B1C1C", "agent2_cusum": "#1F3A8A"}
AGENT_NAME = {"agent1_alert": "Agent 1 (alert)", "agent2_cusum": "Agent 2 (prediction error)"}
AGENT_SHORT = {"agent1_alert": "Agent 1", "agent2_cusum": "Agent 2"}
OUTCOME_COLOR = {"success": "#0CA30C", "retreat": "#C3C2B7", "timeout": "#6F6F66", "failure": "#D03B3B"}
OUTCOME_ORDER = ("failure", "retreat", "timeout", "success")      # failure on the baseline: the rate to compare
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0B0B0B", "#52514E", "#898781", "#E1E0D9", "#C3C2B7", "#FCFCFB"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "hatch.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def header(fig, title, subtitle, note):
    fig.text(0.015, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.015, 0.985 - 0.048, subtitle, ha="left", va="top", fontsize=8.5, color=INK2, linespacing=1.4)
    fig.text(0.015, 0.012, note, ha="left", va="bottom", fontsize=7.5, color=INK2)


def agent_legend(fig, ncol=2, loc=(0.5, 0.055), extra=()):
    h = [Line2D([0], [0], color=AGENT_COLOR[a], lw=2.2, marker="o", ms=5, label=AGENT_NAME[a]) for a in AGENTS] + list(extra)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=loc, ncol=ncol, frameon=False, fontsize=8.5, labelcolor=INK)


def save(fig, out, name):
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(out / f"{name}.{ext}", **kw)
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# ---- figure 1 ---------------------------------------------------------------------------------------------

def fig1_outcomes(runs, seeds, note):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.85, bottom=0.17)
    y, centres = 0.0, {}
    for g in PANELS:
        ys = []
        for kind in AGENTS:
            v = seed_vectors(runs, kind, g, seeds)
            left = 0.0
            for o in OUTCOME_ORDER:
                w = float(np.nanmean(v[o]))
                ax.barh(y, w, left=left, height=0.74, color=OUTCOME_COLOR[o], edgecolor=SURFACE, linewidth=1.2)
                if o == "failure":
                    ax.barh(y, w, left=left, height=0.74, fill=False, hatch="///", edgecolor="#F0A0A0", linewidth=0)
                if w >= 0.06:
                    ax.text(left + w / 2, y, f"{100 * w:.0f}", ha="center", va="center", fontsize=8, fontweight="bold",
                            color="#FFFFFF" if o in ("failure", "success") else INK)
                left += w
            ax.text(-0.015, y, AGENT_SHORT[kind], ha="right", va="center", fontsize=8, color=INK,
                    fontweight="bold", transform=ax.get_yaxis_transform())
            ys.append(y)
            y += 1.0
        centres[g] = float(np.mean(ys))
        y += 0.55
    for g, c in centres.items():
        ax.text(-0.16, c, g, ha="right", va="center", fontsize=10, fontweight="bold", transform=ax.get_yaxis_transform())
        ax.text(-0.16, c + 0.42, DESCRIPTION[g], ha="right", va="center", fontsize=7.2, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.55, -0.6)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([f"{int(100 * t)}%" for t in np.arange(0, 1.01, 0.25)])
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    handles = [Patch(facecolor=OUTCOME_COLOR[o], edgecolor="#F0A0A0" if o == "failure" else SURFACE, hatch="///" if o == "failure" else None, label=o.capitalize())
               for o in ("success", "retreat", "timeout", "failure")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.58, 0.055), ncol=4, frameon=False, fontsize=8.5)
    header(fig, "Outcomes after the drift onset, by regime and agent",
           "Share of post-onset episodes ending in each way (percent printed when 6 or more). Tuned thresholds, closed loop.", note)
    return fig


# ---- figure 2 ---------------------------------------------------------------------------------------------

def _median_ci(x, rng, reps=2000):
    x = np.asarray(x, float)
    m = np.median(x[rng.integers(0, len(x), (reps, len(x)))], axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi)


def _steps(x, signed=False):
    """Steps as the summary prints them: one decimal only for a half step (the medians of an even number of seeds)."""
    t = f"{x:+.1f}" if signed else f"{x:.1f}"
    return t[:-2] if t.endswith(".0") else t


def fig2_delay(replay, seeds, note):
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.25, wspace=0.08, hspace=0.34)
    rng = np.random.default_rng(0)
    levels = np.array(FALSE_ALARM_RATES)
    for ax, g in zip(axes.flat, PANELS):
        for kind in AGENTS:
            med, lo, hi, cens = [], [], [], []
            for lv in range(len(levels)):
                v = level_vectors(replay, kind, g, lv, seeds)
                m, l, h = _median_ci(v["delay"], rng)
                med.append(m), lo.append(l), hi.append(h), cens.append(float(v["censored"].mean()))
            c = AGENT_COLOR[kind]
            ax.fill_between(levels, lo, hi, color=c, alpha=0.16, linewidth=0)
            ax.plot(levels, med, color=c, lw=1.8)
            for x, m, k in zip(levels, med, cens):
                ax.plot(x, m, "o", ms=4.6, color=c, mfc=SURFACE if k >= 0.5 else c, mew=1.3, zorder=3)
        ax.axvline(levels[VERDICT_LEVEL], color=INK2, lw=0.9, ls=(0, (2, 3)))
        c = matched_contrast(replay, g, VERDICT_LEVEL, seeds)
        gap = c["delay_gap"]
        who = "faster" if gap["value"] > 0 else "slower"
        ax.text(0.04, 0.96, f"At 5%: Agent 1 {who} by {_steps(abs(gap['value']))} steps\n[95% CI {_steps(gap['lo'], True)}, {_steps(gap['hi'], True)}]"
                + ("\nnot judged" if g == "R3d" else ""),
                transform=ax.transAxes, fontsize=7.2, color=INK2, va="top", ha="left", linespacing=1.35,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9, "pad": 1.5})
        ax.set_title(f"{g}  {DESCRIPTION[g]}", loc="left", fontsize=8.8,
                     fontweight="bold", color=INK, pad=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, grid="y")
        ax.tick_params(which="minor", length=0)
    axes[0, 0].set_xlim(0.0085, 0.47)
    axes[0, 0].set_ylim(2, 40000)
    axes[0, 0].set_xticks(levels)
    axes[0, 0].set_xticklabels([f"{int(round(100 * r))}%" for r in levels])
    axes[0, 0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].set_yticks([3, 10, 30, 100, 300, 1000])
    axes[0, 0].set_yticklabels(["3", "10", "30", "100", "300", "1000"])
    axes[0, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes[1]:
        ax.set_xlabel("nominal false-alarm rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("median delay (steps)")
    agent_legend(fig, ncol=2, loc=(0.5, 0.05), extra=[
        Line2D([0], [0], color=INK2, lw=0.9, ls=(0, (2, 3)), label="verdict level (5%)"),
        Line2D([0], [0], color=INK2, marker="o", mfc=SURFACE, mew=1.3, lw=0, ms=5, label="median censored (half or more never alert)")])
    header(fig, "Delay to the first alert at equal false-alarm rates",
           "Both detectors replayed on the same trajectories, set to the same nominal false-alarm rate.\nMedian delay over seeds with a 95% bootstrap band. Lower is faster.",
           note)
    return fig


# ---- figure 3 ---------------------------------------------------------------------------------------------

def fig3_false_alarms(replay, seeds, note):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.27, wspace=0.08)
    levels = np.array(FALSE_ALARM_RATES)
    floor = 4e-4
    for ax, g in zip(axes, ("R0", "R1")):
        ax.plot(levels, levels, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for kind in AGENTS:
            rate = []
            for lv in range(len(levels)):
                v = level_vectors(replay, kind, g, lv, seeds)
                rate.append(max(v["fa"] / v["fa_n"], floor))
            ax.plot(levels, rate, "o-", color=AGENT_COLOR[kind], lw=1.8, ms=5, zorder=3)
            ax.annotate(f"{rate[VERDICT_LEVEL]:.3f}", (levels[VERDICT_LEVEL], rate[VERDICT_LEVEL]), textcoords="offset points",
                        xytext=(0, 8 if kind == "agent2_cusum" else -13), ha="center", fontsize=7.8, color=INK2)
        ax.axvline(levels[VERDICT_LEVEL], color=INK2, lw=0.9, ls=(0, (2, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{g}  " + ("no drift" if g == "R0" else "drift without harm"), loc="left", fontsize=9, fontweight="bold", pad=5)
        ax.set_xlabel("nominal false-alarm rate")
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{int(round(100 * r))}%" for r in levels])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style(ax, grid="y")
        ax.tick_params(which="minor", length=0)
    axes[0].set_xlim(0.0085, 0.5)
    axes[0].set_ylim(floor * 0.8, 0.9)
    axes[0].set_yticks([0.001, 0.01, 0.1, 0.5])
    axes[0].set_yticklabels(["0.1%", "1%", "10%", "50%"])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("measured false alarms per episode")
    agent_legend(fig, ncol=3, loc=(0.5, 0.045), extra=[Line2D([0], [0], color=MUTED, lw=1.0, ls=(0, (4, 3)), label="nominal rate")])
    header(fig, "False alarms: measured against nominal rate",
           "Replay on frozen trajectories with nothing harmful to detect. Values printed at the 5% verdict level.", note)
    return fig


# ---- figure 4 ---------------------------------------------------------------------------------------------

def _by_episode_since_onset(rows, k_max):
    out = {k: None for k in range(k_max)}
    for r in post_tau(rows):
        k = r["episode"] - r["tau_ep"]
        if k < k_max:
            out[k] = r["outcome"]
    return out


def fig4_loop(runs, seeds, note):
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.2, wspace=0.08, hspace=0.34)
    ks = np.arange(EPISODES_SINCE_ONSET)
    for ax, g in zip(axes.flat, PANELS):
        for kind in AGENTS:
            per = [_by_episode_since_onset(runs[(kind, g, s)], EPISODES_SINCE_ONSET) for s in seeds]
            for outcome, ls in (("success", "-"), ("retreat", (0, (4, 2)))):
                share = [np.mean([p[k] == outcome for p in per]) for k in ks]
                ax.plot(ks, share, color=AGENT_COLOR[kind], lw=1.8, ls=ls)
        ax.set_title(f"{g}  {DESCRIPTION[g]}", loc="left", fontsize=8.8, fontweight="bold", pad=5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_xticks([0, 3, 6, 9])
        style(ax, grid="y")
    for ax in axes[1]:
        ax.set_xlabel("episodes since the onset")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of runs")
    agent_legend(fig, ncol=4, loc=(0.5, 0.068), extra=[
        Line2D([0], [0], color=INK2, lw=1.8, ls="-", label="ends in success"),
        Line2D([0], [0], color=INK2, lw=1.8, ls=(0, (4, 2)), label="ends in a retreat")])
    header(fig, "Does the agent keep alerting and retreating after the onset?",
           "Share of runs whose episode ends in success (solid) or in a retreat (dashed),\nby episode since the drift onset (episode 0 is the onset episode).",
           note)
    return fig


# ---- figure 5 ---------------------------------------------------------------------------------------------

def _mean_ci(x, rng, reps=2000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def fig5_run_cost(runs, seeds, note):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.975, top=0.80, bottom=0.27, wspace=0.12)
    rng = np.random.default_rng(0)
    for j, ax in enumerate(axes):
        for i, g in enumerate(PANELS):
            for d, kind in zip((-0.17, 0.17), AGENTS):
                rows_by_seed = [runs[(kind, g, s)] for s in seeds]
                if j == 0:
                    vals = [run_totals(post_tau(r))["ccum_run"] for r in rows_by_seed]
                else:
                    vals = [float(recovery(r)["censored"]) for r in rows_by_seed]
                m, lo, hi = _mean_ci(vals, rng)
                ax.plot([lo, hi], [i + d, i + d], color=AGENT_COLOR[kind], lw=2.2, solid_capstyle="round", alpha=0.55)
                ax.plot(m, i + d, "o", color=AGENT_COLOR[kind], ms=6, mec=SURFACE, mew=1.0, zorder=3)
        ax.set_ylim(len(PANELS) - 0.45, -0.6)
        style(ax, grid="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(PANELS)))
    axes[0].set_yticklabels([f"{g}\n{DESCRIPTION[g]}" for g in PANELS], fontsize=8)
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlabel("damage over the whole run (Ccum)")
    axes[0].set_xlim(left=0)
    axes[1].set_xlabel("runs with no success after the onset")
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[1].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    agent_legend(fig, ncol=2, loc=(0.58, 0.045))
    header(fig, "Exploratory: what a whole run costs, and how often it never delivers",
           "Mean over seeds with 95% bootstrap interval. Computed after the verdicts were read; no registered threshold.", note)
    return fig


# ---- figure 6 ---------------------------------------------------------------------------------------------

def fig6_headroom(runs, seeds, note):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.subplots_adjust(left=0.17, right=0.975, top=0.80, bottom=0.30)
    regs = [g for g in PANELS if g in HEADROOM_FAILURE]
    for i, g in enumerate(regs):
        frozen, oracle, always = HEADROOM_FAILURE[g]
        ax.plot([oracle, frozen], [i, i], color=AXIS, lw=5, solid_capstyle="round", zorder=1)
        ax.plot(frozen, i, "s", ms=7, mfc=SURFACE, mec=INK2, mew=1.5, zorder=3)
        ax.plot(always, i, "^", ms=7, mfc=SURFACE, mec=INK2, mew=1.5, zorder=3)
        ax.plot(oracle, i, "D", ms=6.5, color=INK, zorder=3)
        for kind, d in zip(AGENTS, (-0.17, 0.17)):
            f = float(np.nanmean(seed_vectors(runs, kind, g, seeds)["failure"]))
            ax.plot(f, i + d, "o", ms=8, color=AGENT_COLOR[kind], mec=SURFACE, mew=1.0, zorder=4)
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels([f"{g}\n{DESCRIPTION[g]}" for g in regs], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(regs) - 0.5, -0.6)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([f"{int(100 * t)}%" for t in np.arange(0, 1.01, 0.25)])
    ax.set_xlabel("failure rate after the onset")
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    refs = [Line2D([0], [0], marker="s", mfc=SURFACE, mec=INK2, mew=1.5, lw=0, ms=6, label="never revise"),
            Line2D([0], [0], marker="^", mfc=SURFACE, mec=INK2, mew=1.5, lw=0, ms=6, label="always revise"),
            Line2D([0], [0], marker="D", color=INK, lw=0, ms=5.5, label="oracle")]
    agents = [Line2D([0], [0], marker="o", color=AGENT_COLOR[a], lw=0, ms=7, label=AGENT_NAME[a]) for a in AGENTS]
    fig.legend(handles=refs + agents, loc="lower center", bbox_to_anchor=(0.58, 0.07), ncol=3, frameon=False, fontsize=8.3)
    header(fig, "Where the two agents sit between 'never revise' and the oracle",
           "Failure rate after the onset. Grey bar: the room a trigger can use. Lower is better.",
           note + " References: 60 validation seeds (agents.md 11.1).")
    return fig


FIGURES = (("fig1_outcomes", fig1_outcomes, "runs"), ("fig2_delay", fig2_delay, "replay"),
           ("fig3_false_alarms", fig3_false_alarms, "replay"), ("fig4_loop", fig4_loop, "runs"),
           ("fig5_run_cost", fig5_run_cost, "runs"), ("fig6_headroom", fig6_headroom, "runs"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", required=True, help="jsonl of scripts/compare_agents.py")
    ap.add_argument("--replay", required=True, help="jsonl of scripts/replay_detectors.py")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--label", default="validation", help="shown in the footer: validation or test")
    ap.add_argument("--required", type=int, default=50, help="complete paired seeds required (50 validation, 60 test)")
    args = ap.parse_args()

    runs, replay = load_runs(args.compare), load_replay(args.replay)
    found = sorted({s for (_, _, s) in runs})
    seeds = [s for s in found
             if all((k, g, s) in runs for k in AGENTS for g in REGIMES) and all((g, s) in replay for g in REPLAY_REGIMES)]
    if len(seeds) < args.required:
        sys.exit(f"{len(seeds)} complete seeds of {args.required} required: no figure is drawn before the study is complete")
    note = f"{args.label.capitalize()} seeds, {len(seeds)} paired scenarios."
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, src in FIGURES:
        save(fn(runs if src == "runs" else replay, seeds, note), out, name)


if __name__ == "__main__":
    main()
