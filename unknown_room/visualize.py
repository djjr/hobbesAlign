"""
Post-run visualization for Unknown Room simulation logs.

Single run (6-panel dashboard):
    python -m unknown_room.visualize runs/ep.json

Compare reward functions (welfare overlay):
    python -m unknown_room.visualize runs/a.json runs/b.json \\
        --labels "Individual" "Collective" --title "Reward comparison"

Save to file instead of displaying:
    python -m unknown_room.visualize runs/ep.json --out figures/ep.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; overridden if --show
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_log(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

ACTION_TYPES = ["INTERACT", "GIVE", "CLAIM_SHARE", "CLAIM_ALL",
                "MOVE", "SHUFFLE", "DO_NOTHING"]

ACTION_COLORS = {
    "INTERACT":    "#4C72B0",
    "GIVE":        "#55A868",
    "CLAIM_SHARE": "#C44E52",
    "CLAIM_ALL":   "#DD8452",
    "MOVE":        "#8172B2",
    "SHUFFLE":     "#937860",
    "DO_NOTHING":  "#CCCCCC",
}


def extract_series(log: list[dict]) -> dict:
    ticks = [e["tick"] for e in log]
    welfare = [e["collective_welfare"] for e in log]
    living = [e["living_agents"] for e in log]
    pool_counts = [sum(1 for a in e["actions"]
                       if a["action_type"] in ("CLAIM_SHARE", "CLAIM_ALL")
                       and not a["skip_reason"]) for e in log]

    # Active pools per tick: count unique pool target_ids in CLAIM actions
    # (proxy: non-zero pool target interactions)
    active_pools = []
    for e in log:
        pool_targets = set(
            a["target_id"] for a in e["actions"]
            if a["action_type"] in ("CLAIM_SHARE", "CLAIM_ALL")
            and a["target_id"] is not None
        )
        # Also count INTERACT coalitions that produced pools (success + multiple
        # agents same target)
        interact_targets: dict[int, int] = defaultdict(int)
        for a in e["actions"]:
            if a["action_type"] == "INTERACT" and a["success"] and a["target_id"] is not None:
                interact_targets[a["target_id"]] += 1
        new_pools = sum(1 for cnt in interact_targets.values() if cnt > 1)
        active_pools.append(new_pools)

    # Action type distribution per tick (proportion of logged, non-skipped actions)
    action_dist: dict[str, list[float]] = {at: [] for at in ACTION_TYPES}
    for e in log:
        acts = [a for a in e["actions"] if not a["skip_reason"]]
        total = len(acts) or 1
        counts = defaultdict(int)
        for a in acts:
            counts[a["action_type"]] += 1
        for at in ACTION_TYPES:
            action_dist[at].append(counts[at] / total)

    # Zone populations over time
    zone_populations: dict[int, list[int]] = defaultdict(list)
    for e in log:
        zone_counts: dict[int, int] = defaultdict(int)
        for a in e["actions"]:
            if not a["skip_reason"]:
                zone_counts[a["zone_id"]] += 1
        for z in range(5):
            zone_populations[z].append(zone_counts.get(z, 0))

    # Mean yield per successful INTERACT, per tick
    mean_yield = []
    for e in log:
        yields = [a["yield_amount"] for a in e["actions"]
                  if a["action_type"] == "INTERACT" and a["success"]
                  and a["yield_amount"] is not None]
        mean_yield.append(np.mean(yields) if yields else 0.0)

    # Skipped (engaged-as-target) count per tick
    skipped = [sum(1 for a in e["actions"] if a["skip_reason"] == "engaged_as_target")
               for e in log]

    return {
        "ticks": ticks,
        "welfare": welfare,
        "living": living,
        "active_pools": active_pools,
        "action_dist": action_dist,
        "zone_populations": zone_populations,
        "mean_yield": mean_yield,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Six-panel single-run dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(log: list[dict], title: str = "", out: str | None = None, show: bool = False):
    s = extract_series(log)
    ticks = s["ticks"]
    n_ticks = len(ticks)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title or "Unknown Room — Episode Summary", fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.38, wspace=0.32, top=0.91)

    # ------------------------------------------------------------------
    # Panel 1 (top-left): Collective welfare + living agents
    # ------------------------------------------------------------------
    ax1 = axes[0, 0]
    color_w = "#2196F3"
    color_l = "#F44336"

    ax1.plot(ticks, s["welfare"], color=color_w, linewidth=2, label="Collective welfare")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Collective welfare", color=color_w)
    ax1.tick_params(axis="y", labelcolor=color_w)
    ax1.set_xlabel("Tick")
    ax1.set_title("Welfare & Survival")

    ax1b = ax1.twinx()
    ax1b.plot(ticks, s["living"], color=color_l, linewidth=1.5,
              linestyle="--", alpha=0.8, label="Living agents")
    ax1b.set_ylabel("Living agents", color=color_l)
    ax1b.tick_params(axis="y", labelcolor=color_l)
    ax1b.set_ylim(0, max(s["living"]) * 1.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")

    # ------------------------------------------------------------------
    # Panel 2 (top-center): Action type distribution (stacked area)
    # ------------------------------------------------------------------
    ax2 = axes[0, 1]
    bottoms = np.zeros(n_ticks)
    for at in ACTION_TYPES:
        vals = np.array(s["action_dist"][at])
        ax2.fill_between(ticks, bottoms, bottoms + vals,
                         color=ACTION_COLORS[at], alpha=0.85, label=at.replace("_", " ").title())
        bottoms += vals

    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Proportion of actions")
    ax2.set_title("Action Mix")
    ax2.legend(fontsize=6, loc="upper right", ncol=2)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # ------------------------------------------------------------------
    # Panel 3 (top-right): Zone populations
    # ------------------------------------------------------------------
    ax3 = axes[0, 2]
    zone_colors = ["#E91E63", "#9C27B0", "#FF9800", "#00BCD4", "#4CAF50"]
    for z in range(5):
        ax3.plot(ticks, s["zone_populations"][z],
                 color=zone_colors[z], linewidth=1.5, label=f"Zone {z}")
    ax3.set_xlabel("Tick")
    ax3.set_ylabel("Active agents")
    ax3.set_title("Zone Populations")
    ax3.legend(fontsize=7, loc="upper right")

    # ------------------------------------------------------------------
    # Panel 4 (bottom-left): Coalition & pool activity
    # ------------------------------------------------------------------
    ax4 = axes[1, 0]
    bar_color = "#7986CB"
    ax4.bar(ticks, s["active_pools"], color=bar_color, alpha=0.8, label="New pools formed")
    ax4.set_xlabel("Tick")
    ax4.set_ylabel("Coalition pools formed")
    ax4.set_title("Coalition Activity")
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax4b = ax4.twinx()
    ax4b.plot(ticks, s["skipped"], color="#FF7043", linewidth=1.5,
              alpha=0.8, label="Agents skipped\n(engaged as target)")
    ax4b.set_ylabel("Agents skipped", color="#FF7043")
    ax4b.tick_params(axis="y", labelcolor="#FF7043")
    ax4b.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    lines4, labels4 = ax4.get_legend_handles_labels()
    lines4b, labels4b = ax4b.get_legend_handles_labels()
    ax4.legend(lines4 + lines4b, labels4 + labels4b, fontsize=7, loc="upper right")

    # ------------------------------------------------------------------
    # Panel 5 (bottom-center): Mean interaction yield per tick
    # ------------------------------------------------------------------
    ax5 = axes[1, 1]
    ax5.fill_between(ticks, s["mean_yield"], alpha=0.4, color="#26A69A")
    ax5.plot(ticks, s["mean_yield"], color="#00695C", linewidth=1.5)
    # Rolling mean
    window = max(3, n_ticks // 10)
    kernel = np.ones(window) / window
    smoothed = np.convolve(s["mean_yield"], kernel, mode="same")
    ax5.plot(ticks, smoothed, color="#004D40", linewidth=2,
             linestyle="--", label=f"{window}-tick rolling mean")
    ax5.set_xlabel("Tick")
    ax5.set_ylabel("Mean yield (successful INTERACT)")
    ax5.set_title("Extraction Yield")
    ax5.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel 6 (bottom-right): Interaction success rate
    # ------------------------------------------------------------------
    ax6 = axes[1, 2]
    success_rates = []
    for e in log:
        interacts = [a for a in e["actions"] if a["action_type"] == "INTERACT"
                     and not a["skip_reason"]]
        if interacts:
            rate = sum(1 for a in interacts if a["success"]) / len(interacts)
        else:
            rate = 0.0
        success_rates.append(rate)

    gives = [s["action_dist"]["GIVE"][i] for i in range(n_ticks)]
    claim_share = [s["action_dist"]["CLAIM_SHARE"][i] for i in range(n_ticks)]
    claim_all = [s["action_dist"]["CLAIM_ALL"][i] for i in range(n_ticks)]

    ax6.plot(ticks, success_rates, color="#FB8C00", linewidth=2, label="INTERACT success rate")
    ax6.fill_between(ticks, success_rates, alpha=0.2, color="#FB8C00")
    ax6.set_ylim(0, 1.05)
    ax6.set_xlabel("Tick")
    ax6.set_ylabel("Success rate")
    ax6.set_title("Interaction Success Rate")
    ax6.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax6.legend(fontsize=7)

    _save_or_show(fig, out=out, show=show)


# ---------------------------------------------------------------------------
# Comparison overlay (welfare curves for multiple runs)
# ---------------------------------------------------------------------------

COMPARISON_COLORS = [
    "#1565C0", "#B71C1C", "#1B5E20", "#4A148C", "#E65100", "#006064"
]


def plot_comparison(
    logs: list[list[dict]],
    labels: list[str],
    title: str = "",
    out: str | None = None,
    show: bool = False,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title or "Unknown Room — Reward Function Comparison",
                 fontsize=13, fontweight="bold")
    plt.subplots_adjust(wspace=0.3, top=0.88)

    ax_w, ax_l = axes

    for i, (log, label) in enumerate(zip(logs, labels)):
        s = extract_series(log)
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        ax_w.plot(s["ticks"], s["welfare"], color=color, linewidth=2, label=label)
        ax_l.plot(s["ticks"], s["living"], color=color, linewidth=2, label=label)

    ax_w.set_ylim(0, 1.05)
    ax_w.set_xlabel("Tick")
    ax_w.set_ylabel("Collective welfare")
    ax_w.set_title("Collective Welfare")
    ax_w.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_w.legend(fontsize=9)

    ax_l.set_xlabel("Tick")
    ax_l.set_ylabel("Living agents")
    ax_l.set_title("Agent Survival")
    ax_l.legend(fontsize=9)

    _save_or_show(fig, out=out, show=show)


def _save_or_show(fig, *_, out=None, show=False):
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    if not out and not show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize Unknown Room log(s).")
    parser.add_argument("logs", nargs="+", help="One or more JSON log files.")
    parser.add_argument("--labels", nargs="*", help="Labels for comparison mode.")
    parser.add_argument("--title", default="", help="Figure title.")
    parser.add_argument("--out", default=None, help="Save path (e.g. figures/ep.png).")
    parser.add_argument("--show", action="store_true", help="Display interactively.")
    args = parser.parse_args()

    loaded = [load_log(p) for p in args.logs]
    labels = args.labels or [Path(p).stem for p in args.logs]

    if len(loaded) == 1:
        plot_dashboard(loaded[0], title=args.title or labels[0],
                       out=args.out, show=args.show)
    else:
        plot_comparison(loaded, labels, title=args.title,
                        out=args.out, show=args.show)


if __name__ == "__main__":
    main()
