from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import style as ms

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("04_machine_learning", "tables")
BENCH = TABLES / "tuned_benchmark.xlsx"

TASK_TITLE = {"printability": "Printability",
              "cell_response": "Cell Response",
              "cell_response_cellular": "Cell Response, cellular only"}
CLASS_META = {
    "printability": {0: "not extruded", 1: "liquid / beads",
                     2: "extrudable", 3: "optimised"},
    "cell_response": {1: "no cells", 2: "inadequate", 3: "good short",
                      4: "good short, poor long", 5: "good short + long"},
    "cell_response_cellular": {2: "inadequate", 3: "good short",
                               4: "good short, poor long",
                               5: "good short + long"},
}
METRICS = ("precision", "recall", "f1")


def per_class_panel(ax, block: pd.DataFrame, task: str, title: str,
                    show_labels: bool = True):
    labels = block["class"].tolist()
    y = np.arange(len(labels))
    width = 0.26
    for k, metric in enumerate(METRICS):
        vals = block[metric].to_numpy(dtype=float)
        ax.barh(y + (1 - k) * width, vals, height=width,
                color=ms.CATEGORICAL[k], label=metric.capitalize(),
                zorder=3)
        for i, v in enumerate(vals):
            if v <= 0:
                ax.annotate("0.00", (0.0, y[i] + (1 - k) * width),
                            xytext=(3, 0), textcoords="offset points",
                            size=4.8, color=ms.CATEGORICAL[k], va="center",
                            ha="left", zorder=5)
    for i, (_, r) in enumerate(block.iterrows()):
        ax.annotate(f"n = {int(r['support'])}", (0.985, y[i]),
                    xytext=(0, 0), textcoords="offset points", size=5.2,
                    color=ms.MUTED, va="center", ha="right",
                    path_effects=ms.halo(2.0))
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{c} · {CLASS_META[task].get(c, '')}" for c in labels]
        if show_labels else [], size=6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("score", size=7)
    ax.set_title(title, size=6.8, color=ms.MUTED, pad=5)
    ms.grid_axis(ax, "x")
    ms.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)


def confusion_panel(ax, block: pd.DataFrame, task: str, title: str):
    cm = block.set_index("truth")
    cm = cm[[c for c in cm.columns if str(c).startswith("pred_")]]
    cm = cm.dropna(axis=1, how="all")
    cm = cm.loc[:, ~(cm.isna().all(axis=0))]
    cm = cm[cm.notna().any(axis=1)]
    counts = cm.to_numpy(dtype=float)
    counts = np.nan_to_num(counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = counts / counts.sum(axis=1, keepdims=True)
    ax.imshow(np.nan_to_num(frac), cmap=ms.SEQ_NAVY, vmin=0, vmax=1,
              aspect="auto", origin="upper", interpolation="nearest")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = frac[i, j]
            ax.text(j, i, f"{int(counts[i, j])}", ha="center", va="center",
                    size=5.6,
                    color="white" if (np.isfinite(v) and v > .55) else ms.TEXT)
    ticks = [str(c).replace("pred_", "") for c in cm.columns]
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, size=6)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels([str(t).replace("true_", "") for t in cm.index], size=6)
    ax.set_xlabel("predicted", size=7)
    ax.set_ylabel("true", size=7)
    ax.set_title(title, size=6.8, color=ms.MUTED, pad=5)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="tuned_benchmark.xlsx")
    ap.add_argument("--tasks", nargs="*",
                    default=["printability", "cell_response"])
    args = ap.parse_args()

    ms.apply()
    bench = TABLES / args.bench
    if not bench.exists():
        raise SystemExit(f"{bench} not found; run report.py first")
    pc = pd.read_excel(bench, sheet_name="per_class")
    cm = pd.read_excel(bench, sheet_name="confusion")
    board = pd.read_excel(bench, sheet_name="test")

    for task in args.tasks:
        cell = board[(board["task"] == task) & (board["protocol"] == "random")]
        cell = cell[~cell["model"].str.startswith("Dummy")]
        if cell.empty:
            continue
        winner = cell.nlargest(1, "weighted_f1")["model"].iloc[0]

        fig = ms.plt.figure(figsize=(ms.WIDTHS["double"], 4.6))
        gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                              left=0.17, right=0.97, top=0.86, bottom=0.11)
        for k, proto in enumerate(("random", "doi")):
            blk = pc[(pc["task"] == task) & (pc["protocol"] == proto)
                     & (pc["model"] == winner)]
            axp = fig.add_subplot(gs[0, k])
            if len(blk):
                per_class_panel(axp, blk, task,
                                f"per class - {ms.PROTOCOL_LABELS[proto]}",
                                show_labels=(k == 0))
            else:
                axp.set_axis_off()
            ms.panel_tag(axp, "AB"[k], dx=-0.30, dy=1.10)

            blk = cm[(cm["task"] == task) & (cm["protocol"] == proto)
                     & (cm["model"] == winner)]
            axc = fig.add_subplot(gs[1, k])
            if len(blk):
                confusion_panel(axc, blk, task,
                                f"confusion - {ms.PROTOCOL_LABELS[proto]}")
            else:
                axc.set_axis_off()
            ms.panel_tag(axc, "CD"[k], dx=-0.30, dy=1.10)

        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
                   fontsize=6.4, handletextpad=0.4, columnspacing=1.6)
        fig.suptitle(f"{TASK_TITLE[task]} - {ms.short(winner)}", size=9.5,
                     y=0.985, va="top")
        fig.text(0.5, 0.925, "leading model on the random split; the same "
                 "model scored under both protocols", ha="center",
                 va="top", size=6.4, color=ms.MUTED)
        paths = ms.save(fig, f"fig6_model_detail_{task}",
                        step="04_machine_learning")
        print(f"  {task:16s} {winner:28s} -> "
              f"{', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
