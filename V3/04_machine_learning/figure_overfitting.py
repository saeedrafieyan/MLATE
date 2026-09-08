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
PRIMARY = ("weighted", "untuned", "composed")
TASK_TITLE = {"printability": "Printability",
              "cell_response": "Cell Response",
              "cell_response_cellular": "Cell Response, cellular only"}


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
    gap = pd.read_excel(bench, sheet_name="overfitting")
    gap = gap[gap["selection"].isin(PRIMARY)]
    board = pd.read_excel(bench, sheet_name="leaderboard")
    fam = (board.drop_duplicates("model").set_index("model")["family"]
           if "family" in board.columns else None)

    for task in args.tasks:
        cell = gap[(gap["task"] == task)
                   & (~gap["model"].str.startswith("Dummy"))]
        if cell.empty:
            continue
        wide = cell.pivot_table(index="model", columns="protocol",
                                values="weighted_f1_gap")
        order = wide.sort_values("doi", ascending=True).index.tolist()
        n = len(order)

        fig = ms.plt.figure(figsize=(ms.WIDTHS["onehalf"], n * 0.175 + 1.5))
        ax = fig.add_axes([0.34, 0.115, 0.635, 0.80])
        y = np.arange(n)
        h = 0.38
        for k, proto in enumerate(("random", "doi")):
            if proto not in wide.columns:
                continue
            ax.barh(y + (0.5 - k) * h, wide.reindex(order)[proto], height=h,
                    color=ms.PROTOCOL_COLORS[proto], zorder=3,
                    label=ms.PROTOCOL_LABELS[proto])
        ax.axvline(0, color=ms.TEXT, lw=0.8, zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels([ms.short(m) for m in order], size=6)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlabel("weighted F1  (train - test)", size=7)
        ms.grid_axis(ax, "x")
        ms.despine(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0, pad=11 if fam is not None else 2)

        if fam is not None:
            rug = ax.inset_axes([-0.028, 0.0, 0.017, 1.0],
                                transform=ax.transAxes)
            for i, m in enumerate(order):
                rug.add_patch(ms.plt.Rectangle(
                    (0, i - 0.5), 1, 1,
                    facecolor=ms.FAMILY_COLORS.get(fam.get(m), ms.SLATE),
                    lw=0))
            rug.set_xlim(0, 1)
            rug.set_ylim(-0.7, n - 0.3)
            rug.set_xticks([])
            rug.set_yticks([])
            rug.grid(False)
            for side in ("top", "right", "left", "bottom"):
                rug.spines[side].set_visible(False)

        ax.legend(loc="lower right", frameon=False, fontsize=6.4,
                  handletextpad=0.5)
        fig.suptitle(f"{TASK_TITLE[task]} - overfitting diagnostic",
                     size=9.5, y=0.995, va="top")
        fig.text(0.5, 0.962,
                 "a large bar means the model fits its training partition far "
                 "better than the held-out one;\nunder study grouping that "
                 "structure is the publication, not the formulation",
                 ha="center", va="top", size=6.2, color=ms.MUTED)
        paths = ms.save(fig, f"figS13_overfitting_{task}",
                        step="04_machine_learning")
        print(f"  {task:16s} {n:2d} models -> "
              f"{', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
